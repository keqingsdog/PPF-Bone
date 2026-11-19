//! Optional clock telemetry, independent of the simulation backend and its ABI.
//!
//! Like the original unfiltered nvidia-smi query, this observes ALL physical
//! NVIDIA devices in index order. The backend does not expose a selected ordinal,
//! so these values must not be described as the selected solver device's clock.

#[cfg(windows)]
mod platform {
    use std::ffi::{c_char, c_void};
    use std::sync::Mutex;

    type Init = unsafe extern "C" fn() -> i32;
    type Shutdown = unsafe extern "C" fn() -> i32;
    type DeviceCount = unsafe extern "C" fn(*mut u32) -> i32;
    type DeviceHandle = unsafe extern "C" fn(u32, *mut *mut c_void) -> i32;
    type DeviceClock = unsafe extern "C" fn(*mut c_void, u32, *mut u32) -> i32;

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
    }

    // Store opaque addresses as integers so the process-wide mutex can own them.
    // No references to driver memory are created; handles are only passed back
    // to NVML while the library is loaded and the mutex is held.
    struct Library(usize);

    impl Library {
        fn open() -> Option<Self> {
            let name: Vec<u16> = "nvml.dll\0".encode_utf16().collect();
            // LOAD_LIBRARY_SEARCH_SYSTEM32: never search cwd/PATH for this DLL.
            let module = unsafe { LoadLibraryExW(name.as_ptr(), std::ptr::null_mut(), 0x800) };
            if module.is_null() { None } else { Some(Self(module as usize)) }
        }

        fn symbol(&self, name: &[u8]) -> Option<*mut c_void> {
            let address = unsafe {
                GetProcAddress(self.0 as *mut c_void, name.as_ptr() as *const c_char)
            };
            if address.is_null() { None } else { Some(address) }
        }
    }

    impl Drop for Library {
        fn drop(&mut self) {
            unsafe { FreeLibrary(self.0 as *mut c_void); }
        }
    }

    struct Nvml {
        _library: Library,
        shutdown: Shutdown,
        clock: DeviceClock,
        devices: Vec<usize>,
    }

    impl Nvml {
        fn open() -> Option<Self> {
            let library = Library::open()?;
            // Symbol names and function pointer signatures are NVML's C API.
            let (init, shutdown, count, handle, clock) = unsafe {
                (
                    std::mem::transmute::<*mut c_void, Init>(library.symbol(b"nvmlInit_v2\0")?),
                    std::mem::transmute::<*mut c_void, Shutdown>(library.symbol(b"nvmlShutdown\0")?),
                    std::mem::transmute::<*mut c_void, DeviceCount>(library.symbol(b"nvmlDeviceGetCount_v2\0")?),
                    std::mem::transmute::<*mut c_void, DeviceHandle>(library.symbol(b"nvmlDeviceGetHandleByIndex_v2\0")?),
                    std::mem::transmute::<*mut c_void, DeviceClock>(library.symbol(b"nvmlDeviceGetClockInfo\0")?),
                )
            };
            if unsafe { init() } != 0 { return None; }
            // Construct the owner immediately so every subsequent failure calls
            // nvmlShutdown before unloading the library.
            let mut nvml = Self { _library: library, shutdown, clock, devices: Vec::new() };
            let mut device_count = 0;
            if unsafe { count(&mut device_count) } != 0 || device_count == 0 { return None; }
            for index in 0..device_count {
                let mut device = std::ptr::null_mut();
                if unsafe { handle(index, &mut device) } != 0 || device.is_null() { return None; }
                nvml.devices.push(device as usize);
            }
            Some(nvml)
        }

        fn sample(&self) -> Option<String> {
            let mut values = Vec::with_capacity(self.devices.len());
            for &device in &self.devices {
                let mut mhz = 0;
                // NVML_CLOCK_SM = 1. Query every device freshly; if any fails,
                // omit the entire sample rather than silently shifting indices.
                if unsafe { (self.clock)(device as *mut c_void, 1, &mut mhz) } != 0 {
                    return None;
                }
                values.push(mhz.to_string());
            }
            // Preserve the previous multi-GPU nvidia-smi stdout representation.
            Some(values.join("\n"))
        }
    }

    impl Drop for Nvml {
        fn drop(&mut self) {
            unsafe { (self.shutdown)(); }
            // _library is dropped after this body, never before shutdown.
        }
    }

    struct Shared {
        users: usize,
        nvml: Option<Nvml>,
    }

    // Failed initialization is cached for this cohort as well: unavailable
    // telemetry never turns into repeated load/init attempts on physics steps.
    static STATE: Mutex<Option<Shared>> = Mutex::new(None);

    pub struct Sampler {
        registered: bool,
    }

    impl Sampler {
        pub fn new() -> Self {
            let Ok(mut state) = STATE.lock() else { return Self { registered: false }; };
            let shared = state.get_or_insert_with(|| Shared { users: 0, nvml: Nvml::open() });
            shared.users += 1;
            Self { registered: true }
        }

        pub fn sample(&mut self) -> Option<String> {
            if !self.registered { return None; }
            let state = STATE.lock().ok()?;
            state.as_ref()?.nvml.as_ref()?.sample()
        }
    }

    impl Drop for Sampler {
        fn drop(&mut self) {
            if !self.registered { return; }
            if let Ok(mut state) = STATE.lock() {
                if let Some(shared) = state.as_mut() {
                    shared.users -= 1;
                    if shared.users == 0 {
                        // Shutdown/free while holding the same mutex as init;
                        // a concurrent new run cannot race the final shutdown.
                        *state = None;
                    }
                }
            }
        }
    }
}

#[cfg(not(windows))]
mod platform {
    use std::time::{Duration, Instant};

    pub struct Sampler {
        last_attempt: Option<Instant>,
    }

    impl Sampler {
        pub fn new() -> Self { Self { last_attempt: None } }

        pub fn sample(&mut self) -> Option<String> {
            let now = Instant::now();
            if self.last_attempt.is_some_and(|last| now.duration_since(last) < Duration::from_secs(1)) {
                return None;
            }
            self.last_attempt = Some(now);
            let output = std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=clocks.current.sm")
                .arg("--format=csv,noheader,nounits")
                .output().ok()?;
            if !output.status.success() { return None; }
            let text = String::from_utf8(output.stdout).ok()?;
            let values: Option<Vec<u32>> = text.lines().map(|line| line.trim().parse().ok()).collect();
            let values = values?;
            if values.is_empty() { return None; }
            Some(values.iter().map(u32::to_string).collect::<Vec<_>>().join("\n"))
        }
    }
}

pub use platform::Sampler;
