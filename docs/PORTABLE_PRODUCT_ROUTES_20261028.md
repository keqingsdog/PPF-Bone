# Portable source and assembled runtime routes

Ordinary frontend/service scenes do not require a historical product profile. The service build worker passes no contract when none is selected; real extension payloads still request their actual contract/capability at decode. Invalid explicit product selections remain errors.

An assembled source runtime can create a relative v2 locator with:

    python tools/configure_source_product.py --relative-root RUNTIME_ROOT --backend cpu --python-relative python/python.exe

Choose the actual assembled Python entry and backend (cpu/cuda/metal/rocm). Optional --build-relative (default target/BACKEND/release), --bin-relative (default bin), and --backend-library-relative select existing files inside the same runtime root. CPU uses the selected native Python binding as its library identity; GPU routes use their actual platform library names. Existing configs are refused instead of silently overwritten. This writes a locator only: extension Build obtains support from real executable capabilities. It does not download Python, build artifacts or claim a platform is validated.

Windows uses .dll and command.bat; Linux uses .so and command.sh; macOS uses .dylib and command.sh with LF. Historical absolute CUDA v1 configuration remains an explicitly selected compatibility route. Ordinary source discovery is separate from those historical profiles. POSIX runtime packagers now include the two pure FEM schema helpers used by frontend imports, without bundling all Blender code.

Validated on Windows CPU: real relative configuration/backend check; ordinary real Blender-encoded Cube through build_worker main without product configuration, generated native command completed0.31 simulation seconds with ten finite frames; missing-config extension and invalid-root rejection; real configured CPU capability query. Linux/macOS/Metal/ROCm paths are source-level corrections and host unit cases, not native or packaging execution on those platforms. Bundle scripts do not yet automatically invoke the new relative configurator; use the explicit tool after assembling the runtime. See BUILDING_BONES.md for normal compiler entry points.
