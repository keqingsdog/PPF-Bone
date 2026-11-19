"""Observe exceptions in one explicitly launched private test executable.

No attach, process enumeration, or product build. Example:
python diagnose_windows.py <private-test.exe> --output <new-result.json>
"""
import argparse
import ctypes as c
from ctypes import wintypes as w
import json
from pathlib import Path
import struct
import subprocess
import time

UP = c.c_size_t


class ExceptionRecord(c.Structure):
    _fields_ = [("code", w.DWORD), ("flags", w.DWORD), ("record", c.c_void_p),
                ("address", c.c_void_p), ("count", w.DWORD), ("info", UP * 15)]


class ExceptionInfo(c.Structure):
    _fields_ = [("record", ExceptionRecord), ("first", w.DWORD)]


class LoadInfo(c.Structure):
    _fields_ = [("file", w.HANDLE), ("base", c.c_void_p), ("offset", w.DWORD),
                ("size", w.DWORD), ("name", c.c_void_p), ("unicode", w.WORD)]


class CreateInfo(c.Structure):
    _fields_ = [("file", w.HANDLE), ("process", w.HANDLE), ("thread", w.HANDLE),
                ("base", c.c_void_p), ("offset", w.DWORD), ("size", w.DWORD),
                ("tls", c.c_void_p), ("start", c.c_void_p), ("name", c.c_void_p),
                ("unicode", w.WORD)]


class Payload(c.Union):
    _fields_ = [("exception", ExceptionInfo), ("create", CreateInfo),
                ("load", LoadInfo), ("exit_code", w.DWORD), ("raw", c.c_byte * 160)]


class Event(c.Structure):
    _fields_ = [("code", w.DWORD), ("pid", w.DWORD), ("tid", w.DWORD), ("data", Payload)]


class Startup(c.Structure):
    _fields_ = [("cb", w.DWORD), ("reserved", w.LPWSTR), ("desktop", w.LPWSTR),
                ("title", w.LPWSTR), ("x", w.DWORD), ("y", w.DWORD),
                ("cx", w.DWORD), ("cy", w.DWORD), ("xc", w.DWORD), ("yc", w.DWORD),
                ("fill", w.DWORD), ("flags", w.DWORD), ("show", w.WORD),
                ("reserved2", w.WORD), ("reserved3", c.POINTER(c.c_byte)),
                ("stdin", w.HANDLE), ("stdout", w.HANDLE), ("stderr", w.HANDLE)]


class Process(c.Structure):
    _fields_ = [("process", w.HANDLE), ("thread", w.HANDLE), ("pid", w.DWORD), ("tid", w.DWORD)]


def run(executable, output):
    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.CreateProcessW.argtypes = [w.LPCWSTR, w.LPWSTR, c.c_void_p, c.c_void_p,
                                    w.BOOL, w.DWORD, c.c_void_p, w.LPCWSTR,
                                    c.POINTER(Startup), c.POINTER(Process)]
    kernel.OpenThread.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenThread.restype = w.HANDLE
    kernel.GetThreadContext.argtypes = [w.HANDLE, c.c_void_p]
    kernel.ReadProcessMemory.argtypes = [w.HANDLE, c.c_void_p, c.c_void_p, UP, c.POINTER(UP)]
    kernel.WaitForDebugEvent.argtypes = [c.POINTER(Event), w.DWORD]
    kernel.ContinueDebugEvent.argtypes = [w.DWORD, w.DWORD, w.DWORD]
    kernel.TerminateProcess.argtypes = [w.HANDLE, w.UINT]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    startup = Startup(); startup.cb = c.sizeof(Startup)
    process = Process()
    result = {"executable": str(executable), "events": [], "modules": []}
    cmd = c.create_unicode_buffer(subprocess.list2cmdline([str(executable)]))
    ok = kernel.CreateProcessW(str(executable), cmd, None, None, False, 0x08000002,
                               None, str(executable.parent), c.byref(startup), c.byref(process))
    if not ok:
        raise c.WinError(c.get_last_error())
    deadline = time.monotonic() + 30
    def memory(address, size):
        buf = c.create_string_buffer(size); count = UP()
        if not kernel.ReadProcessMemory(process.process, address, buf, size, c.byref(count)):
            return None
        return bytes(buf[:count.value])
    try:
        while time.monotonic() < deadline:
            event = Event()
            if not kernel.WaitForDebugEvent(c.byref(event), 1000):
                continue
            if event.code == 3:
                result["image_base"] = hex(event.data.create.base or 0)
                if event.data.create.file: kernel.CloseHandle(event.data.create.file)
            elif event.code == 6:
                result["modules"].append(hex(event.data.load.base or 0))
                if event.data.load.file: kernel.CloseHandle(event.data.load.file)
            elif event.code == 1:
                exc = event.data.exception
                item = {"code": hex(exc.record.code), "address": hex(exc.record.address or 0),
                        "first_chance": bool(exc.first), "tid": event.tid,
                        "parameters": [hex(v) for v in exc.record.info[:exc.record.count]]}
                result["events"].append(item)
                if exc.record.code != 0x80000003:
                    thread = kernel.OpenThread(0x0048, False, event.tid)
                    buf = c.create_string_buffer(1250)
                    aligned = (c.addressof(buf) + 15) & ~15
                    c.c_uint32.from_address(aligned + 48).value = 0x00100003
                    if thread and kernel.GetThreadContext(thread, aligned):
                        registers = {}
                        for name, offset in [("rax",120),("rcx",128),("rdx",136),("rbx",144),
                                             ("rsp",152),("rbp",160),("rsi",168),("rdi",176),
                                             ("r8",184),("r9",192),("r10",200),("r11",208),
                                             ("r12",216),("r13",224),("r14",232),("r15",240),("rip",248)]:
                            registers[name] = hex(c.c_uint64.from_address(aligned + offset).value)
                        item["registers"] = registers
                        stack = memory(int(registers["rsp"],16),512)
                        item["stack_qwords"] = [hex(v) for v in struct.unpack("<"+"Q"*(len(stack)//8), stack)] if stack else None
                        rip = int(registers["rip"],16)
                        code = memory(rip-32,96)
                        item["code_address"] = hex(rip-32)
                        item["code_bytes"] = code.hex() if code else None
                    else:
                        item["context_error"] = c.get_last_error()
                    if thread: kernel.CloseHandle(thread)
                    # Stop only the private process created above.
                    kernel.TerminateProcess(process.process, 0xDEAD)
                    kernel.ContinueDebugEvent(event.pid,event.tid,0x00010002)
                    break
            elif event.code == 5:
                result["exit_code"] = hex(event.data.exit_code)
                kernel.ContinueDebugEvent(event.pid,event.tid,0x00010002)
                break
            kernel.ContinueDebugEvent(event.pid,event.tid,0x00010002)
        else:
            result["timeout"] = True
            kernel.TerminateProcess(process.process, 0xDEAD)
    finally:
        kernel.CloseHandle(process.thread); kernel.CloseHandle(process.process)
        output.write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    run(args.executable.resolve(),args.output.resolve())
