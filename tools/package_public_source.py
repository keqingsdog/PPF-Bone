from pathlib import Path, PurePosixPath
import argparse
import json
import os
import shutil
import stat
import subprocess
import zipfile

ROOT = Path(__file__).absolute().parents[1]
MANIFEST = ".ppf-public-source-files.json"
REQUIRED = {
    "LICENSE", "NOTICE_BONE_FORK.md", "Cargo.toml",
    "crates/ppf-cts-compute/src/build/mod.rs",
    "coordination/windows-resume-20261004/restoring-curve/native-scalar/generate_identity.py",
    "tools/package_public_source.py",
}
LOCAL_DIRS = {
    ".git", ".hg", ".svn", "target", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".cache", ".venv", "node_modules", ".claude", ".codex", ".aws", ".ssh",
    ".idea", ".vscode", ".gnupg",
}
LOCAL_NAMES = {
    ".env", ".npmrc", ".pypirc", ".netrc", "credentials.json", "id_rsa", "id_ed25519",
    "local_source_build.json", "local_source_policy.json", ".ds_store",
}
LOCAL_SUFFIXES = {
    ".whl", ".pyc", ".pyo", ".obj", ".o", ".a", ".pdb", ".exe", ".dll", ".lib",
    ".exp", ".so", ".dylib", ".pyd", ".cubin", ".fatbin", ".pem", ".pfx", ".p12",
    ".ppk", ".key", ".pc2", ".blend1", ".blend2", ".blend@", ".log",
}
NATIVE_DIRS = {"build-linux-native", "build-mac-native", "build-win-native"}
NATIVE_LOCAL_DIRS = {
    "dist", "downloads", "python", "python_full", "rust", "deps", "cuda", "rocm",
    "msvc", "msys64", "mingit", "7zip", "ffmpeg", "temp_ffmpeg", "patchelf",
    "nasm", "jupyter", "ppf-cts", "fast_check",
}
MAGIC = (
    b"\x7fELF", b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf", b"\xce\xfa\xed\xfe",
    b"\xcf\xfa\xed\xfe", b"!<arch>\n",
)
REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def valid_rel(name):
    if not isinstance(name, str) or any(c in name for c in "\\:\0\r\n"):
        raise ValueError(f"invalid source path: {name!r}")
    p = PurePosixPath(name)
    if p.is_absolute() or name != p.as_posix() or not p.parts or p == PurePosixPath("."):
        raise ValueError(f"invalid source path: {name!r}")
    if any(part in {".", ".."} for part in p.parts):
        raise ValueError(f"invalid source path: {name!r}")
    return p


def is_local(p):
    parts = tuple(part.lower() for part in p.parts)
    leaf = parts[-1]
    if (any(x in LOCAL_DIRS for x in parts) or leaf in LOCAL_NAMES
            or leaf.startswith(".env.") or p.suffix.lower() in LOCAL_SUFFIXES
            or (parts[0] in NATIVE_DIRS and len(parts) > 1
                and parts[1] in NATIVE_LOCAL_DIRS)):
        return True
    return (
        parts[:2] in {("examples", "cache"), ("calibration", "report_data")}
        or parts[:4] in {
            ("crates", "ppf-cts-compute", backend, "build")
            for backend in ("cpu", "cuda", "metal", "rocm")
        }
        or ("src", "kernels", "build") in tuple(zip(parts, parts[1:], parts[2:]))
    )


def plain_path(root, p):
    here = root
    for i, part in enumerate(p.parts):
        here = here / part
        info = here.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & REPARSE_POINT:
            raise ValueError(f"refusing linked source path: {p}")
        expected = stat.S_ISREG if i == len(p.parts) - 1 else stat.S_ISDIR
        if not expected(info.st_mode):
            raise ValueError(f"not an ordinary source path: {p}")
    return here


def source_paths(root):
    # Do not let Git discover an unrelated repository above an extracted ZIP.
    if os.path.lexists(root / ".git"):
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        raw = subprocess.check_output(["git", "-C", str(root), "ls-files", "--cached", "-z"], env=env)
        names = [os.fsdecode(part) for part in raw.split(b"\0") if part]
    else:
        manifest = plain_path(root, PurePosixPath(MANIFEST))
        with manifest.open(encoding="utf-8") as source:
            data = json.load(source)
        if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("files"), list):
            raise ValueError(f"unrecognized source inventory: {MANIFEST}")
        names = data["files"]
    checked = [str(valid_rel(name)) for name in names]
    if len(checked) != len(set(checked)):
        raise ValueError("source inventory has duplicate paths or unresolved Git index entries")
    candidates = [p for p in map(PurePosixPath, sorted(checked)) if str(p) != MANIFEST and not is_local(p)]
    if len(candidates) != len({str(p).casefold() for p in candidates}):
        raise ValueError("source inventory has paths that collide on case-insensitive filesystems")
    missing = sorted(REQUIRED - {str(p) for p in candidates})
    if missing:
        raise ValueError(f"missing required source inputs: {', '.join(missing)}")
    # Refuse, rather than silently omit, an expected path changed into a link.
    return [(p, plain_path(root, p)) for p in candidates]


def zip_info(relative):
    info = zipfile.ZipInfo(f"PPF_Bone_source/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (0o100755 if PurePosixPath(relative).suffix.lower() == ".sh" else 0o100644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def main():
    parser = argparse.ArgumentParser(description="Package tracked or inventoried public source files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    temp = out.with_name(out.name + ".tmp")
    try:
        if out.is_relative_to(ROOT.resolve()):
            raise ValueError("output must be outside the source tree")
        if os.path.lexists(out) or os.path.lexists(temp):
            raise ValueError("output or temporary output already exists")
        files = source_paths(ROOT)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Source packaging refused: {exc}\n")
    created = False
    try:
        with zipfile.ZipFile(temp, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
            created = True
            inventory = json.dumps({"version": 1, "files": [str(p) for p, _ in files]},
                                   ensure_ascii=True, indent=2) + "\n"
            z.writestr(zip_info(MANIFEST), inventory.encode("utf-8"))
            for p, path in files:
                # Revalidate before each read, and do not follow final-path links
                # even if they replace a file between validation and open.
                plain_path(ROOT, p)
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
                with os.fdopen(os.open(path, flags), "rb") as source:
                    first = source.read(16)
                    if first.startswith(MAGIC):
                        raise ValueError(f"refusing compiled content: {p}")
                    with z.open(zip_info(p), "w", force_zip64=True) as target:
                        target.write(first)
                        shutil.copyfileobj(source, target)
        if os.path.lexists(out):
            raise ValueError("output now exists; refusing overwrite")
        temp.replace(out)
        created = False
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Source packaging refused: {exc}\n")
    finally:
        if created:
            temp.unlink(missing_ok=True)
    print(f"ZIP={out} SIZE_BYTES={out.stat().st_size}")


if __name__ == "__main__":
    main()
