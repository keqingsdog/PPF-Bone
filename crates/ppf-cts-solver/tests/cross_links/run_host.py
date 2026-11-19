"""Direct neutral C++ cross-link math tests, in a C++ developer environment.

Use --out with a NEW private directory. This does not build a native backend.
The optimized host test uses the shared neutral functions. Native product
acceptance remains a separate gate using the real selected backend.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def run(folder, cxx):
    folder.mkdir(parents=True, exist_ok=False)
    crate = Path(__file__).resolve().parents[2]
    source = crate / "tests/kernels/cross_links.cpp"
    include = crate / "src/kernels"
    binary = folder / "cross_links.exe"
    if Path(cxx).stem.lower() in ("cl", "clang-cl"):
        command = [cxx, "/nologo", "/utf-8", "/EHsc", "/std:c++17", "/fp:strict",
                   "/O2", "/I" + str(include), "/Fe:" + str(binary),
                   "/Fo:" + str(folder / "cross_links.obj"), str(source)]
    else:
        command = [cxx, "-std=c++17", "-O2", "-ffp-contract=off", "-Wno-attributes",
                   "-I" + str(include), str(source), "-o", str(binary)]
    built = subprocess.run(command, cwd=folder, capture_output=True, text=True)
    (folder / "compile.log").write_text(built.stdout + built.stderr, encoding="utf8")
    built.check_returncode()
    result = subprocess.run([str(binary)], cwd=folder, capture_output=True, text=True)
    (folder / "run.log").write_text(result.stdout + result.stderr, encoding="utf8")
    print(result.stdout, end="")
    result.check_returncode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "cl" if os.name == "nt" else "c++"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out:
        run(args.out.resolve(), args.cxx)
    else:
        with tempfile.TemporaryDirectory(prefix="cross-links-host-") as parent:
            run(Path(parent) / "run", args.cxx)
