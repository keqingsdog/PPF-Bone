"""Run 22 focused tensor-storage checks in a standard C++ developer shell.

Only --out (which must not exist) receives generated negative-control headers,
objects, executables and logs. Production headers are read-only. This is a host
regression; native backend/device acceptance is separate.
"""
import argparse
import os
import platform
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "cl" if os.name == "nt" else "c++"))
    args = parser.parse_args()
    crate = Path(__file__).resolve().parents[2]
    include = crate / "src/kernels"
    source = crate / "tests/kernels/restoring_tensor_rounding.cpp"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # A deliberately wrong private variant proves the real binding failure is
    # PSD storage rather than an earlier coordinate/scalar/gradient failure.
    header = (include / "energy/model/restoring_stitch.hpp").read_text(encoding="utf8")
    header = header.replace('#include "../../zz_joint_ext/restoring_context.hpp"',
                            '#include "' + (include / "zz_joint_ext/restoring_context.hpp").as_posix() + '"', 1)
    header = header.replace("namespace restoring_stitch {", "namespace restoring_stitch_borrowed_raw {", 1)
    header = header.replace("narrow_hessian_component(projected,projected_scale,",
                            "narrow_hessian_component(projected,raw_scale,", 1)
    variant = out / "borrowed_raw_negative_control.hpp"
    variant.write_text(header, encoding="utf8")
    cxx = shutil.which(args.cxx) or args.cxx
    binary = out / ("restoring_tensor_rounding.exe" if os.name == "nt" else "restoring_tensor_rounding")
    macro = 'PPF_TEST_BORROWED_RAW_HEADER="' + variant.as_posix() + '"'
    x86_64 = platform.machine().lower() in {'amd64', 'x86_64'}
    if Path(cxx).stem.lower() in ("cl", "clang-cl"):
        command = [cxx, "/nologo", "/utf-8", "/EHsc", "/std:c++17", "/O2",
                   "/fp:precise", "/fp:contract", *(["/arch:AVX2"] if x86_64 else []), "/D" + macro,
                   "/I" + str(include), "/Fe:" + str(binary),
                   "/Fo:" + str(out / "restoring_tensor_rounding.obj"), str(source)]
    else:
        command = [cxx, "-std=c++17", "-O2", "-ffp-contract=fast", *(["-mavx2", "-mfma"] if x86_64 else []),
                   "-Wno-attributes", "-D" + macro, "-I" + str(include), str(source), "-o", str(binary)]
    result = subprocess.run(command, cwd=out, capture_output=True, text=True)
    (out / "compile.log").write_text(result.stdout + result.stderr, encoding="utf8")
    result.check_returncode()
    result = subprocess.run([str(binary)], cwd=out, capture_output=True, text=True)
    (out / "run.log").write_text(result.stdout + result.stderr, encoding="utf8")
    print(result.stdout, end="")
    result.check_returncode()


if __name__ == "__main__":
    main()
