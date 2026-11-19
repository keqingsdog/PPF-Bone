"""Package a source tree without dropping build inputs with generic names.

Generated locations are excluded by exact root-relative path. No directory is
excluded merely because its basename is ``coordination``, ``build`` or ``cache``.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import stat
import zipfile

ROOT_EXCLUDED = {
    'target', 'local', 'cache',
    'crates/ppf-cts-compute/cpu/build',
    'crates/ppf-cts-compute/cuda/build',
    'crates/ppf-cts-compute/metal/build',
    'crates/ppf-cts-compute/rocm/build',
    'crates/ppf-cts-solver/src/kernels/build',
}
METADATA_DIRS = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache'}
METADATA_SUFFIXES = {'.pyc', '.pyo'}


def _reparse(path: Path) -> bool:
    info = path.stat(follow_symlinks=False)
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def package_source(root: Path, output: Path, prefix: str) -> int:
    root = root.absolute()
    if not root.is_dir() or _reparse(root):
        raise ValueError('source root must be a real directory')
    root = root.resolve(strict=True)
    output = output.absolute()
    if output.resolve().is_relative_to(root):
        raise ValueError('archive must be outside the source tree')
    if not prefix or prefix in {'.', '..'} or '/' in prefix or '\\' in prefix or ':' in prefix:
        raise ValueError('prefix must be one safe directory name')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'archive already exists: {output}')
    # Keep failed writes away from the final delivery name.  A leftover
    # incomplete file is visible and never mistaken for the finished archive.
    incomplete = output.with_name(output.name + '.incomplete')
    if incomplete.exists():
        raise FileExistsError(f'incomplete archive already exists: {incomplete}')
    files_to_add = []
    for base, dirs, files in os.walk(root, followlinks=False):
        folder = Path(base)
        retained = []
        for name in sorted(dirs):
            path = folder / name
            if _reparse(path):
                raise ValueError(f'reparse directory refused: {path}')
            rel = path.relative_to(root).as_posix()
            if name not in METADATA_DIRS and rel not in ROOT_EXCLUDED:
                retained.append(name)
        dirs[:] = retained
        for name in sorted(files):
            path = folder / name
            if _reparse(path):
                raise ValueError(f'reparse file refused: {path}')
            if path.suffix.lower() not in METADATA_SUFFIXES:
                files_to_add.append(path)
    with zipfile.ZipFile(incomplete, 'x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True, strict_timestamps=False) as archive:
        for path in files_to_add:
            archive.write(path, f'{prefix}/{path.relative_to(root).as_posix()}')
    if os.name == 'nt':
        os.rename(incomplete, output)  # Windows refuses an existing destination.
    else:
        os.link(incomplete, output)  # Atomic create; fails if the destination exists.
        os.unlink(incomplete)  # Remove only the incomplete file created above.
    count = len(files_to_add)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix', default='ppf-contact-solver-bones')
    args = parser.parse_args()
    count = package_source(args.root, args.output, args.prefix)
    print(f'SOURCE_PACKAGE_WRITTEN files={count} path={args.output}')


if __name__ == '__main__':
    main()
