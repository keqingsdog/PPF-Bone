#!/usr/bin/env python3
"""Build a Blender extension for one Python ABI using official cbor2 wheels."""
from pathlib import Path
import argparse
import importlib.util
import re
import shutil
import subprocess
import tempfile
import os
import stat
import zipfile

ADDON = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('ppf_wheel_fetch', ADDON / 'wheels' / 'fetch.py')
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def stage(target, abi):
    if target.exists():
        raise SystemExit(f'Staging directory already exists: {target}')
    shutil.copytree(ADDON, target, ignore=shutil.ignore_patterns('*.whl', '__pycache__', '*.pyc'))
    wheels = target / 'wheels'
    selected = [row for row in fetch.WHEELS if abi == 'universal' or f'-{abi}-{abi}-' in row[0]]
    if len(selected) != (10 if abi == 'universal' else 5):
        raise RuntimeError(f'Expected five official wheels for {abi}')
    for row in selected:
        fetch.fetch_one(*row, dest=wheels)
    if abi == 'universal':
        for row in selected:
            name = row[0]
            parts = name[:-4].split('-')
            destination = target / '_vendor' / parts[2] / parts[4]
            destination.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(wheels / name) as wheel:
                wheel.extractall(destination)
            (wheels / name).unlink()
    manifest = target / 'blender_manifest.toml'
    text = manifest.read_text(encoding='utf-8')
    block = 'wheels = [\n' + ''.join(f'  "./wheels/{row[0]}",\n' for row in selected) + ']\n'
    if abi == 'universal':
        block = ''
    if text.count('# PPF_WHEELS_BUILD_PLACEHOLDER') != 1:
        raise RuntimeError('Expected exactly one wheels placeholder in source manifest')
    text = text.replace('# PPF_WHEELS_BUILD_PLACEHOLDER', block)
    # Official Blender moved from CPython 3.11 to 3.13 in Blender 5.1.
    # Make Install from Disk reject the wrong archive before replacement.
    watched = r'(?m)^blender_version_min = "[^"]+"$'
    if abi != 'universal':
        if abi == 'cp311':
            text, replaced = re.subn(watched, lambda m: m.group() + '\nblender_version_max = "5.1.0"', text)
        else:
            text, replaced = re.subn(watched, 'blender_version_min = "5.1.0"', text)
        if replaced != 1:
            raise RuntimeError('Expected one minimum Blender version in the source manifest')
    manifest.write_text(text, encoding='utf-8')


def mark_shell_executable(archive):
    """Preserve file contents while declaring Unix mode for shipped shell entry points."""
    names = {'launch.sh', 'capture.sh'}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=archive.parent, suffix='.zip', delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(archive, 'r') as source, zipfile.ZipFile(temporary, 'w') as dest:
            dest.comment = source.comment
            found = set()
            for info in source.infolist():
                if info.filename in names:
                    if info.filename in found:
                        raise ValueError(f'duplicate shell entry: {info.filename}')
                    found.add(info.filename)
                    info.create_system = 3
                    info.external_attr = (stat.S_IFREG | 0o755) << 16
                dest.writestr(info, source.read(info))
            if found != names:
                raise ValueError(f'missing shell entry: {names - found}')
        os.replace(temporary, archive)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--abi', choices=('cp311','cp313','universal'), help='Select ABI, or detect it from --blender')
    parser.add_argument('--blender', type=Path, help='Blender executable for extension build')
    parser.add_argument('--output', type=Path, help='Extension ZIP output path')
    parser.add_argument('--stage-dir', type=Path, help='Keep prepared extension directory instead of ZIP')
    args = parser.parse_args()
    if bool(args.stage_dir) == bool(args.output):
        parser.error('select exactly one of --stage-dir or --output')
    if args.output and not args.blender:
        parser.error('--output requires --blender')
    if not args.abi and not args.blender:
        parser.error('select --abi or --blender to detect it')
    if args.stage_dir and not args.blender:
        stage(args.stage_dir.resolve(), args.abi)
        return

    with tempfile.TemporaryDirectory(prefix='ppf-addon-build-') as tmp:
        # Factory startup still reconciles installed wheels. Give package
        # commands their own profile so the user's live packages stay in use.
        env = {key: value for key, value in os.environ.items()
               if not key.upper().startswith('BLENDER_USER_')}
        env['BLENDER_USER_RESOURCES'] = str(Path(tmp) / 'blender-profile')
        blender = [str(args.blender), '--factory-startup']
        result = subprocess.run(
            blender + ['--background', '--python-expr',
                       'import sys; print("PPF_TARGET_ABI=cp%d%d" % sys.version_info[:2])'],
            capture_output=True, text=True, check=True, env=env,
        )
        match = re.search(r'PPF_TARGET_ABI=(cp\d+)', result.stdout)
        if not match or match.group(1) not in ('cp311', 'cp313'):
            raise SystemExit('Unsupported target Blender Python ABI (expected cp311/cp313)')
        abi = match.group(1)
        if args.abi and args.abi != 'universal' and abi != args.abi:
            raise SystemExit(f'Target Blender Python ABI does not match requested {args.abi}')
        abi = args.abi or abi
        if args.stage_dir:
            stage(args.stage_dir.resolve(), abi)
            return
        args.output.parent.mkdir(parents=True, exist_ok=True)
        staged = Path(tmp) / 'addon'
        stage(staged, abi)
        archive = str(args.output.resolve())
        subprocess.run(blender + ['--command', 'extension', 'build',
            '--source-dir', str(staged), '--output-filepath', archive], check=True, env=env)
        mark_shell_executable(args.output.resolve())
        subprocess.run(blender + ['--command', 'extension', 'validate', archive], check=True, env=env)

if __name__ == '__main__':
    main()
