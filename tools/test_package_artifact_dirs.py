"""Focused checks for backend group selection and stale-solver preflight."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main, mock
import subprocess
import importlib.util

spec = importlib.util.spec_from_file_location('package_windows_runtime', Path(__file__).with_name('package_windows_runtime.py'))
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class ArtifactGroupTests(TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def group(self, directory, marker):
        directory.mkdir(parents=True)
        for name in package.ARTIFACTS:
            (directory / name).write_text(marker if name == '.ppf-backend' else 'test')
        return directory

    def test_explicit_groups_are_selected_together(self):
        source = self.root / 'source'
        self.group(source / 'target/cpu/release', 'cpu')
        cpu = self.group(self.root / 'actual-cpu', 'cpu')
        cuda = self.group(self.root / 'actual-cuda', 'cuda')
        groups = package.artifact_groups(source, ('cpu', 'cuda'), cpu, cuda)
        self.assertEqual(groups, {'cpu': cpu.resolve(), 'cuda': cuda.resolve()})

    def test_unspecified_group_keeps_default_release_directory(self):
        source = self.root / 'source'
        cpu = self.group(source / 'target/cpu/release', 'cpu')
        self.assertEqual(package.artifact_groups(source, ('cpu',)), {'cpu': cpu.resolve()})

    def test_incomplete_explicit_group_does_not_fallback(self):
        source = self.root / 'source'
        self.group(source / 'target/cpu/release', 'cpu')
        cpu = self.group(self.root / 'actual-cpu', 'cpu')
        (cpu / '_ppf_cts_py.dll').unlink()
        with self.assertRaisesRegex(SystemExit, 'Required runtime input missing'):
            package.artifact_groups(source, ('cpu',), cpu)

    def test_wrong_backend_marker_is_rejected(self):
        cpu = self.group(self.root / 'actual-cpu', 'cuda')
        with self.assertRaisesRegex(SystemExit, 'Backend marker mismatch'):
            package.artifact_groups(self.root, ('cpu',), cpu)

    def test_probe_requires_exact_capability_and_success(self):
        solver = self.root / 'ppf-contact-solver.exe'
        outcomes = [(2, '', 'unknown option'), (0, 'typed_b32_v0\n', ''),
                    (0, 'typed_b32_v1\nextra\n', '')]
        for returncode, stdout, stderr in outcomes:
            with self.subTest(returncode=returncode, stdout=stdout):
                result = subprocess.CompletedProcess([], returncode, stdout, stderr)
                with mock.patch.object(package.subprocess, 'run', return_value=result):
                    with self.assertRaisesRegex(SystemExit, 'capability probe rejected'):
                        package.require_fem_retry(solver, {})
        success = subprocess.CompletedProcess([], 0, 'typed_b32_v1\n', '')
        with mock.patch.object(package.subprocess, 'run', return_value=success) as run:
            package.require_fem_retry(solver, {})
        self.assertEqual(run.call_args.args[0], [str(solver), '--probe-fem-retry'])

    def test_old_solver_rejected_before_output_creation(self):
        source = self.root / 'source'
        cpu = self.group(self.root / 'actual-cpu', 'cpu')
        for path in ('blender_addon/core/fem_scale.py', 'blender_addon/core/fem_body_control_events.py',
                     'LICENSE', 'NOTICE_BONE_FORK.md', 'README_BONE_FORK.md', 'BUILDING_BONES.md', 'pyproject.toml'):
            file = source / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('test')
        for path in ('crates/ppf-cts-core/src', 'crates/ppf-cts-solver/src'):
            (source / path).mkdir(parents=True)
        runtime, mingit, redist = self.root / 'runtime', self.root / 'mingit', self.root / 'redist'
        for file in (runtime / 'python/python.exe', runtime / 'bin/ffmpeg.exe', mingit / 'cmd/git.exe',
                     *(redist / name for name in ('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll'))):
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('test')
        output = self.root / 'new-output'
        argv = ['package_windows_runtime.py', '--source-root', str(source), '--third-party-runtime', str(runtime),
                '--mingit', str(mingit), '--msvc-redist', str(redist), '--backend', 'cpu',
                '--cpu-artifact-dir', str(cpu), '--output', str(output)]
        stale = subprocess.CompletedProcess([], 2, '', 'unknown option')
        with mock.patch('sys.argv', argv), mock.patch.object(package.subprocess, 'run', return_value=stale):
            with self.assertRaisesRegex(SystemExit, 'capability probe rejected'):
                package.main()
        self.assertFalse(output.exists())


    def test_ffmpeg_is_opt_in_and_missing_explicit_input_fails_early(self):
        source = self.root / 'source'
        cpu = self.group(self.root / 'cpu', 'cpu')
        for path in ('blender_addon/core/fem_scale.py', 'blender_addon/core/fem_body_control_events.py',
                     'LICENSE', 'NOTICE_BONE_FORK.md', 'README_BONE_FORK.md', 'BUILDING_BONES.md', 'pyproject.toml'):
            file = source / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('test')
        for path in ('crates/ppf-cts-core/src', 'crates/ppf-cts-solver/src'):
            (source / path).mkdir(parents=True)
        runtime, mingit, redist = self.root / 'runtime', self.root / 'mingit', self.root / 'redist'
        for file in (runtime / 'python/python.exe', mingit / 'cmd/git.exe',
                     *(redist / name for name in ('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll'))):
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('test')
        def run(args, **kwargs):
            value = 'typed_b32_v1' if '--probe-fem-retry' in args else ('cpu' if '--backend' in args else 'ok')
            return subprocess.CompletedProcess(args, 0, value + '\n', '')
        base = ['package_windows_runtime.py', '--source-root', str(source), '--third-party-runtime', str(runtime),
                '--mingit', str(mingit), '--msvc-redist', str(redist), '--backend', 'cpu', '--cpu-artifact-dir', str(cpu)]
        default = self.root / 'default'
        with mock.patch('sys.argv', base + ['--output', str(default)]), mock.patch.object(package.subprocess, 'run', side_effect=run):
            package.main()
        self.assertFalse((default / 'bin/ffmpeg.exe').exists())
        missing = self.root / 'missing'
        with mock.patch('sys.argv', base + ['--include-ffmpeg', '--output', str(missing)]):
            with self.assertRaisesRegex(SystemExit, 'ffmpeg.exe'):
                package.main()
        self.assertFalse(missing.exists())
        ffmpeg = runtime / 'bin/ffmpeg.exe'
        ffmpeg.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg.write_text('optional')
        included = self.root / 'included'
        with mock.patch('sys.argv', base + ['--include-ffmpeg', '--output', str(included)]), mock.patch.object(package.subprocess, 'run', side_effect=run):
            package.main()
        self.assertEqual((included / 'bin/ffmpeg.exe').read_text(), 'optional')

if __name__ == '__main__':
    main()
