"""Focused tests for rule 8's scope, duplicate and entry-seed handling."""

import importlib.util
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("check-shared-wiring.py")
SPEC = importlib.util.spec_from_file_location("shared_wiring_scope_test", SCRIPT)
wiring = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wiring)
ROOT = SCRIPT.resolve().parents[3]
KERNELGEN = wiring.load_kernelgen(ROOT / "crates/ppf-cts-solver/src/kernels")


class ScopeCensusTests(unittest.TestCase):
    def graph(self, *sources):
        base = Path(tempfile.gettempdir()).resolve()
        temp = tempfile.TemporaryDirectory(prefix="ppf_wiring_scope_", dir=base)
        directory = Path(temp.name).resolve()
        self.assertEqual(directory.parent, base)
        self.addCleanup(temp.cleanup)
        paths = []
        for index, source in enumerate(sources):
            path = directory / f"case_{index}.kernel.cpp"
            path.write_text(source, encoding="utf-8")
            paths.append(path.name)
        return wiring.neutral_call_graph(directory, paths, KERNELGEN)[0]

    def test_namespaced_twins_reach_both_bare_callees(self):
        graph = self.graph('''
            namespace alpha {
              [[seam::device_fn]] bool finite(float x) { return x == x; }
              [[seam::device_fn]] void alpha_entry() { finite(1.f); }
            }
            namespace beta {
              [[seam::device_fn]] bool finite(float x) { return x == x; }
              [[seam::device_fn]] void beta_entry() { finite(2.f); }
            }
        ''')
        self.assertIn("alpha::finite", graph)
        self.assertIn("beta::finite", graph)
        self.assertEqual(graph["alpha::alpha_entry"], {"alpha::finite", "beta::finite"})
        self.assertEqual(wiring.reached({"finite"}, graph), {"alpha::finite", "beta::finite"})
        self.assertEqual(wiring.reached({"alpha::finite"}, graph), {"alpha::finite"})

    def test_same_qualified_definition_is_rejected_even_across_files(self):
        source = 'namespace alpha { [[seam::device_fn]] void run(float x) {} }'
        with self.assertRaises(SystemExit):
            self.graph(source, source.replace('float x', 'float renamed'))

    def test_nested_namespaces_and_struct_owners_are_distinct(self):
        graph = self.graph('''
            namespace outer::inner {
              struct First { struct Shared { [[seam::device_fn]] void step() {} }; };
              struct Second { struct Shared { [[seam::device_fn]] void step() {} }; };
            }
            namespace outer { namespace extra {
              [[seam::device_fn]] void run() {}
            }}
        ''')
        self.assertIn("outer::inner::First::Shared::step", graph)
        self.assertIn("outer::inner::Second::Shared::step", graph)
        self.assertIn("outer::extra::run", graph)

    def test_comments_and_string_literals_cannot_invent_namespaces(self):
        graph = self.graph('''
            // namespace bogus { [[seam::device_fn]] void fake() {} }
            const char *message = "namespace pretend {";
            namespace real { [[seam::device_fn]] void run() {} }
        ''')
        self.assertEqual(set(graph), {"real::run"})

    def test_twin_callees_are_unioned_but_invalid_duplicates_fail(self):
        graph = self.graph('''
            namespace scope {
              [[seam::device_fn]] void first() {}
              [[seam::device_fn]] void second() {}
              [[seam::device_fn]] [[seam::cooperative]] void run(int x) { first(); }
              [[seam::device_fn]] [[seam::serial]] void run(int x) { second(); }
            }
        ''')
        self.assertEqual(graph["scope::run"], {"scope::first", "scope::second"})
        with self.assertRaises(SystemExit):
            self.graph('''
              [[seam::device_fn]] [[seam::serial]] void run(int x) {}
              [[seam::device_fn]] [[seam::serial]] void run(int x) {}
            ''')
        with self.assertRaises(SystemExit):
            self.graph('''
              [[seam::device_fn]] void run(int x) {}
              [[seam::device_fn]] [[seam::cooperative]] void run(int x) {}
            ''')

    def test_actual_constructor_shape_distinguishes_arities(self):
        graph = self.graph('''
            namespace joint { struct Dual {
              [[seam::host_device_fn]] Dual(): x(0.f) {}
              [[seam::host_device_fn]] explicit Dual(float value): x(value) {}
              float x;
            }; [[seam::device_fn]] void entry() { Dual(1.f); } }
        ''')
        self.assertIn("joint::Dual::Dual[arity=0]", graph)
        self.assertIn("joint::Dual::Dual[arity=1]", graph)
        self.assertEqual(wiring.matching_bodies("joint::Dual::Dual[arity=0]", graph),
                         {"joint::Dual::Dual[arity=0]"})
        self.assertEqual(graph["joint::entry"], {
            "joint::Dual::Dual[arity=0]", "joint::Dual::Dual[arity=1]"})
        with self.assertRaises(SystemExit):
            self.graph('''
              [[seam::device_fn]] void f(float x) {}
              [[seam::device_fn]] void f(int x) {}
            ''')

    def test_all_generated_entry_seeds_must_resolve(self):
        graph = self.graph('''
            namespace a { [[seam::device_fn]] void go() {} }
            namespace b { [[seam::device_fn]] void go() {} }
            [[seam::device_fn]] void global() {}
        ''')
        seeds = wiring.generated_entry_seeds(["go", "a::go", "::global"], graph)
        self.assertEqual(wiring.reached(seeds, graph), set(graph))
        with self.assertRaises(SystemExit):
            wiring.generated_entry_seeds(["go", "missing::entry"], graph)
        with self.assertRaises(SystemExit):
            wiring.generated_entry_seeds(["go", "go"], graph)
        wrapper_seeds = {"a::go_entry": [("test", 1)], "go_entry_launch": [("test", 2)]}
        resolved = wiring.resolve_entry_seeds(wrapper_seeds, graph)
        self.assertEqual(set(resolved), {"a::go", "b::go"})
        self.assertEqual(resolved["a::go"], [("test", 1), ("test", 2)])
        with self.assertRaises(SystemExit):
            wiring.resolve_entry_seeds({"unknown_entry": [("test", 3)]}, graph)

    def test_ambiguous_exception_cannot_excuse_both_namespaces(self):
        graph = self.graph('''
            namespace a { [[seam::device_fn]] void finite() {} }
            namespace b { [[seam::device_fn]] void finite() {} }
        ''')
        grants, errors = wiring.resolve_reachability_exceptions(
            {("metal", "finite"): ("today", "reason")}, graph)
        self.assertEqual(grants, {})
        self.assertEqual(len(errors), 1)
        self.assertIn("ambiguous", errors[0])
        grants, errors = wiring.resolve_reachability_exceptions(
            {("metal", "a::finite"): ("today", "reason")}, graph)
        self.assertEqual(set(grants), {("metal", "a::finite")})
        self.assertEqual(errors, [])

    def test_explicit_out_of_line_qualified_definition(self):
        graph = self.graph('''
            namespace outer {
              struct Holder { void step(); };
              [[seam::device_fn]] void Holder::step() {}
            }
        ''')
        self.assertEqual(set(graph), {"outer::Holder::step"})

    def test_unsupported_or_unbalanced_namespace_fails_closed(self):
        with self.assertRaises(SystemExit):
            self.graph('namespace { [[seam::device_fn]] void run() {} }')
        with self.assertRaises(SystemExit):
            self.graph('namespace broken { [[seam::device_fn]] void run() {}')


class DriverAndPathTests(unittest.TestCase):
    def fixture_root(self):
        temp = tempfile.TemporaryDirectory(prefix="ppf_wiring_driver_")
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / "crates/ppf-cts-solver/src/driver").mkdir(parents=True)
        (root / "crates/ppf-cts-solver/src/kernels/sub").mkdir(parents=True)
        return root

    def test_table_order_reads_multiline_inline_and_ignores_comment(self):
        root = self.fixture_root()
        path = root / "crates/ppf-cts-solver/src/driver/kernels.rs"
        path.write_text('''pub static TABLE: [KernelDecl; id::COUNT] = [
            decl_generated(\n id::FIRST, FIRST_NAME),
            // decl_generated(id::PHANTOM, PHANTOM_NAME),
            decl_generated(id::SECOND, SECOND_NAME),
        ];
        decl_generated(id::OUTSIDE, OUTSIDE_NAME);
        ''', encoding="utf-8")
        self.assertEqual(wiring.driver_table_order(root), ["first", "second"])

    def test_compact_kernel_ids_resolve_real_launch_order_and_mismatch_fails(self):
        root = self.fixture_root()
        driver = root / "crates/ppf-cts-solver/src/driver"
        (driver / "kernels.rs").write_text('''
            pub const FIRST: KernelId = KernelId(0);
            pub const SECOND:KernelId=KernelId(1);
            decl_generated(id::FIRST, FIRST_NAME);
            decl_generated(id::SECOND, SECOND_NAME);
        ''', encoding="utf-8")
        launch = driver / "launch.rs"
        launch.write_text('''static LAUNCH: [Launch; id::COUNT] = [
            launch_first,
            launch_second,
        ];
        generated_thunk!(launch_first, FirstArgs, first_entry);
        generated_thunk!(launch_second, SecondArgs, second_entry);
        ''', encoding="utf-8")
        with mock.patch.object(wiring, "_rendered_thunks", return_value=""):
            msgs = []
            self.assertEqual(wiring.check_eleven(root, msgs), 2)
            self.assertEqual(msgs, [])
            launch.write_text(launch.read_text().replace("launch_second,\n", "launch_first,\n"), encoding="utf-8")
            msgs = []
            wiring.check_eleven(root, msgs)
            self.assertTrue(any('dispatches `first_entry`' in msg for msg in msgs))

    def test_kernel_path_matches_build_list_and_missing_path_fails(self):
        root = self.fixture_root()
        kernel = root / "crates/ppf-cts-solver/src/kernels/sub/case.kernel.cpp"
        kernel.write_text('[[seam::entry]]\nvoid case_entry() {}\n', encoding="utf-8")
        build = root / "crates/ppf-cts-solver/build.rs"
        build.write_text('const KERNELS: [&str; 1] = [\n "sub/case",\n];', encoding="utf-8")
        msgs = []
        self.assertEqual(wiring.check_fourteen(root, msgs), 0)
        self.assertEqual(msgs, [])
        build.write_text('const KERNELS: [&str; 1] = [\n "sub/other",\n];', encoding="utf-8")
        self.assertEqual(wiring.check_fourteen(root, msgs), 1)
        self.assertIn("sub/case.kernel.cpp", msgs[-1])

    def test_actual_table_and_compact_tail_indices(self):
        rows = wiring.driver_table_order(ROOT)
        self.assertEqual(len(rows), 366)
        self.assertEqual(rows[-8:], ["ftt_embed", "ftt_evaluate", "ftt_gather",
                         "ftt_combine", "ftt_mask_values", "rqc_freeze", "rqc_local", "rqc_modes"])
        text = (ROOT / "crates/ppf-cts-solver/src/driver/kernels.rs").read_text(encoding="utf-8")
        ids = {name: int(value) for name, value in wiring.KERNEL_ID_RE.findall(text)}
        self.assertEqual([ids[name.upper()] for name in rows[-8:]], list(range(358, 366)))

    def test_production_append_order_and_fail_closed_cases(self):
        root = self.fixture_root()
        kernels = root / "crates/ppf-cts-solver/src/kernels"
        seam = kernels / "seam"
        seam.mkdir()
        helper = root / "crates/ppf-cts-compute/seam/kernel_order.py"
        helper.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "crates/ppf-cts-compute/seam/kernel_order.py", helper)
        order = seam / "kernel_order.txt"
        order.write_text("sub/z.kernel.cpp\nsub/a.kernel.cpp\n", encoding="utf-8")
        (kernels / "sub/z.kernel.cpp").write_text("z", encoding="utf-8")
        (kernels / "sub/a.kernel.cpp").write_text("a", encoding="utf-8")
        body = kernels / "sub/body.kernel.cpp"
        body.write_text("", encoding="utf-8")
        generator = types.SimpleNamespace(read_source=lambda path: (None, None, None, [
            types.SimpleNamespace(name=name, emit_entry=True)
            for name in Path(path).read_text(encoding="utf-8").split()]))
        driver = root / "crates/ppf-cts-solver/src/driver/kernels.rs"
        driver.write_text('pub static TABLE: [KernelDecl; id::COUNT] = [\n'
                          'decl_generated(id::Z, Z_NAME),\n'
                          'decl_generated(id::A, A_NAME),\n];\n', encoding="utf-8")
        with mock.patch.object(wiring, "_load_kernelgen", return_value=generator):
            self.assertEqual(wiring.canonical_entry_order(root), ["z", "a"])
            msgs = []
            self.assertEqual(wiring.check_thirteen(root, msgs), (0, 0))
            self.assertEqual(msgs, [])
            driver.write_text(driver.read_text().replace("id::Z", "id::TEMP").replace(
                "id::A", "id::Z").replace("id::TEMP", "id::A"), encoding="utf-8")
            msgs = []
            self.assertEqual(wiring.check_thirteen(root, msgs), (0, 2))
            self.assertTrue(msgs)
            body.write_text("omitted", encoding="utf-8")
            with self.assertRaises(SystemExit):
                wiring.canonical_entry_order(root)
            body.write_text("", encoding="utf-8")
            order.write_text("sub/z.kernel.cpp\nsub/z.kernel.cpp\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                wiring.canonical_entry_order(root)
            order.write_text("sub/z.kernel.cpp\nsub/a.kernel.cpp\n", encoding="utf-8")
            (kernels / "sub/a.kernel.cpp").write_text("z", encoding="utf-8")
            with self.assertRaises(SystemExit):
                wiring.canonical_entry_order(root)
            order.unlink()
            with self.assertRaises(SystemExit):
                wiring.canonical_entry_order(root)


if __name__ == "__main__":
    unittest.main()
