"""Launch options for the Windows native CPU and CUDA solvers."""
from pathlib import Path

SESSION_SOLVER_MODE_VERSION = 1
ANIMATION_SWITCH = "PPF_ANIMATION_BOUNDED_PROGRESS"

def validate_preserved_mode(launcher_path, mode):
    """Check the existing command before a preserve-output rebuild replaces it."""
    launcher_path = Path(launcher_path)
    try:
        command = launcher_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        command = ""
    lines = command.splitlines()
    seal = "@rem PPF_SESSION_SOLVER_MODE="
    sealed_modes = [line[len(seal):] for line in lines if line.startswith(seal)]
    if sealed_modes:
        previous = sealed_modes[0] if len(sealed_modes) == 1 else None
    else:
        switch = '@set "' + ANIMATION_SWITCH + '='
        assignments = [line[len(switch):] for line in lines if line.startswith(switch)]
        previous = {('1"',): "animation", ('"',): "strict", (): "strict"}.get(tuple(assignments))
        if "_r14_solver_launcher.py" in command:
            check_legacy_r14_animation_resume(launcher_path.with_name("stdout.log"))
    if previous != mode:
        raise ValueError("SOLVER_MODE_CHECKPOINT_MISMATCH: transfer again and start a new run to change Solver Mode")


def check_legacy_r14_animation_resume(stdout_path):
    """Keep a previous unsealed animation run from continuing in strict."""
    labels = {
        f"[PPFbone3 trial] backend={backend} mode=animation recipe=G-CPU+D2-CUDA"
        for backend in ("cpu", "cuda")
    }
    try:
        log = Path(stdout_path).open(encoding="utf-8-sig", errors="replace")
    except FileNotFoundError:
        return
    with log:
        if any(line.rstrip("\r\n") in labels for line in log):
            raise ValueError(
                "LEGACY_ANIMATION_RESUME_REQUIRES_RETRANSFER: transfer again and start "
                "a new run, or resume with the previous interpreter"
            )


def candidate_switches(backend, mode):
    """Native feature switches shared by the CLI and compatibility runs."""
    switches = ["BODY_FULL_NATIVE", "BODY_FULL_FACTOR_COMPACT", "SPARSE_CAPTURE_ARITH",
                "SPARSE_CAPTURE_COMPACT", "COMPACT_JOINT_H_EXPORT"]
    if backend == "cpu":
        switches += ["CPU_C2_PARALLEL_PROBE", "CPU_RIGID_QUERY_CULL", "CPU_EE_QUERY_PROBE",
                     "CPU_POINT_QUERY_PROBE", "CPU_MIXED_QUERY_CULL", "CCD_SKIP_COVERED_POINT_EDGE"]
    elif backend == "cuda":
        switches += ["JOINT_STRIDE_GRAPH", "JOINT_CLASSIC_PRODUCT_GRAPH", "ROOT_EXACT_FINAL",
                     "BODY_FULL_NATIVE_CONTACT_ONLY", "BODY_FULL_ANCHORED_FREE_FLIGHT",
                     "CUDA_CONTACT_MERGE_REFIT"]
    else:
        raise ValueError(f"unknown backend {backend!r}")
    if mode == "animation":
        switches.append("ANIMATION_BOUNDED_PROGRESS")
    elif mode != "strict":
        raise ValueError(f"unknown mode {mode!r}")
    options = {"PPF_" + switch: "1" for switch in switches}
    if backend == "cpu":
        options["PPF_B31_CERTIFICATE_MODE"] = "off"
        options["PPF_B32_CERTIFICATE_MODE"] = "reuse"
    return options

def configured(backend, mode="strict"):
    if mode not in ("strict", "animation"):
        raise ValueError(f"session solver_mode must be strict or animation, got {mode!r}")
    if backend not in ("cpu", "cuda"):
        if mode == "animation":
            raise ValueError("Current solver product does not support Animation Mode")
        return mode, {}, [ANIMATION_SWITCH]
    enabled = candidate_switches(backend, mode)
    managed = set(candidate_switches("cpu", "animation")) | set(candidate_switches("cuda", "animation"))
    return mode, enabled, sorted(managed - enabled.keys())


def solver_env(env, backend, mode="strict"):
    mode, enabled, cleared = configured(backend, mode)
    for name in cleared:
        env.pop(name, None)
    env.update(enabled)
    if enabled:
        print(f"[PPFbone3 trial] backend={backend} mode={mode} recipe=G-CPU+D2-CUDA", flush=True)
    return env


def configure_ordinary_batch(command_path, build_dir, mode="strict"):
    directory = Path(build_dir)
    try:
        backend = (directory / ".ppf-backend").read_text(encoding="utf-8-sig").strip().lower()
    except FileNotFoundError:
        backend = None
    mode, enabled, cleared = configured(backend, mode)
    header = ["@rem PPF_SESSION_SOLVER_MODE=" + mode, "@setlocal DisableDelayedExpansion"]
    header.extend('@set "' + name + '="' for name in cleared)
    header.extend('@set "' + name + '=' + value + '"' for name, value in enabled.items())
    if enabled:
        header.append(f"@echo [PPFbone3 trial] backend={backend} mode={mode} recipe=G-CPU+D2-CUDA")
    path = Path(command_path)
    path.write_bytes(("\r\n".join(header) + "\r\n").encode("ascii") + path.read_bytes())
