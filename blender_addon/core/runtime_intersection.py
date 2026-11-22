"""Pure display of native runtime intersection receipts (optional, v1).
Build violations use a different namespace. Never infer Blender mesh indices
from native topology IDs, and never report the asked step as committed time.
"""
import math

_KINDS = {
    "face_edge": ("Face {a} / edge {b}", (3, 2)),
    "edge_edge": ("Edge {a} / edge {b}", (2, 2)),
    "collision_mesh": ("Solver element {a} / collision-mesh element {b}", (None, None)),
    "point_point": ("Point {a} / point {b}", (1, 1)),
}
_NAMESPACES = {"solver_vertex", "constraint_collision_mesh_vertex"}


def _index(x):
    return x if type(x) is int and 0 <= x < 2**32 else None


def _time(x):
    return x if type(x) in (int, float) and 0 <= x < 1e12 and math.isfinite(x) else None


def intersection_rows(records, *, details=False):
    """Return display templates with values; no Blender dependency or side effects."""
    if not isinstance(records, (list, tuple)):
        return []
    valid = []
    for rec in records[:256]:
        if not isinstance(rec, dict):
            continue
        kind = rec.get("type")
        if not isinstance(kind, str) or kind not in _KINDS:
            continue
        a, b = _index(rec.get("elem0")), _index(rec.get("elem1"))
        if a is None or b is None:
            continue
        valid.append((rec, *_KINDS[kind], a, b))
    if not valid:
        return []
    out = [("Candidate step rejected. Pairs reported: {count}.", {"count": len(valid)})]
    runtime = valid[0][0].get("runtime")
    if isinstance(runtime, dict) and runtime.get("schema_version") == 1 and runtime.get("accepted") is False:
        f = _index(runtime.get("last_accepted_solver_frame"))
        step = _index(runtime.get("logical_step"))
        time = _time(runtime.get("step_start_seconds"))
        dt = _time(runtime.get("requested_step_seconds"))
        if f is not None:
            out.append(("Last written solver frame: {frame}", {"frame": f}))
        if step is not None and time is not None:
            out.append(("Rejected substep {step} at {time:.6f} s", {"step": step, "time": time}))
        elif time is not None:
            out.append(("Rejected step start: {time:.6f} s", {"time": time}))
        if details and dt is not None:
            out.append(("Requested step: {dt:.6g} s (not accepted)", {"dt": dt}))
        if details and runtime.get("mesh_toi_stream") == "data/advance.contact_toi.out":
            out.append(("Mesh CCD fractions: data/advance.contact_toi.out (not specific to a pair)", {}))
    for rec, label, widths, a, b in valid[:16 if details else 1]:
        out.append((label, {"a": a, "b": b}))
        labels = []
        for side in (0, 1):
            obj = rec.get(f"object_refs{side}")
            name = obj.get("object_name") if isinstance(obj, dict) else None
            if isinstance(name, str) and name.strip():
                labels.append(" ".join(name[:120].split())[:64])
            else:
                labels.append(None)
        if labels[0] and labels[1]:
            out.append(("{first} / {second}", {"first": labels[0], "second": labels[1]}))
        elif any(labels):
            idx = 0 if labels[0] else 1
            out.append(("Object on side {side}: {name}", {"side": idx + 1, "name": labels[idx]}))
        if details:
            for n, expected in enumerate(widths):
                ids = rec.get(f"vertex_ids{n}")
                ns = rec.get(f"vertex_namespace{n}")
                if not isinstance(ids, list) or not isinstance(ns, str) or ns not in _NAMESPACES or not 1 <= len(ids) <= 3:
                    continue
                if expected is not None and len(ids) != expected:
                    continue
                if any(_index(v) is None for v in ids):
                    continue
                out.append(("Side {side}: {namespace} IDs {ids}", {
                    "side": n + 1, "namespace": ns, "ids": ", ".join(map(str, ids)),
                }))
    if details and len(valid) > 16:
        out.append(("See intersection_records.json for additional records.", {}))
    return out
