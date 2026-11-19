"""MCP Task Processing System for thread-safe Blender operations with integrated decorator support."""

import threading
import time

from .integration import get_integrated_handlers, initialize_integrated_system

_get_integrated_handlers = get_integrated_handlers
_initialize_integrated_system = initialize_integrated_system


# MCP Task Processing System (keep existing task system)
_mcp_task_queue = []
_mcp_results = {}
_mcp_result_times = {}
_mcp_lock = threading.Lock()
_mcp_task_id_counter = 0

# Network threads must never read bpy to identify the active editing context.
# Native main-thread lifecycle callbacks advance this generation under the same
# lock as admission and execution claim. Paths and UUIDs cannot detect a reload
# of the same file, an A -> B -> A round trip, or undo/redo of the same IDs.
_mcp_context_generation = 0
_mcp_context_transitions = {}


def _begin_context_change(kind):
    global _mcp_context_generation
    with _mcp_lock:
        _mcp_context_generation += 1
        _mcp_context_transitions[kind] = _mcp_context_transitions.get(kind, 0) + 1


def _finish_context_change(kind):
    with _mcp_lock:
        depth = _mcp_context_transitions.get(kind, 0)
        if depth <= 1:
            _mcp_context_transitions.pop(kind, None)
        else:
            _mcp_context_transitions[kind] = depth - 1


def begin_file_change(*_args):
    _begin_context_change("file")


def finish_file_change(*_args):
    # Successful and failed loads both invalidate old work. Never roll back
    # generation on failure, even when Blender kept the original file.
    _finish_context_change("file")


def begin_undo_change(*_args):
    _begin_context_change("undo")


def finish_undo_change(*_args):
    _finish_context_change("undo")


def begin_redo_change(*_args):
    _begin_context_change("redo")


def finish_redo_change(*_args):
    _finish_context_change("redo")


def _context_hooks():
    # Only called by addon lifecycle on the main thread, never by HTTP work.
    import bpy
    return ((bpy.app.handlers.load_pre, begin_file_change),
            (bpy.app.handlers.load_post, finish_file_change),
            (bpy.app.handlers.load_post_fail, finish_file_change),
            (bpy.app.handlers.undo_pre, begin_undo_change),
            (bpy.app.handlers.undo_post, finish_undo_change),
            (bpy.app.handlers.redo_pre, begin_redo_change),
            (bpy.app.handlers.redo_post, finish_redo_change))


def _remove_context_hooks():
    for callbacks, callback in _context_hooks():
        for existing in list(callbacks):
            if (getattr(existing, "__module__", None) == callback.__module__
                    and getattr(existing, "__name__", None) == callback.__name__):
                callbacks.remove(existing)


def register_context_handlers():
    """Install persistent hooks after all other PPF registration completes."""
    from bpy.app.handlers import persistent
    _begin_context_change("registration")
    try:
        _remove_context_hooks()
        for callbacks, callback in _context_hooks():
            persistent(callback)
            if callback in (begin_file_change, begin_undo_change, begin_redo_change):
                # Close before other PPF callbacks that might pump queued work.
                callbacks.insert(0, callback)
            else:
                callbacks.append(callback)
    except BaseException:
        # A partial installation must remain closed, but a later successful
        # register may recover. Do not strand this call's registration depth.
        with _mcp_lock:
            _mcp_context_transitions["disabled"] = 1
        raise
    else:
        # Re-enable after ordinary unregister/register churn without reopening
        # an outer native transition. An inner undo post must not admit work
        # before the outer post, and registration must not erase that depth.
        with _mcp_lock:
            _mcp_context_transitions.pop("disabled", None)
    finally:
        _finish_context_change("registration")


def unregister_context_handlers():
    """Close admission before addon teardown, even for ownerless requests."""
    _begin_context_change("disabled")
    _remove_context_hooks()


# Tasks whose reader has gone away, mapped to when the cancellation was
# recorded. Closing the response stream is the transport's cancellation
# signal, and a cancelled task must not leave a result behind for nobody.
#
# An entry is only needed while the task can still finish and store a result,
# so it is dropped as soon as that happens, and reaped on the same schedule as
# an abandoned result in case it never does.
_cancelled_tasks: dict = {}

# Reap results that were never collected (client disconnected or the
# get_mcp_result timeout already fired before the handler finished). The
# default get_mcp_result timeout is 5.0s, so 2x that is comfortably past
# the point any live reader would still be waiting, and a waiting reader
# always consumes its own entry within its timeout window first.
_RESULT_REAP_SECONDS = 10.0


def post_mcp_task(task_type, args, *, owner=None):
    """Post a task to be processed by the Blender timer in the main thread."""
    global _mcp_task_id_counter

    with _mcp_lock:
        if owner is not None and owner.is_set():
            raise RuntimeError("MCP listener stopped before task admission")
        if _mcp_context_transitions:
            raise RuntimeError(
                "Blender file or undo history is changing; task not admitted. "
                "Inspect the current context after it finishes and submit a new request."
            )
        _mcp_task_id_counter += 1
        task_id = f"mcp_task_{_mcp_task_id_counter}"

        task = {
            "id": task_id,
            "type": task_type,
            "args": args,
            "timestamp": time.time(),
            "context_generation": _mcp_context_generation,
        }

        if owner is not None:
            task["owner"] = owner
        _mcp_task_queue.append(task)

    return task_id


def try_get_mcp_result(task_id):
    """Take the result of *task_id* if it is ready.

    Returns ``(True, result)`` once, then ``(False, None)`` forever after, so
    a caller that polls can tell "not finished yet" from "finished and already
    collected". This is what lets a transport wait on a long tool call without
    committing to a fixed deadline up front.
    """
    with _mcp_lock:
        if task_id in _mcp_results:
            result = _mcp_results.pop(task_id)
            _mcp_result_times.pop(task_id, None)
            return True, result
    return False, None


def cancel_mcp_task(task_id, *, owner=None):
    """Drop *task_id*, whether it is queued, running, or already finished.

    A queued task is removed before it runs. A task taken into the current
    batch is checked again immediately before it is claimed for execution.
    After that claim, its handler may run to completion; only the result is
    discarded. This never interrupts, rolls back or repeats the handler.
    """
    with _mcp_lock:
        removed_from_queue = False
        for index, task in enumerate(_mcp_task_queue):
            if task["id"] == task_id:
                del _mcp_task_queue[index]
                removed_from_queue = True
                break
        _mcp_results.pop(task_id, None)
        _mcp_result_times.pop(task_id, None)
        # A task taken off the queue never runs, so there is no later result
        # to suppress and nothing to remember. Recording it anyway would grow
        # the map by one entry for every cancelled call, forever.
        if owner is not None and owner.is_set():
            # The immutable owner also guards snapshotted/claimed tasks.
            # No tombstone is needed after unregister clears global state.
            _cancelled_tasks.pop(task_id, None)
        elif not removed_from_queue:
            _cancelled_tasks[task_id] = time.time()


def stop_task_owner(owner):
    """Linearize listener shutdown against task admission and execution claim.

    Claimed handlers may finish once; their owner prevents result storage.
    An owner is never reset or reused by a later listener.
    """
    with _mcp_lock:
        owner.set()
        _mcp_task_queue[:] = [
            task for task in _mcp_task_queue if task.get("owner") is not owner
        ]


def get_mcp_result(task_id, timeout=5.0, *, owner=None):
    """Wait up to *timeout* seconds for the result of a posted task.

    Suits a caller that knows the work is short. A caller that cannot bound
    the work should poll ``try_get_mcp_result`` instead, so a slow task is
    reported as still running rather than as failed.
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        if owner is not None and owner.is_set():
            cancel_mcp_task(task_id, owner=owner)
            return {"status": "error", "message": "MCP listener stopped"}
        found, result = try_get_mcp_result(task_id)
        if found:
            return result

        time.sleep(0.01)  # Small sleep to prevent busy waiting

    # Timeout occurred
    return {
        "status": "error",
        "message": f"Task {task_id} timed out after {timeout} seconds",
    }


def process_mcp_tasks():
    """Process MCP tasks in the main thread. Called by Blender timer."""
    tasks_to_process = []

    # Get all pending tasks
    with _mcp_lock:
        tasks_to_process = _mcp_task_queue[:]
        _mcp_task_queue.clear()

    # Process each task. Passing this lock-protected check is the execution
    # claim boundary, not a promise that cancellation can stop a running
    # handler. Never hold the bookkeeping lock while executing add-on code.
    for task in tasks_to_process:
        task_id = task["id"]
        with _mcp_lock:
            owner = task.get("owner")
            if (owner is not None and owner.is_set()) or task_id in _cancelled_tasks:
                _cancelled_tasks.pop(task_id, None)
                continue
            context_changed = (task.get("context_generation") != _mcp_context_generation
                            or bool(_mcp_context_transitions))
        task_type = task["type"]
        args = task["args"]

        if context_changed:
            result = {
                "status": "error",
                "message": "Blender file or undo history changed; task not executed. "
                           "Inspect the current file and submit a new request.",
            }
        else:
            try:
                result = _execute_blender_task(task_type, args)
            except Exception as e:
                result = {"status": "error", "message": str(e)}

        # A claimed handler may itself load a file. Preserve its actual result:
        # the context fence does not interrupt, roll back or replay claimed work.

        # Store result, unless the reader has gone away.
        with _mcp_lock:
            owner = task.get("owner")
            if (owner is not None and owner.is_set()) or task_id in _cancelled_tasks:
                _cancelled_tasks.pop(task_id, None)
                continue
            _mcp_results[task_id] = result
            _mcp_result_times[task_id] = time.time()

    # Reap any results abandoned by a reader that already timed out or
    # disconnected, so _mcp_results does not grow without bound over a long
    # session. A live get_mcp_result deletes its own entry well within its
    # timeout window, so this only evicts entries no reader will ever read.
    now = time.time()
    with _mcp_lock:
        stale_ids = [
            tid
            for tid, stored_at in _mcp_result_times.items()
            if now - stored_at > _RESULT_REAP_SECONDS
        ]
        for tid in stale_ids:
            _mcp_results.pop(tid, None)
            del _mcp_result_times[tid]
        # A cancellation whose task never came back is reaped on the same
        # schedule, so the map cannot grow without bound over a long session.
        for tid in [
            tid
            for tid, marked_at in _cancelled_tasks.items()
            if now - marked_at > _RESULT_REAP_SECONDS
        ]:
            del _cancelled_tasks[tid]


def _execute_blender_task(task_type, args):
    """Execute a Blender task in the main thread where bpy access is safe."""
    handlers = _get_integrated_handlers()
    handler = handlers.get(task_type)
    if handler:
        return handler(args)
    return {"status": "error", "message": f"Unknown task type: {task_type}"}


def clear_task_state():
    """Drop any pending tasks and results. Called on addon unregister so
    a subsequent reload doesn't try to run handlers that point into freed
    module namespaces. Also invalidate tasks already copied into a batch."""
    global _mcp_context_generation
    with _mcp_lock:
        _mcp_context_generation += 1
        _mcp_task_queue.clear()
        _mcp_results.clear()
        _mcp_result_times.clear()
        _cancelled_tasks.clear()


# Initialize the integrated system on module load
_initialize_integrated_system()
