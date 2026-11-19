"""Interpreter-exit cleanup: Python locks/threads/transports only, never bpy.

Backend.disconnect closes SSH transports or drops native references; it does
not stop a user server/solver. No engine effect is dispatched here.
"""
from threading import current_thread


def detach_runner(runner, join_timeout=0.25, _current_thread=current_thread):
    """Idempotently retire local I/O and detach its backend without RNA access."""
    runner._interrupt.set()
    runner.stop()  # Pure Python: retire tickets/frames, stop/wake, clear I/O queue.
    # Match _adopt_backend's lock: an attempt is either retired or captured here.
    with runner._connect_lock:
        runner._connect_epoch += 1
        backend = runner._backend
        runner._backend = None
        runner._project_name = None
    try:
        if backend is not None:
            backend.disconnect()
    finally:
        worker = runner._worker
        if worker is not _current_thread() and worker.is_alive():
            worker.join(timeout=join_timeout)
