"""Private aggregate publication prototype for mixed Static/Pin captures."""
from pathlib import Path
import json
import os
import shutil
import uuid
import numpy as np

from . import capture_provenance as proof_tools, pc2

_MISSING = object()


def _snapshot_field(owner, key):
    return owner, key, owner[key] if key in owner else _MISSING


def _restore_field(record):
    owner,key,value=record
    if value is _MISSING:
        if key in owner:
            del owner[key]
    else:
        owner[key]=value


def _directory_sync(parent):
    if os.name!="posix":
        return  # Windows durability not qualified for this prototype.
    fd=os.open(parent,os.O_RDONLY | getattr(os,"O_DIRECTORY",0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _stat_witness(path):
    try:
        st=os.lstat(path)
        return st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns
    except FileNotFoundError:
        return None


def _pin_snapshot(pin):
    operations=tuple(q.op_type for q in pin.operations)
    embedded=[i for i,q in enumerate(operations) if q=="EMBEDDED_MOVE"]
    if len(embedded)>1:
        raise ValueError("Pin has duplicate embedded operations; mixed capture cannot change them atomically")
    return (pin, bool(pin.has_captured_anim), int(pin.operations_index), operations,
            embedded[0] if embedded else None)


def _pin_apply(saved):
    pin,_,_,old_ops,where=saved
    live=tuple(q.op_type for q in pin.operations)
    if live!=old_ops:
        raise RuntimeError("Pin operations changed after group commit preparation")
    if where is None:
        q=pin.operations.add()
        q.op_type="EMBEDDED_MOVE"
        if len(old_ops):
            pin.operations.move(len(old_ops),0)
    elif where!=0:
        pin.operations.move(where,0)
    pin.has_captured_anim=True


def _pin_restore(saved):
    pin,old_flag,old_active,old_ops,where=saved
    coll=pin.operations
    # Blender CollectionProperty's .move() moves *values*; the pointer can
    # continue identifying the array slot. Invert only our sentinel delta,
    # without deleting or re-creating any pre-existing author's operation.
    actual=tuple(q.op_type for q in coll)
    if actual!=old_ops:
        sentinels=[i for i,t in enumerate(actual) if t=="EMBEDDED_MOVE"]
        if where is not None:
            if len(actual)!=len(old_ops) or len(sentinels)!=1:
                raise RuntimeError("Pin operations changed during failed capture")
            start=sentinels[0]
            if start!=where:
                coll.move(start,where)
        elif len(actual)==len(old_ops)+1 and len(sentinels)<=1:
            # If assigning the sentinel op_type failed, new entry is still
            # an ordinary default op appended at the end.
            coll.remove(sentinels[0] if sentinels else len(old_ops))
        else:
            raise RuntimeError("Pin operations changed during failed capture")
    if tuple(q.op_type for q in coll)!=old_ops:
        raise RuntimeError("Pin operation order could not be restored")
    pin.operations_index=old_active
    pin.has_captured_anim=old_flag


def _validate_task(scene, kind, obj, entry, pin):
    from .uuid_registry import get_object_uuid, get_object_by_uuid
    from .encoder.pin import _get_pin_indices
    from ..ui.dynamics import pin_capture_ops
    uid=entry["obj_uuid"]
    if not uid or get_object_uuid(obj)!=uid or get_object_by_uuid(uid)!=obj or scene.objects.get(obj.name)!=obj:
        raise ValueError("Capture task does not own a live object in the source Scene")
    arr=entry["frames"]
    if (not isinstance(arr,np.ndarray) or arr.dtype!=np.dtype("<f4") or not arr.flags.c_contiguous
        or arr.ndim!=3 or arr.shape[2]!=3 or len(arr)<=0 or arr.shape[1]<=0
        or entry["frames_done"]!=len(arr)
        or entry["frame_end"]-entry["frame_start"]+1!=len(arr)
        or not np.isfinite(arr).all()):
        raise ValueError(f"{obj.name}: mixed capture requires a complete finite packed float32 buffer")
    if kind=="PIN":
        resolved=pin_capture_ops._resolve_capture_pin(scene,entry)
        if resolved.as_pointer()!=pin.as_pointer():
            raise ValueError(f"{obj.name}: mixed capture Pin identity changed")
        if any(op.op_type=="TORQUE" for op in pin.operations):
            raise ValueError(f"{obj.name}: PIN acquired a TORQUE operation")
        if pin_capture_ops._pin_has_vertex_co_fcurves(pin):
            raise ValueError(f"{obj.name}: PIN acquired manual keyframes")
        if len(entry["pin_indices"])!=arr.shape[1] or not np.array_equal(entry["pin_indices"],_get_pin_indices(obj,entry["vg_name"])):
            raise ValueError(f"{obj.name}: PIN membership changed")
        return arr,pc2.pin_anim_pc2_key(obj,entry["vg_name"]),pc2._pin_anim_cache
    if kind!="STATIC" or pin is not None or arr.shape[1]!=len(obj.data.vertices) or entry["n_verts"]!=arr.shape[1]:
        raise ValueError(f"{obj.name}: Static capture scope or topology changed")
    return arr,pc2.static_deform_pc2_key(obj),pc2._static_deform_cache


def _regular_path(parent,path):
    if path.parent!=parent or path.name in {".",".."}:
        raise ValueError("Mixed capture destination escaped its dedicated cache directory")
    proof_tools._regular_or_absent(os.fspath(path))


def commit_batch_capture(scene,tasks,*,recheck=None):
    """Publish all complete Static/Pin captures or restore original state.

    tasks are (kind, obj, complete entry, pin_item or None). Scheduler owns
    origin/timeline and any prior save protections; recheck() verifies frozen
    sampling identities, ranges and other live author settings, without
    evaluating frames. Complete arrays are retained by the committed cache;
    caller must not mutate/reuse any array after success. Readers/writers from
    this add-on honor the durable group barrier, which is fail-closed on crash.
    No global atomicity, concurrent old writer, or Windows crash claim.
    """
    import bpy
    from .capture_origin import require_capture_ranges
    if not bpy.data.filepath:
        raise ValueError("Save the project before using mixed capture")
    tasks=tuple(tasks)
    if not tasks:
        raise ValueError("Mixed capture has no completed targets")
    parent=Path(pc2.get_pc2_dir()).absolute()
    marker=parent/proof_tools.GROUP_PENDING_NAME
    proof_tools._assert_group_idle(parent)
    parent.mkdir(parents=True,exist_ok=True)
    generation=uuid.uuid4().hex
    stage_cleanup=[]
    backups={}
    witnesses={}
    touched=[]
    staged_rows=[]
    pin_plans=[]
    seen_keys=set()
    seen_pins=set()
    scenes={scene.as_pointer():scene}
    marker_owned=False
    undos=[]
    rollback_failed=False
    old_project_dir=pc2._capture_project_dir
    metadata=[]
    try:
        if recheck:recheck()
        require_capture_ranges(scene,[entry for _,_,entry,_ in tasks])
        for kind,obj,entry,pin in tasks:
            arr,key,memory=_validate_task(scene,kind,obj,entry,pin)
            live=Path(pc2.get_pc2_path(key)).absolute()
            side=Path(os.fspath(live)+proof_tools.PROOF_SUFFIX)
            pair_pending=Path(os.fspath(live)+proof_tools.PENDING_SUFFIX)
            for f in (live,side,pair_pending):
                _regular_path(parent,f)
            if os.path.lexists(pair_pending):
                raise ValueError(f"{live.name}: prior single-cache transaction needs recovery")
            if live.name.casefold() in seen_keys:
                raise ValueError(f"Duplicate mixed-capture destination: {live.name}")
            seen_keys.add(live.name.casefold())
            if pin is not None:
                pptr=pin.as_pointer()
                if pptr in seen_pins:
                    raise ValueError("Pin appears twice in mixed capture")
                seen_pins.add(pptr)
                pin_plans.append(_pin_snapshot(pin))
            if len(obj.users_scene)==1:
                scenes[obj.users_scene[0].as_pointer()]=obj.users_scene[0]
            record=proof_tools.make_proof(obj,kind,entry)
            record["generation"]=generation
            record.update(shape=list(arr.shape),payload_sha256=proof_tools._payload_digest(arr))
            record["proof_sha256"]=proof_tools._proof_digest(record)
            s_pc2=Path(proof_tools._temp(parent,'.mix.pc2')); stage_cleanup.append(s_pc2)
            s_proof=Path(proof_tools._temp(parent,'.mix.json')); stage_cleanup.append(s_proof)
            pc2.write_pc2(s_pc2,arr)
            with s_pc2.open('rb+') as f:os.fsync(f.fileno())
            with s_proof.open('wb') as f:
                f.write(proof_tools._canonical(record));f.flush();os.fsync(f.fileno())
            for target in (live,side):
                stamp=_stat_witness(target)
                if stamp is not None:
                    backup=Path(proof_tools._temp(parent,'.mix.backup'))
                    # Retain independently even if rollback later succeeds for
                    # only a subset of paths.
                    backups[target]=backup
                    shutil.copy2(target,backup)
                    with backup.open('rb+') as f:os.fsync(f.fileno())
                else:
                    backups[target]=None
                witnesses[target]=stamp
            staged_rows.append((obj,key,memory,arr,record,live,side,s_pc2,s_proof,
                                memory.get(key,_MISSING)))
        if recheck:recheck()
        require_capture_ranges(scene,[entry for _,_,entry,_ in tasks])
        metadata=[_snapshot_field(owner,k) for owner in scenes.values()
                  for k in (proof_tools.BINDINGS,pc2._CAPTURE_OWNERS)]
        if any(_stat_witness(path)!=stamp for path,stamp in witnesses.items()):
            raise RuntimeError("Capture input was changed by another writer; mixed commit cancelled")
        manifest={"schema":"ppf.mixed-capture-pending.v1","generation":generation,
                  "backup_records":[[p.name,b.name if b is not None else None] for p,b in backups.items()]}
        # Exclusive and persisted before exposing any target pathname.
        with marker.open('xb') as f:
            marker_owned=True
            f.write(proof_tools._canonical(manifest));f.flush();os.fsync(f.fileno())
        _directory_sync(parent)
        for obj,key,memory,arr,record,live,side,s_pc2,s_proof,old in staged_rows:
            for src,destination in ((s_pc2,live),(s_proof,side)):
                # Mark before replace to handle errors even after a rename.
                touched.append(destination)
                os.replace(src,destination)
        for obj,key,memory,arr,record,live,side,s_pc2,s_proof,old in staged_rows:
            proof_tools.bind(scene,key,record)
            pc2._record_capture_owner(obj,key)
            # Exclusive takeover of frozen completed sample buffer: no third
            # full retained copy. The old reference is kept until commit.
            memory[key]=arr
        for plan in pin_plans:
            undos.append(plan)
            _pin_apply(plan)
        pc2._capture_project_dir=pc2.get_pc2_dir()
        _directory_sync(parent)
        marker.unlink()
        marker_owned=False
        return generation
    except BaseException as original:
        failures=[]
        if marker_owned:
            for plan in reversed(undos):
                try:_pin_restore(plan)
                except BaseException as exc:failures.append(str(exc))
            for _,key,memory,_,_,_,_,_,_,old in staged_rows:
                if old is _MISSING:
                    memory.pop(key,None)
                else:
                    memory[key]=old
            pc2._capture_project_dir=old_project_dir
            for record in metadata:
                try:_restore_field(record)
                except BaseException as exc:failures.append(str(exc))
            for target in reversed(touched):
                try:
                    bak=backups[target]
                    if bak is None:
                        if os.path.lexists(target):
                            target.unlink()
                    else:
                        rebound=Path(proof_tools._temp(parent,'.mix.restore'))
                        stage_cleanup.append(rebound)
                        shutil.copy2(bak,rebound)
                        with rebound.open('rb+') as f:os.fsync(f.fileno())
                        os.replace(rebound,target)
                except BaseException as exc:
                    failures.append(f"{target.name}: {exc}")
            if not failures:
                try:
                    _directory_sync(parent)
                    marker.unlink()
                    marker_owned=False
                except BaseException as exc:
                    failures.append(str(exc))
        rollback_failed=bool(failures)
        if rollback_failed:
            raise RuntimeError("Mixed capture commit/rollback uncertain; cache directory blocked and backups retained: " +
                               "; ".join(failures)) from original
        raise
    finally:
        for path in stage_cleanup:
            proof_tools._remove_temp(path)
        if not rollback_failed:
            for path in backups.values():
                proof_tools._remove_temp(path)
