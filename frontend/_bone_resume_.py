"""Host-only real capability gate. No CUDA import, no fabricated support."""
import json,subprocess

def capability(profile, child_env, twist_resume_mode=None):
    proc=subprocess.run([str(profile.exe_path),'--resume-capability-json'],
        cwd=str(profile.runtime_dir),env=child_env,capture_output=True,text=True,encoding='utf-8',timeout=10)
    try:value=json.loads(proc.stdout) if proc.returncode==0 else None
    except ValueError:value=None
    if not isinstance(value,dict) or value.get('kind')!='ppf.bone_resume_capability' or value.get('version')!=1 or value.get('feature')!='bone_checkpoint_resume_v1' or value.get('supported') is not True or value.get('target_frame_override') is not True:
        raise ValueError('BONE_RESUME_CAPABILITY_UNAVAILABLE: selected host does not advertise completed bone_checkpoint_resume_v1')
    if twist_resume_mode is not None and (twist_resume_mode != 'continuous_from_current' or not isinstance(value.get('supported_modes'),list) or twist_resume_mode not in value['supported_modes']):
        raise ValueError('TWIST_RESUME_CAPABILITY_UNAVAILABLE')
    if twist_resume_mode is not None and value.get('backend') != profile.backend:
        raise ValueError('TWIST_RESUME_BACKEND_MISMATCH')
    return value

def resume_arguments(forwarded):
    """Parse only protected resume controls; reject duplicate/ambiguous forms."""
    values={};i=0
    while i<len(forwarded):
        arg=forwarded[i];key=arg.split('=',1)[0]
        if key in ('--load','--target-frame','--twist-resume-mode'):
            if key in values:raise ValueError('RESUME_ARGUMENT_DUPLICATE: '+key)
            if '=' in arg:value=arg.split('=',1)[1]
            else:
                i+=1
                if i>=len(forwarded):raise ValueError('RESUME_ARGUMENT_MISSING: '+key)
                value=forwarded[i]
            if key == '--twist-resume-mode':
                if value != 'continuous_from_current':raise ValueError('TWIST_RESUME_MODE_INVALID')
                values[key]=value
            else:
                try:values[key]=int(value)
                except ValueError:raise ValueError('RESUME_ARGUMENT_INVALID: '+key)
        i+=1
    load=values.get('--load',0);target=values.get('--target-frame')
    if '--twist-resume-mode' in values and load<=0:
        raise ValueError('TWIST_RESUME_MODE_INVALID: positive checkpoint required')
    if target is not None and (load<=0 or target<=load or target>2147483647):
        raise ValueError('RESUME_TARGET_INVALID: explicit positive checkpoint and later i32 target required')
    return load,target

def twist_mode(forwarded):
    for i, arg in enumerate(forwarded):
        if arg.startswith('--twist-resume-mode='):
            return arg.split('=', 1)[1]
        if arg == '--twist-resume-mode':
            return forwarded[i+1]
    return None


def v2_history_candidate(output, frame):
    """Classify bounded V2 envelope solely for exe-rebind eligibility.

    This is not checkpoint validation; native restore checks all identity,
    history and actual pose before pruning or advancing.
    """
    import math, os, stat, struct
    from pathlib import Path
    if type(frame) is not int or frame <= 0:
        return False
    path=Path(output)/f'bone_history_{frame}.bin'
    try:
        info=path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400 or not 44<=info.st_size<=256*1024*1024:
            return False
        with path.open('rb') as stream: head=stream.read(44)
        magic,version,stored_frame,clock,length=struct.unpack('<8sIidQ',head[:32])
        return (magic==b'PPFBONE1' and version==1 and stored_frame==frame and math.isfinite(clock) and clock>=0
                and length==info.st_size-32 and head[32:40]==b'PPFCHV2\x00' and struct.unpack('<I',head[40:44])[0]==2)
    except (OSError,ValueError,struct.error):
        return False
