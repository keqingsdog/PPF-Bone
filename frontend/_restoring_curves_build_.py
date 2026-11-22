"""Native-authority Build adapter. Query is pure host; capability is separate."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import uuid
from . import _restoring_curves_ as curve
from . import _r14_product_adapter as r14

def encode(doc):
    return json.dumps(doc,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')

_SHELL_DIAGNOSTIC = re.compile(
    r"Degenerate shell face ([0-9]{1,19}) on vertices "
    r"([0-9]{1,19}), ([0-9]{1,19}), ([0-9]{1,19}):"
)
_SHELL_RATIO = re.compile(r"conditioned at ([0-9.eE+-]+).*?below the ([0-9.eE+-]+)")
_TRI_ROW = struct.Struct("=QQQ")  # Same native-endian uint64 layout as export_fixed.
_XYZ_ROW = struct.Struct("=ddd")
_DMAP_ROW = struct.Struct("=I")


def _triangle_matches(scene, face, vertices):
    try:
        with (scene / 'bin' / 'tri.bin').open('rb') as stream:
            if face >= os.fstat(stream.fileno()).st_size // _TRI_ROW.size:
                return False
            stream.seek(face * _TRI_ROW.size)
            return _TRI_ROW.unpack(stream.read(_TRI_ROW.size)) == vertices
    except (OSError, OverflowError, struct.error):
        return False


def _initial_triangle(scene, vertices):
    """Read the same displaced solver-space positions as the collision overlay."""
    def read_row(stream, row, index):
        stream.seek(row.size * index)
        return row.unpack(stream.read(row.size))

    base = scene / 'bin'
    try:
        with ((base / 'vert.bin').open('rb') as points,
              (base / 'vert_dmap.bin').open('rb') as groups,
              (base / 'displacement.bin').open('rb') as offsets):
            tri = []
            for vertex in vertices:
                group, = read_row(groups, _DMAP_ROW, vertex)
                xyz = read_row(points, _XYZ_ROW, vertex)
                offset = read_row(offsets, _XYZ_ROW, group)
                tri.append([p + d for p, d in zip(xyz, offset)])
            if all(math.isfinite(v) for xyz in tri for v in xyz):
                return tri
    except (OSError, OverflowError, struct.error):
        pass
    return None


def _shell_face_owner(face, vertices, statistics_objects):
    """Best-effort label; ownership also has to agree with all three vertices."""
    try:
        matched = []
        for obj in statistics_objects:
            # Captured Static can be in the non-static solver triangle pool.
            if obj['static_object'] is not False:
                continue
            start, count = obj['face_range']
            if start <= face < start + count:
                matched.append(obj)
        if len(matched) != 1:
            return None
        obj = matched[0]
        local = tuple(obj['vertex_indices'].index(v) for v in vertices)
        name = ' '.join(obj['object_name'].split())
        if name:
            return name[:90], face - obj['face_range'][0], local
    except Exception:
        # Unavailable or malformed optional metadata must not hide the native error.
        return None
    return None


def _mass_query_failure(scene, staging, returncode, stderr, statistics_objects=()):
    lines = [' '.join(line.split()) for line in
             stderr[-65536:].decode('utf-8', errors='replace').splitlines()]
    specific = next(((line, m) for line in lines if (m := _SHELL_DIAGNOSTIC.search(line))), None)
    violations = []
    if specific:
        line, match = specific
        face, *indices = map(int, match.groups())
        vertices = tuple(indices)
        verified = _triangle_matches(scene, face, vertices)
        owner = _shell_face_owner(face, vertices, statistics_objects) if verified else None
        if owner:
            name, tri, local = owner
            where = f'"{name}", exported triangle {tri} (0-based), local vertices {", ".join(map(str, local))}'
            where += f' (solver face {face})'
        else:
            where = f'solver face {face}, vertex IDs {", ".join(map(str, vertices))}'
        ratio = _SHELL_RATIO.search(line)
        quality = f' ({ratio[1]} < {ratio[2]})' if ratio else ''
        reason = (f'Degenerate triangle: {where}. Rest vertices are nearly collinear{quality}. '
                  'Check and retriangulate the source face.')
        triangle = _initial_triangle(scene, vertices) if verified else None
        if triangle is not None:
            entry = dict(triangle=triangle, solver_face=face)
            if owner:
                entry.update(object_name=owner[0], exported_triangle=owner[1])
            violations.append(dict(type='degenerate_shell_face', count=1, entries=[entry]))
    else:
        panic_reason = next((lines[i + 1] for i, line in enumerate(lines[:-1])
                             if ' panicked at ' in line and line.endswith(':') and lines[i + 1]), '')
        fallback = next((line for line in reversed(lines) if line and not line.startswith(
            ('note:', 'stack backtrace:', 'RUST_BACKTRACE', 'at '))), '')
        reason = f'Native mass query failed (exit {returncode}): '
        reason += (panic_reason or fallback)[:200] or 'no diagnostic on stderr'
    error = curve.CurveError('RESTORING_MASS_INVALID', f'{reason} | Full log: {staging / "stderr.log"}')
    if violations:
        error.violations = violations
    return error

def parse(data):
    if len(data)>curve.MAX_JSON:curve.fail('authority JSON bytes','RESTORING_BUDGET_EXCEEDED')
    def unique(items):
        out={}
        for k,v in items:
            if k in out:curve.fail('duplicate authority JSON key')
            out[k]=v
        return out
    try:return json.loads(data,object_pairs_hook=unique,parse_constant=lambda _:curve.fail('nonfinite authority'))
    except (ValueError,UnicodeDecodeError):curve.fail('authority JSON')

def snapshot(path):
    """Read-only query protects old output/checkpoint files too, beyond base seal."""
    out={}
    for root,dirs,files in os.walk(path):
        for name in dirs+files:
            file=Path(root)/name
            if file.is_symlink():curve.fail('query snapshot symlink','RESTORING_BASIS_MISMATCH')
        for name in files:
            file=Path(root)/name
            out[file.relative_to(path).as_posix()]=hashlib.sha256(file.read_bytes()).hexdigest()
    return out

def validate_authority(doc,request,request_bytes):
    curve.keys(doc,'kind version scene_uuid basis_sha256 request_sha256 solver_identity authority_sha256 targets','authority')
    if doc['kind']!='ppf.restoring_curves.mass_authority' or type(doc['version']) is not int or doc['version']!=1:curve.fail('authority version','RESTORING_VERSION_UNSUPPORTED')
    for k in ('scene_uuid','basis_sha256'):
        if doc[k]!=request[k]:curve.fail(k,'RESTORING_BASIS_MISMATCH')
    if doc['request_sha256']!=hashlib.sha256(request_bytes).hexdigest():curve.fail('request hash','RESTORING_BASIS_MISMATCH')
    curve.digest(doc['authority_sha256'],'authority_sha256')
    identity=doc['solver_identity']
    curve.keys(identity,'source_stamp selected_backend abi_version layout_sha256 kernel_table_sha256','solver_identity')
    if not isinstance(identity['source_stamp'],str) or not identity['source_stamp'] or identity['selected_backend'] not in ('cpu','cuda') or type(identity['abi_version']) is not int or identity['abi_version']<=0:curve.fail('solver_identity','RESTORING_CAPABILITY_UNAVAILABLE')
    for k in ('layout_sha256','kernel_table_sha256'):curve.digest(identity[k],k)
    rows=doc['targets']
    if not isinstance(rows,list) or len(rows)!=len(request['targets']):curve.fail('authority row count','RESTORING_TARGET_INVALID')
    aliases=set()
    for requested,row in zip(request['targets'],rows):
        curve.keys(row,'target needs_mass native_kind target_index metric_eff metric_unit quality mobility_eigenvalues zero_direction_count anisotropy_ratio','authority target')
        curve.target(row['target'],True)
        expected=requested['target']
        if type(row['needs_mass']) is not bool or row['needs_mass']!=requested['needs_mass'] or any(row['target'].get(k)!=v for k,v in expected.items()):curve.fail('authority target order/identity','RESTORING_TARGET_INVALID')
        kind=curve.uint(row['native_kind'],'native_kind');index=curve.uint(row['target_index'],'target_index')
        tar=row['target'];anchor=tar['kind']=='JOINT' and tar['channel']=='anchor';angular=tar['kind']=='JOINT' and not anchor;stitch=tar['kind']=='STITCH_ROW'
        if kind>4 or angular!=(kind in (0,1)) or anchor!=(kind in (2,3)) or stitch!=(kind==4) or stitch and index!=tar['row_index']:curve.fail('native family/index','RESTORING_TARGET_INVALID')
        alias=(kind,index,tar.get('channel','stitch'))
        if alias in aliases:curve.fail('native alias','RESTORING_CONFLICT')
        aliases.add(alias)
        if not row['needs_mass']:
            if any(row[k] is not None for k in ('metric_eff','metric_unit','quality','mobility_eigenvalues','zero_direction_count','anisotropy_ratio')):curve.fail('geometry-only authority nulls')
            continue
        curve.number(row['metric_eff'],'metric_eff',positive=True)
        if row['metric_unit']!=('kg_m2' if angular else 'kg'):curve.fail('metric_unit','RESTORING_MASS_INVALID')
        quality=row['quality']
        accepted=('LOCAL_RIGID_ESTIMATE',) if kind==0 else ('FEM_LOCAL_MASS_ESTIMATE',) if kind in (1,3) else ('ANCHOR_MEAN_COMPLIANCE_ESTIMATE',) if kind==2 else ('STITCH_NODAL_LOCAL_ESTIMATE',)
        if quality not in accepted:curve.fail('quality','RESTORING_MASS_INVALID')
        if anchor:
            eig=row['mobility_eigenvalues']
            if not isinstance(eig,list) or len(eig)!=3:curve.fail('eigenvalues','RESTORING_MASS_INVALID')
            for v in eig:curve.number(v,'eigenvalue',nonnegative=True)
            positive=[v for v in eig if v>0]
            if not positive or type(row['zero_direction_count']) is not int or row['zero_direction_count']!=sum(v==0 for v in eig):curve.fail('zero directions','RESTORING_MASS_INVALID')
            ratio=max(positive)/min(positive)
            curve.number(row['anisotropy_ratio'],'anisotropy_ratio',positive=True)
            if abs(ratio-row['anisotropy_ratio'])>ratio*1e-6 or abs(3/sum(eig)-row['metric_eff'])>row['metric_eff']*1e-6:curve.fail('anchor diagnostics','RESTORING_MASS_INVALID')
        elif any(row[k] is not None for k in ('mobility_eigenvalues','zero_direction_count','anisotropy_ratio')):curve.fail('nonanchor diagnostics','RESTORING_SCHEMA_INVALID')
    return doc

def query(exe,scene,request,*,statistics_objects=()):
    curve.keys(request,'kind version scene_uuid basis_sha256 targets','mass request')
    if request['kind']!='ppf.restoring_curves.mass_request' or type(request['version']) is not int or request['version']!=1:curve.fail('request version','RESTORING_VERSION_UNSUPPORTED')
    curve.canonical_uuid(request['scene_uuid'],'scene_uuid');curve.digest(request['basis_sha256'],'basis')
    if not isinstance(request['targets'],list) or len(request['targets'])>curve.MAX_BINDINGS:curve.fail('request targets','RESTORING_BUDGET_EXCEEDED')
    seen=set()
    for r in request['targets']:
        curve.keys(r,'target needs_mass','request target')
        if type(r['needs_mass']) is not bool:curve.fail('needs_mass')
        t=r['target']
        if t.get('kind')=='STITCH_ROW':
            curve.keys(t,'kind row_index rows_sha256','request stitch target')
            curve.uint(t['row_index'],'row_index');curve.digest(t['rows_sha256'],'rows_sha256')
            key='STITCH_ROW:'+str(t['row_index'])
        else:key=curve.target(t,True)
        if key in seen:curve.fail('duplicate query target','RESTORING_CONFLICT')
        seen.add(key)
    exe=Path(exe).resolve();scene=Path(scene).resolve()
    staging=scene.parent/'.restoring-curves-build'/str(uuid.uuid4())
    staging.mkdir(parents=True)
    req_bytes=encode(request);req=staging/'request.json';req.write_bytes(req_bytes)
    before=snapshot(scene);exe_before=hashlib.sha256(exe.read_bytes()).hexdigest()
    environment=os.environ.copy()
    tree=Path(__file__).resolve().parent.parent
    environment['PATH']=os.pathsep.join([str(exe.parent),str(tree/'bin'),environment.get('PATH','')])
    command=[str(exe),'--path',str(scene),'--restoring-curves-mass-query',str(req)]
    product_context=r14.solver_context(exe)
    child_cwd=tree
    if product_context is not None:
        selected_exe,child_cwd,environment=product_context
        command[0]=selected_exe
    result=subprocess.run(command,cwd=child_cwd,env=environment,capture_output=True,timeout=120)
    (staging/'stdout.json').write_bytes(result.stdout);(staging/'stderr.log').write_bytes(result.stderr)
    exe_after=hashlib.sha256(exe.read_bytes()).hexdigest()
    audit=dict(command=command,returncode=result.returncode,selected_exe_sha256_before=exe_before,selected_exe_sha256_after=exe_after,request_sha256=hashlib.sha256(req_bytes).hexdigest(),response_sha256=hashlib.sha256(result.stdout).hexdigest(),session_unchanged=snapshot(scene)==before,identity_evidence='COMPILED_SOLVER_ONLY_NOT_LOADED_BACKEND')
    (staging/'audit.json').write_bytes(encode(audit))
    if exe_before!=exe_after or not audit['session_unchanged']:curve.fail('query changed executable/session','RESTORING_BASIS_MISMATCH')
    if result.returncode!=0:
        raise _mass_query_failure(scene, staging, result.returncode, result.stderr,
                                  statistics_objects)
    return validate_authority(parse(result.stdout),request,req_bytes),str(staging)

def seal_marker(base_digest,sidecar_bytes):
    digest=hashlib.sha256(sidecar_bytes).hexdigest()
    return dict(required_feature='restoring_curves_v1',sidecar_sha256=digest,sidecar_byte_length=len(sidecar_bytes),combined_inputs_sha256=hashlib.sha256((base_digest+digest).encode('ascii')).hexdigest())

def capability(exe):
    """Opened-backend gate is distinct from the pure-host compiled identity."""
    exe=Path(exe).resolve()
    tree=Path(__file__).resolve().parent.parent
    environment=os.environ.copy();environment['PATH']=os.pathsep.join([str(exe.parent),str(tree/'bin'),environment.get('PATH','')])
    product_context=r14.solver_context(exe)
    child_cwd=tree
    if product_context is not None:
        selected_exe,child_cwd,environment=product_context
        exe=Path(selected_exe)
    result=subprocess.run([str(exe),'--restoring-curves-capability-json'],cwd=child_cwd,env=environment,capture_output=True,timeout=30)
    if result.returncode!=0:curve.fail('actual opened backend capability probe failed','RESTORING_CAPABILITY_UNAVAILABLE')
    doc=parse(result.stdout)
    curve.keys(doc,'kind version capability contract_version channel_mask backend source_stamp abi_version layout_sha256 kernel_table_sha256','capability')
    if doc['kind']!='ppf.restoring_curves.capability' or doc['capability']!='restoring_curves_v1' or any(type(doc[k]) is not int or doc[k]!=v for k,v in (('version',1),('contract_version',1),('channel_mask',15))) or doc['backend'] not in ('cpu','cuda') or type(doc['abi_version']) is not int or doc['abi_version']<=0 or not isinstance(doc['source_stamp'],str) or not doc['source_stamp']:curve.fail('incompatible backend capability','RESTORING_CAPABILITY_UNAVAILABLE')
    for k in ('layout_sha256','kernel_table_sha256'):curve.digest(doc[k],k)
    return doc

def build_and_seal(fixed,path,context):
    """Run native authority after ordinary complete base seal, then resolve SI."""
    from . import _run_directory
    from . import _cbor_bridge_ as cbor
    author=getattr(fixed,'_restoring_curves_author',None)
    if not curve.active(author):return
    if context.get('complete') is not True:curve.fail('mass query requires complete base Build','RESTORING_BASIS_MISMATCH')
    path=Path(path)
    ranges=getattr(fixed,'_restoring_curve_pair_ranges',[])
    has_pairs=any(b['enabled'] and b['target']['kind']=='STITCH_PAIR' for b in author['bindings'])
    total_rows=len(fixed._stitch_ind)
    import tomllib
    with (path/'info.toml').open('rb') as handle:
        exported_count=tomllib.load(handle).get('count',{}).get('stitch')
    if type(exported_count) is not int or exported_count!=total_rows:curve.fail('exported stitch count','RESTORING_TARGET_INVALID')
    pair_rows=curve.expand_pairs(author,ranges,len(ranges),total_rows)
    if has_pairs:
        rows_digest=curve.stitch_rows_hash((path/'bin/stitch_ind.bin').read_bytes(),(path/'bin/stitch_w.bin').read_bytes(),total_rows)
    else:rows_digest=None
    mapping={}; requests=[]
    for binding in author['bindings']:
        if not binding['enabled']:continue
        tar=binding['target'];key=curve.target(tar)
        targets=[tar] if tar['kind']=='JOINT' else [dict(kind='STITCH_ROW',row_index=i,rows_sha256=rows_digest) for i in pair_rows[tar['pair_id']]]
        mapping[key]=list(range(len(requests),len(requests)+len(targets)))
        if len(requests)+len(targets)>curve.MAX_BINDINGS:curve.fail('query expansion budget','RESTORING_BUDGET_EXCEEDED')
        requests.extend(dict(target=t,needs_mass=binding['mode']=='AUTO_FREQUENCY') for t in targets)
    request=dict(kind='ppf.restoring_curves.mass_request',version=1,scene_uuid=fixed._scene_uuid,basis_sha256=context['scene_inputs_sha256'],targets=requests)
    exe=Path(_run_directory(contract=getattr(fixed,'_build_contract',None)))/('ppf-contact-solver.exe' if os.name=='nt' else 'ppf-contact-solver')
    authority,staging=query(exe,path,request,
                            statistics_objects=getattr(fixed,'_statistics_objects',()))
    rows=authority['targets']
    def resolver(tar):
        return [(rows[i]['target'],rows[i]) for i in mapping[curve.target(tar)]]
    sidecar=curve.resolve(author,fixed._scene_uuid,context['scene_inputs_sha256'],resolver)
    raw=encode(sidecar)
    if len(raw)>curve.MAX_JSON:curve.fail('sidecar bytes','RESTORING_BUDGET_EXCEEDED')
    sidecar_path=path/'bin/restoring_curves_v1.json'
    temporary=sidecar_path.with_suffix('.json.pending')
    temporary.write_bytes(raw);os.replace(temporary,sidecar_path)
    context['restoring_curves_v1']=seal_marker(context['scene_inputs_sha256'],raw)
    (path/'bin/pdrd_scene_context.cbor').write_bytes(cbor.dumps_envelope_with_version(2,'ppf.pdrd_scene_context',context))
    fixed._restoring_curves_build_audit=dict(staging=staging,request_sha256=authority['request_sha256'],authority_sha256=authority['authority_sha256'],solver_identity=authority['solver_identity'],sidecar_sha256=context['restoring_curves_v1']['sidecar_sha256'],combined_inputs_sha256=context['restoring_curves_v1']['combined_inputs_sha256'],stitch_row_count=total_rows,pair_rows=pair_rows)
    (Path(staging)/'build-seal-audit.json').write_bytes(encode(fixed._restoring_curves_build_audit))
