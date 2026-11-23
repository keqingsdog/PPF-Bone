"""Isolated prototype; preserve the projected outer contour and all source constraints."""
from collections import Counter,defaultdict
import numpy as np
from mathutils import Vector,geometry
from mathutils.bvhtree import BVHTree
from .bone_weight_precision import _clip,cross,area,MIN_PROTECTIVE_THICKNESS
from .bone_weight_proximity import check_python


def ordered_boundary(faces):
    sides=Counter((int(a),int(b)) for t in faces for a,b in zip(t,np.roll(t,-1)))
    if any(v!=1 for v in sides.values()):raise ValueError('invalid cap winding')
    loop={a:b for a,b in sides if (b,a) not in sides}
    if not loop or len(set(loop.values()))!=len(loop):raise ValueError('invalid cap boundary')
    first=min(loop);path=[first];cursor=loop[first]
    while cursor!=first and len(path)<len(loop):
        path.append(cursor);cursor=loop[cursor]
    if len(path)!=len(loop) or cursor!=first:raise ValueError('cap has multiple boundaries')
    return path


def local_cap_mesh(p,source,q,good,protect,scale,old_low,old_high,protected):
    cap_xy=q[good]
    src=np.asarray(source)
    base_xy=p[:,:2]
    src_xy=p[src,:2]
    src_z=p[src,2]
    eps=1.e-10*scale*scale
    tol=max(1.e-9,scale*1.e-7)
    mins=cap_xy.min(axis=1)
    maxs=cap_xy.max(axis=1)
    bot=old_low.copy()
    top=old_high.copy()
    inc_down=np.zeros(len(q));inc_up=np.zeros(len(q))
    conditions=[]
    # Keep existing generated elevations, even between formerly neighboring
    # samples, unless retessellation makes a fully selected source point/face
    # stick out. Resolve the narrow tangential defect without changing the
    # outline in any of x, y or z.
    def cover(k,xy,z):
        if not len(z):return
        ids=good[k]
        basis=(cap_xy[k,1:]-cap_xy[k,:1]).T
        uv=np.linalg.solve(basis,(xy-cap_xy[k,0]).T).T
        barycentric=np.c_[1-uv.sum(axis=1),uv]
        allowable=~protected[ids]
        conditions.append((ids,barycentric,z))
        low_req=np.maximum(barycentric@bot[ids]-z,0.)
        high_req=np.maximum(z-barycentric@top[ids],0.)
        adjusted=barycentric[:,allowable].sum(axis=1)
        if np.any((adjusted<1e-8)&((low_req>tol)|(high_req>tol))):
            raise ValueError('would have to change the protected outer edge')
        usable=adjusted>=1e-8
        if np.any(usable):
            for req,adjust in ((low_req,inc_down),(high_req,inc_up)):
                required=np.max((req[usable]/adjusted[usable]))
                if required>0:adjust[ids[allowable]]=np.maximum(adjust[ids[allowable]],required+tol)
    def candidate_triangles(pts):
        return np.nonzero(np.all(maxs>=pts.min(axis=0)-tol,axis=1)&np.all(mins<=pts.max(axis=0)+tol,axis=1))[0]
    # Source range of each cap face from exact nonempty projected intersections.
    for t,z in zip(src_xy,src_z):
        total=0.
        for k in candidate_triangles(t):
            xy,zs=_clip(t,z,cap_xy[k],eps)
            if len(zs):
                cover(k,xy,zs)
                if len(zs)>=3:total+=abs(area(xy))
        target=abs(area(t))
        if total < target-max(eps*64,target*1.e-7):
            raise ValueError('selected source triangle lost from footprint')
    # All target points constrain the envelope even when they are not mesh
    # vertices in the candidate and lie on no fully selected source face.
    for i,point in enumerate(base_xy):
        found=False
        for k in candidate_triangles(point[None,:]):
            t=cap_xy[k]
            if min(cross(b-a,point-a) for a,b in zip(t,np.roll(t,-1,axis=0)))>=-eps*64:
                cover(k,point[None,:],p[i:i+1,2]);found=True
        if not found:raise ValueError('selected weighted point lost')
    if not np.isfinite(np.r_[bot,top]).all():raise ValueError('unsupported unbounded patch')
    bot-=inc_down
    top+=inc_up
    if np.min(top-bot)<protect-tol:raise ValueError('insufficient cap thickness')
    if any(np.any(bary@bot[ids]>z+tol) or np.any(bary@top[ids]<z-tol) for ids,bary,z in conditions):
        raise ValueError('selected source coverage did not survive retriangulation')
    return bot,top


def repair(points, source, vertices, faces, offset, cycles=6):
    """Benchmark receives positive absolute world-space offset; never changes it."""
    original_v=np.asarray(vertices,dtype=float)
    original_f=np.asarray(faces,dtype=np.int32)
    if offset<=0:return original_v,original_f,[{'close_pairs':0,'unchanged_zero':True}]
    close,_=check_python(original_v,original_f,offset)
    history=[{'close_pairs':len(close),'triangles':len(faces)}]
    if not close:return original_v,original_f,history
    xyz=np.asarray(points,dtype=float)
    origin=xyz.mean(axis=0)
    delta=xyz-origin
    _,_,axes=np.linalg.svd(delta,full_matrices=False)
    if np.linalg.det(axes)<0:axes[2]*=-1
    p=delta@axes.T
    scale=float(max(np.ptp(p,axis=0)))
    tol=max(1.e-9,scale*1.e-7)
    n=len(original_v)//2
    original_local=(original_v-origin)@axes.T
    pool=original_local[:n,:2]
    cap=original_f[original_f.min(axis=1)>=n]-n
    outer=ordered_boundary(cap)
    outline=set(outer)
    protected_xy=pool[outer]
    available={i for i in range(n)}
    names=np.arange(n,dtype=int)
    world=original_v
    tris=original_f
    for iteration in range(cycles):
        generation_n=len(world)//2
        # Identify closest bridges between independent triangles; prefer one
        # interior endpoint on a short connecting edge, never a boundary vertex.
        connected={tuple(sorted((int(a),int(b)))) for t in cap for a,b in zip(t,np.roll(t,-1))}
        scores=defaultdict(float)
        for i,j in close:
            candidates=[]
            x,y=tris[i]%generation_n,tris[j]%generation_n
            for a in x:
                for b in y:
                    if a==b or tuple(sorted((a,b))) not in connected:continue
                    if names[a] in outline and names[b] in outline:continue
                    candidates.append((np.linalg.norm(pool[names[a]]-pool[names[b]]),names[a],names[b]))
            if candidates:
                d,a,b=min(candidates)
                for choice in (a,b):
                    if choice not in outline:scores[choice]+=1/max(d,1.e-9)
        if not scores:break
        remove=[]
        # One batch of disjoint critical strips avoids one recomputation per vertex.
        for k in sorted(scores,key=lambda k:(-scores[k],int(k))):
            if all(np.linalg.norm(pool[k]-pool[j])>2.5*offset for j in remove):
                remove.append(k)
            if len(remove)>=max(2,min(10,1+len(close)//8)):break
        attempt=available-set(remove)
        # Greedy check (no new boundary points, no duplicate output coordinates).
        source_ids=sorted(attempt)
        g={i:k for k,i in enumerate(source_ids)}
        boundary_indices=[g[i] for i in outer]
        d=geometry.delaunay_2d_cdt([Vector(v) for v in pool[source_ids]],[],[boundary_indices],1,tol)
        good=np.asarray(d[2],dtype=int)
        if not len(good) or good.ndim!=2 or good.shape[1]!=3 or any(len(ids)!=1 for ids in d[3]):break
        q=np.asarray(d[0],dtype=float)
        names=np.array([source_ids[ids[0]] for ids in d[3]],dtype=int)
        signed=np.array([cross(b-a,c-a) for a,b,c in q[good]])
        if np.any(np.abs(signed)<=1.e-10*scale*scale):break
        good=np.array([row if cc>0 else row[::-1] for row,cc in zip(good,signed)])
        boundary=ordered_boundary(good)
        if set(names[boundary])!=outline:break
        order_idx=[next(i for i,j in enumerate(names) if j==idx) for idx in outer]
        if np.max(np.linalg.norm(q[order_idx]-protected_xy,axis=1))>tol:break
        try:
            bot,top=local_cap_mesh(p,source,q,good,MIN_PROTECTIVE_THICKNESS,scale, original_local[names,2],original_local[names+n,2],np.isin(names,outer))
        except ValueError as exc:
            history.append({'rejected':str(exc),'removed':list(map(int,remove))})
            break
        cap=good
        n2=len(q)
        world=np.vstack((np.c_[q,bot],np.c_[q,top]))@axes+origin
        fs=[list(t[::-1]) for t in good]+[list(t+n2) for t in good]
        fs.extend(face for a,b in zip(boundary,np.roll(boundary,-1))
              for face in [(a,b,b+n2),(a,b+n2,a+n2)])
        tris=np.asarray(fs,dtype=np.int32)
        inc=Counter((a,b) for f in tris for a,b in zip(f,np.roll(f,-1)))
        assert all(h==inc[b,a]==1 for (a,b),h in list(inc.items()))
        rounded=np.asarray(world,dtype=np.float32).astype(float)-origin
        tv=rounded[tris]
        cross3=np.cross(tv[:,1]-tv[:,0],tv[:,2]-tv[:,0])
        vol=np.einsum('ij,ij->i',tv[:,0],np.cross(tv[:,1],tv[:,2])).sum()/6
        bvh=BVHTree.FromPolygons(rounded.tolist(),tris.tolist(),all_triangles=True)
        face_sets=[frozenset(f) for f in tris]
        if vol<=1.e-15 or not np.all(np.linalg.norm(cross3,axis=1)>2.e-14) or any(i<j and not face_sets[i]&face_sets[j] for i,j in bvh.overlap(bvh)):
            history.append({'rejected':'degenerate or self-intersecting result','removed':list(map(int,remove))})
            break
        possible_thin,_=check_python(world,tris,MIN_PROTECTIVE_THICKNESS*.5)
        if any((tris[i].max()<n2 and tris[j].min()>=n2) or (tris[j].max()<n2 and tris[i].min()>=n2) for i,j in possible_thin):
            history.append({'rejected':'minimum separation of the two caps did not hold','removed':list(map(int,remove))})
            break
        close,_=check_python(world,tris,offset)
        history.append({'triangles':len(tris),'removed':list(map(int,remove)),'close_pairs':len(close),'thickness_floor_m':float(np.min(top-bot))})
        if not close:return world,tris,history
        available=attempt
    return None,None,history
