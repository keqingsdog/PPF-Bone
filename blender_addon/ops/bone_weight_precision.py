"""Bounded concave height-envelope fitting for connected, thin skin patches."""
import numpy as np
from collections import Counter
from mathutils import Vector, geometry
from mathutils.bvhtree import BVHTree


MAX_SAMPLES = 160
MIN_PROTECTIVE_THICKNESS = 0.0005  # metres


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def area(points):
    return sum(cross(a, b) for a, b in zip(points, np.roll(points, -1, axis=0))) * .5


def _clip(src_xy,src_z,q,eps):
    points,values=src_xy,src_z
    for a,b in zip(q,np.roll(q,-1,axis=0)):
        if not len(points):return points,values
        out,z=[],[]
        for i, cur in enumerate(points):
            prev=points[i-1]
            x,y=cross(b-a,prev-a),cross(b-a,cur-a)
            u,v=x>=-eps,y>=-eps
            if u!=v:
                # Intersect the same relaxed half plane used for inside classification.
                t=(x+eps)/(x-y)
                out.append(prev+t*(cur-prev))
                z.append(values[i-1]+t*(values[i]-values[i-1]))
            if v:
                out.append(cur)
                z.append(values[i])
        points,values=np.asarray(out),np.asarray(z)
    return points,values


def fit_height_envelope(points, source, budget, strength):
    """Return a concave, closed piecewise height envelope on one PCA plane.

    Sources are the fully selected mesh triangles; isolated positive-weight
    vertices still constrain the cap. This is not arbitrary 3D reconstruction.
    The caller falls back to the original fitter for any rejection.
    """
    xyz=np.asarray(points,dtype=float)
    if len(xyz)<4 or len(xyz)>MAX_SAMPLES or not source or budget<12:
        raise ValueError('unsupported selected sheet size')
    if not np.isfinite(xyz).all():
        raise ValueError('nonfinite skin samples')
    # Separate components must not be merged into a new collision bridge.
    relatives={}
    for a,b,c in source:
        for i,j,k in ((a,b,c), (b,c,a), (c,a,b)):
            relatives.setdefault(i,set()).update((j,k))
    remaining=set(relatives)
    pending=[remaining.pop()]
    while pending:
        neighbors=relatives[pending.pop()] & remaining
        pending.extend(neighbors)
        remaining.difference_update(neighbors)
    if remaining:
        raise ValueError('disconnected selected skin surfaces')
    origin=xyz.mean(axis=0)
    delta=xyz-origin
    _,s,axes=np.linalg.svd(delta,full_matrices=False)
    if s[-1]>.22*s[0] or s[1]<.10*s[0]:raise ValueError('not a thin selected footprint')
    if np.linalg.det(axes)<0:axes[2]*=-1
    p=delta@axes.T
    scale=float(max(np.ptp(p,axis=0)))
    eps=1.e-10*scale*scale
    tol=max(1.e-9,scale*1.e-7)
    if scale<1.e-6:raise ValueError('too small')
    cdt=geometry.delaunay_2d_cdt([Vector(v) for v in p[:,:2]],[],[],0,tol)
    q=np.asarray(cdt[0],dtype=float)
    T=np.asarray(cdt[2],dtype=int)
    if len(T)==0 or T.shape[1]!=3:raise ValueError('invalid triangulation')
    # Every projected source point maps to exactly one output CDT vertex.
    inp=[[] for _ in p]
    for j,ids in enumerate(cdt[3]):
        for i in ids:inp[i].append(j)
    if any(len(ids)!=1 for ids in inp):raise ValueError('could not map triangulation')
    qp=np.asarray([ids[0] for ids in inp],dtype=int)
    q0,q1,q2=q[T[:,0]],q[T[:,1]],q[T[:,2]]
    signed=np.array([cross(b-a,c-a) for a,b,c in zip(q0,q1,q2)])
    radii=np.linalg.norm(q0-q1,axis=1)*np.linalg.norm(q0-q2,axis=1)*np.linalg.norm(q2-q1,axis=1)/np.maximum(2*np.abs(signed),eps)
    radii[np.abs(signed)<=eps]=np.inf
    T=np.array([t if d>0 else t[::-1] for t,d in zip(T,signed)])
    src=np.asarray(source,dtype=int); projected=q[qp[src]]
    src_edge=np.unique(np.sort(np.concatenate((src[:,[0,1]],src[:,[1,2]],src[:,[2,0]])),axis=1),axis=0)
    src_e=q[qp[src_edge]]
    reference=np.abs(signed[np.isfinite(radii)]).sum()*.5
    footprints=[]
    for cutoff in np.unique(radii[np.isfinite(radii)]):
        selected=radii<=cutoff
        good=T[selected]
        if len(np.unique(good))!=len(q):continue
        counts=Counter((a,b) for t in good for a,b in zip(t,np.roll(t,-1)))
        if any(n!=1 for n in counts.values()):continue
        boundary=[(a,b) for a,b in counts if (b,a) not in counts]
        end={a:b for a,b in boundary}
        if not end or len(end)!=len(boundary) or len(set(end.values()))!=len(end):continue
        if 2*(len(good)+len(boundary))>budget: continue
        i=min(end); seen={i}; j=end[i]
        while j not in seen:
            seen.add(j);j=end[j]
        if j!=i or len(seen)!=len(end):continue
        ap=float(np.abs(signed[selected]).sum()*.5)
        if ap>=.995*reference:continue
        # Reject any proper crossing of an authored source edge with contour boundary.
        # Remaining intersections (including grazing/overlap) are covered by exact per-face clip/area below.
        ay=q[[a for a,b in boundary]]; by=q[[b for a,b in boundary]]
        def ce(f,g):return f[...,0]*g[...,1]-f[...,1]*g[...,0]
        u=ce(by-ay,src_e[:,0,None,:]-ay[None,:,:])
        v=ce(by-ay,src_e[:,1,None,:]-ay[None,:,:])
        ee=(src_e[:,1]-src_e[:,0])[:,None,:]
        u2=ce(ee,ay[None,:,:]-src_e[:,0,None,:])
        v2=ce(ee,by[None,:,:]-src_e[:,0,None,:])
        if np.any((u*v<-eps*eps)&(u2*v2<-eps*eps)):continue
        footprints.append((ap, good, boundary))
    if not footprints:
        raise ValueError('no covered concavity within triangle budget')
    target_area=reference+(min(a for a,_,_ in footprints)-reference)*strength
    for ap,good,boundary in sorted(footprints,key=lambda x:abs(x[0]-target_area)):
        bot=np.full(len(q),np.inf);top=np.full(len(q),-np.inf)
        np.minimum.at(bot,qp,p[:,2]);np.maximum.at(top,qp,p[:,2])
        candidates=q[good]; mins=candidates.min(axis=1);maxs=candidates.max(axis=1)
        face_z=p[src,2]
        for t,z in zip(projected,face_z):
            target=abs(area(t)); total=0
            appropriate=np.nonzero(np.all(maxs>=t.min(axis=0)-tol,axis=1)&np.all(mins<=t.max(axis=0)+tol,axis=1))[0]
            for ind in appropriate:
                ix=good[ind]
                xy,zs=_clip(t,z,candidates[ind],eps)
                if len(zs):
                    bot[ix]=np.minimum(bot[ix],zs.min())
                    top[ix]=np.maximum(top[ix],zs.max())
                    if len(zs)>=3: total+=abs(area(xy))
            if total < target-max(eps*64,target*1.e-7): break
        else:
            # Both caps share the same planar graph; thickness is positive everywhere,
            # so extrusion walls can only intersect along prescribed shared edges.
            # At lower strength retain more depth while preserving coverage.
            # On a height graph, top-bottom >= floor in projection. To keep a
            # normal-space minimum even on slopes, inflate by a conservative
            # Lipschitz bound derived from both piecewise-linear graph slopes.
            bot=strength*bot+(1-strength)*float(bot.min())
            top=strength*top+(1-strength)*float(top.max())
            qf=q[good]
            A=qf[:,1:]-qf[:,:1]
            Z=np.stack((bot[good[:,1:]]-bot[good[:,:1]],top[good[:,1:]]-top[good[:,:1]]),axis=2)
            slopes=np.linalg.solve(A,Z)
            protect=MIN_PROTECTIVE_THICKNESS*np.sqrt(1+np.linalg.norm(slopes,axis=1).max()**2)
            if protect>max(MIN_PROTECTIVE_THICKNESS,scale*.25):
                continue
            bot-=protect*.5;top+=protect*.5
            n=len(q)
            verts=np.vstack((np.c_[q,bot],np.c_[q,top]))@axes+origin
            faces=[(int(c),int(b),int(a)) for a,b,c in good]
            faces += [(int(a+n),int(b+n),int(c+n)) for a,b,c in good]
            faces += [tuple(map(int,f)) for a,b in boundary for f in [(a,b,b+n),(a,b+n,a+n)]]
            incidence=Counter((a,b) for t in faces for a,b in zip(t,t[1:]+t[:1]))
            if any(h!=1 or incidence[b,a]!=1 for (a,b),h in list(incidence.items())):
                continue
            v=np.asarray(verts,dtype=np.float32).astype(float)-origin
            fv=v[np.array(faces)]
            normal=np.cross(fv[:,1]-fv[:,0],fv[:,2]-fv[:,0])
            vol=np.einsum('ij,ij->i',fv[:,0],np.cross(fv[:,1],fv[:,2])).sum()/6
            if not np.all(np.linalg.norm(normal,axis=1)>2.e-14) or vol<=1.e-15:continue
            bvh=BVHTree.FromPolygons(v.tolist(),faces,all_triangles=True)
            face_sets=[frozenset(f) for f in faces]
            if any(i<j and not face_sets[i]&face_sets[j] for i,j in bvh.overlap(bvh)):
                continue
            return verts.tolist(),faces
    raise ValueError('no covered concavity within triangle budget')

