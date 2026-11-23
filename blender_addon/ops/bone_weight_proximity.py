"""Bounded, local triangle clearance screening for one PPF bone mesh only.

Not a replacement for the native solve-start validator or collision masks.
Exact kernel parity tested on the current hair and skirt fixtures.
"""
import numpy as np

def check_python(v,f,offset):
    if offset<=0:return [],0
    sep=2*offset
    tri=v[f]; lo=tri.min(axis=1); hi=tri.max(axis=1)
    # This bounded NumPy experiment, not an algorithm change. Chunk if extending to a large mesh.
    p,q=np.nonzero(np.triu(np.all((hi[:,None]+sep>=lo)&(lo[:,None]-sep<=hi),axis=-1),1))
    allowed=~(f[p,:,None]==f[q,None,:]).any(axis=(1,2))
    p,q=p[allowed],q[allowed]
    if len(p)>8192:
        raise ValueError('mesh clearance exceeds the bounded local check budget')
    a,b=tri[p],tri[q]
    def norm(x):return (x*x).sum(-1)
    def dot(x,y):return (x*y).sum(-1)
    def ps(x,u,v):
      d=v-u; t=np.clip(dot(x-u,d)/np.maximum(norm(d),1.e-24),0,1)
      return norm(x-u-t[...,None]*d)
    def pt(x,t):
      m,n,g=t[:,0],t[:,1]-t[:,0],t[:,2]-t[:,0]
      pm=x-m
      nn,ng,gg=norm(n),dot(n,g),norm(g)
      pn,pg=dot(pm,n),dot(pm,g)
      denom=np.maximum(nn*gg-ng*ng,1.e-40)
      u=(pn*gg-pg*ng)/denom;v=(pg*nn-pn*ng)/denom
      interior=(u>=0)&(v>=0)&(u+v<=1)
      interior_dist=norm(pm-u[...,None]*n-v[...,None]*g)
      bd=np.min(np.array([ps(x,t[:,i],t[:,(i+1)%3]) for i in range(3)]),axis=0)
      return np.where(interior,interior_dist,bd)
    vals=[pt(a[:,i],b) for i in range(3)]+[pt(b[:,i],a) for i in range(3)]
    for i in range(3):
      x,d=a[:,i],a[:,(i+1)%3]-a[:,i]
      for j in range(3):
        y,e=b[:,j],b[:,(j+1)%3]-b[:,j]
        r=y-x; dd,ee,de=norm(d),norm(e),dot(d,e)
        den=np.maximum(dd*ee-de*de,1.e-40)
        rd,re=dot(r,d),dot(r,e)
        s=(ee*rd-de*re)/den;t=(de*rd-dd*re)/den
        internal=(s>=0)&(s<=1)&(t>=0)&(t<=1)
        vals.append(np.where(internal,norm(x+s[...,None]*d-y-t[...,None]*e),np.inf))
    pass_test=np.min(vals,axis=0)<sep*sep
    return list(zip(p[pass_test].tolist(),q[pass_test].tolist())),len(p)

