from decimal import Decimal,localcontext
from pathlib import Path
import struct,math,subprocess,json,os
P=Path(os.environ.get('PPF_RESTORING_TEST_BUILD',Path(__file__).parent))
def f(x):return struct.unpack('f',struct.pack('f',float(x)))[0]
def narrow(x,energy=False):
 try:y=f(x)
 except OverflowError:return None
 return y if math.isfinite(y) and (energy or not x or y!=0) and (not energy or x>=0) else None
def D(x):return Decimal.from_float(f(x))
cases=[('angular',[.01,.2,.1,.3]),('angular',[1e30,1e-30,-1e30,1]),
 ('angular',[.01,.00010550889419391751,0,27*2**-149,0,0,0,0,.0004735227848868817]),
 ('angular',[.01,.00010550889419391751,0,-27*2**-149,0,0,0,0,.0004735227848868817]),
 ('angular',[1,.375,0,2**-149,0,0,0,0,.375]),
 ('angular',[1,.375,0,-2**-149,0,0,0,0,.375]),
 ('anchor',[.01,.2,0,0,0,.3,.4,.2]),('anchor',[1e30,1e-30,-1e30,0,0,1,0,0]),
 ('anchor',[.01,0,0,0,0,1e-30,0,0]),('anchor',[.01,0,0,0,0,1e20,0,0,1e-30]),
 ('stitch',[.01,.2,.1,1,0,.3,.4,.2]),('stitch',[.001,1e30,1,1,0,1e-8,0,0]),
 ('stitch',[1e-20,1e30,1,1,0,1e-8,0,0]),('stitch',[1e30,1e-30,1,1,-1e30,1,0,0]),
 ('stitch',[.01,0,1,1,0,0,0,0]),('stitch',[.01,0,.1,1,0,.1,0,0]),
 ('stitch',[.01,0,1e-30,1e-30,0,1e-30,0,0]),
 ('anchor',[1e-30,1e30,0,0,0,1,0,0])]
rows=[]
with localcontext() as context:
 context.prec=200
 for kind,args in cases:
  a=list(map(D,args));dt,c=a[:2];k=a[8] if len(a)>8 else Decimal(1);cd=c/dt
  expected=[];code=164
  if kind=='angular':
   x0,x=a[2:4];scalar=narrow(k*x*x/2,True)
   if scalar is None:code=162
   else:expected=[Decimal.from_float(scalar)+cd*(x-x0)**2/2,k*x+cd*(x-x0),k+cd]
  elif kind=='anchor':
   old=a[2:5];v=a[5:8];r=sum(x*x for x in v).sqrt();radius=narrow(r)
   if radius is not None:
    scalar=narrow(k*D(radius)**2/2,True)
    if scalar is None:code=162
    else:
     g=k*D(radius);n=[x/r if r else Decimal(0) for x in v];t=g/r if r else k
     expected=[D(scalar)+sum(cd*(x-y)**2/2 for x,y in zip(v,old))]
     expected += [g*n[i]+cd*(v[i]-old[i]) for i in range(3)]
     expected += [k*n[i]*n[j]+t*((1 if i==j else 0)-n[i]*n[j])+(cd if i==j else 0) for i in range(3) for j in range(3)]
  else:
   eps,rest,q0=a[2:5];v=a[5:8];n2=sum(x*x for x in v);r=(n2+eps*eps).sqrt();q=narrow(r-rest)
   if q is not None:
    scalar=narrow(k*D(q)**2/2,True)
    if scalar is None:code=162
    else:
     dq=D(q)-q0;g=k*D(q)+cd*dq;h=k+cd;t=g/r;rad=h*n2/(r*r)+g*eps*eps/(r*r*r)
     n=[x/n2.sqrt() if n2 else Decimal(0) for x in v]
     expected=[D(scalar)+cd*dq*dq/2,D(q),rad,t]
     expected +=[g*x/r for x in v]
     expected +=[rad*n[i]*n[j]+t*((1 if i==j else 0)-n[i]*n[j]) for i in range(3) for j in range(3)]
     expected +=[max(rad,0)*n[i]*n[j]+max(t,0)*((1 if i==j else 0)-n[i]*n[j]) for i in range(3) for j in range(3)]
  truth=[narrow(x,i==0) for i,x in enumerate(expected)]
  if kind=='angular' and len(expected)>1 and truth[1] is None and abs(expected[1])<=Decimal(2)**-150:
   truth[1]=0.0
  wanted=bool(expected) and all(x is not None for x in truth)
  run=subprocess.run([str(P/'channels_probe.exe'),kind,*[str(f(x)) for x in args]],capture_output=True,text=True,check=True)
  fields=run.stdout.split();ok=bool(int(fields[0]));got=list(map(float,fields[2:]));actualcode=int(fields[1])
  assert ok==wanted,(kind,args,run.stdout,truth,expected)
  if ok:
   assert len(got)==len(expected),(kind,len(got),len(expected))
   if kind=='angular' and len(args)>8 and abs(args[3])<=27*2**-149:
    assert f(got[1])==truth[1],(kind,args,got[1],truth[1])
   if kind=='angular':blocks=[range(0,1),range(1,2),range(2,3)]
   elif kind=='anchor':blocks=[range(0,1),range(1,4),range(4,13)]
   else:blocks=[range(0,1),range(1,2),range(2,4),range(4,7),range(7,16),range(16,25)]
   scales={i:max(abs(float(expected[j])) for j in block) for block in blocks for i in block}
   for i,(x,y) in enumerate(zip(got,expected)):
    # Independent algebra accumulates full precision; f32 derivative projection
    # has expected cancellation residuals, bounded relative to its matrix scale.
    tolerance=max(abs(float(y))*3e-6,scales[i]*4e-7,1.401298464324817e-45)
    assert abs(x-float(y))<=tolerance,(kind,args,i,x,str(y),tolerance)
  else:assert actualcode==code,(kind,args,actualcode,code)
  rows.append(dict(kind=kind,args=list(map(f,args)),ok=ok,fault=actualcode,actual=got,gold=[str(x) for x in expected]))
(P/'CHANNELS_RESULT.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
scope='real CUDA readback' if os.environ.get('PPF_RESTORING_TEST_DEVICE')=='1' else 'compiled host f32'
print('PASS',len(rows),scope,'angular/anchor/stitch cases vs Decimal200; not the full backend SASS guard')
