from decimal import Decimal,localcontext
from pathlib import Path
import json,math,struct,subprocess,os
P=Path(os.environ.get('PPF_RESTORING_TEST_BUILD',Path(__file__).parent))
def f(x):return struct.unpack('f',struct.pack('f',float(x)))[0]
cases=[('small',1e-8,0,0,1,1),('small2',1e-4,0,0,1,1),
 ('large',1e20,0,0,1,1),('tiny',1e-30,0,0,1e-30,1e-30),
 ('ordinary',3,4,0,.1,1),('zero',0,0,0,1,1),
 ('compress',.1,0,0,.1,1),('cancel',1,0,0,1e-8,1),
 ('cancel2',1e20,0,0,1e10,1e20),
 ('underflow',1e-30,0,0,1,1),('extreme',1e38,1e38,1e38,1e38,1e38)]
rows=[]
with localcontext() as ctx:
 ctx.prec=200
 for name,*args in cases:
  values=list(map(f,args));d=list(map(Decimal.from_float,values))
  truth=(sum(x*x for x in d[:3])+d[3]*d[3]).sqrt()-d[4]
  try:gold=f(truth);expected=math.isfinite(gold) and (not truth or gold!=0)
  except OverflowError:gold=None;expected=False
  result=subprocess.run([str(P/'coordinate_probe.exe'),*map(str,values)],capture_output=True,text=True,check=True)
  ok,q,code,old,legacy,oldcode=result.stdout.split();ok=bool(int(ok));q=float(q)
  assert ok==expected,(name,result.stdout,str(truth),gold)
  if ok:assert abs(q-float(truth))<=max(abs(float(truth))*4e-7,1.401298464324817e-45),(name,q,str(truth))
  rows.append(dict(name=name,input=values,truth=str(truth),f32=q,ok=ok,legacy=float(legacy),legacy_ok=bool(int(old))))
(P/'COORDINATE_RESULT.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print('PASS',len(rows),'compiled host device-f32 coordinate cases; Decimal200 independent oracle; not device/SASS')
