import argparse, os, pathlib, subprocess, zipfile
p=argparse.ArgumentParser();p.add_argument('--blender',required=True);p.add_argument('--work-dir',required=True);a=p.parse_args()
here=pathlib.Path(__file__).resolve().parent;source=here.parents[1];out=pathlib.Path(a.work_dir).resolve()
if out.exists() and any(out.iterdir()):raise SystemExit('work-dir must be empty')
out.mkdir(parents=True,exist_ok=True)
wheel=next((source/'blender_addon'/'wheels').glob('*win_amd64.whl'))
with zipfile.ZipFile(wheel) as z:z.extractall(out/'test_deps')
env=os.environ.copy();env.update(PPF_SAVEAS_TEST_ROOT=str(out),PPF_SAVEAS_SOURCE=str(source))
for name,args in [('lifecycle',[]),('cold',['saved']),('cold',['copy']),('cold',['partial']),('cold',['recovered']),('cold',['badsource']),('failure',[]),('samepath',['run_integrated']),('shared',[]),('cold',['shared']),('conflict',[]),('panel',[])]:
 cmd=[a.blender,'--background','--factory-startup','--python-exit-code','1','--python',str(here/(name+'.py')),'--',*args]
 with (out/(name+('_'+args[0] if name=='cold' else '')+'.log')).open('w',encoding='utf8') as log:
  result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise SystemExit(f'{name} failed ({result.returncode}); see {log.name}')
 print(name,args,'PASS',flush=True)
