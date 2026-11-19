"""Isolated Cargo cache/guard policy regression with an inert cuobjdump.

Run in a Rust linker developer shell. This never reads a real CUDA image and
is not native/SASS acceptance; the exact current fp64_guard module is used.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    source=(ROOT/'crates/ppf-cts-solver/build.rs').read_text(encoding='utf-8')
    watch='println!("cargo:rerun-if-env-changed=PPF_FORMAL_CUDA_GUARD");'
    assert watch in source
    module=source[source.index('mod fp64_guard {'):]
    assert 'const KNOWN: [&str; 0]' in module
    with tempfile.TemporaryDirectory(prefix='ppf-formal-guard-') as temporary:
        project=Path(temporary);(project/'src').mkdir();(project/'tools').mkdir()
        (project/'Cargo.toml').write_text('[package]\nname="guard-policy-fixture"\nversion="0.0.0"\nedition="2021"\n',encoding='utf-8')
        (project/'src/lib.rs').write_text('',encoding='utf-8');(project/'inert-image').write_bytes(b'not-a-CUDA-image')
        (project/'build.rs').write_text(module+'\nfn main(){'+watch+'fp64_guard::check(std::path::Path::new("inert-image"));}\n',encoding='utf-8')
        # Exists and exits successfully, deliberately supplies no SASS.
        (project/'stub.rs').write_text('fn main(){let args:Vec<_>=std::env::args().collect();if args.iter().any(|s|s=="--list-elf"){println!("ELF file 1: fixture.sm_89.cubin");}else if std::env::var_os("PPF_MOCK_SELECTOR_FAIL").is_some(){if args.iter().any(|s|s=="-arch"){std::process::exit(1);}else{println!("Function : fixture\\n /*0000*/ FADD R0, R0, R0;");}}}\n',encoding='utf-8')
        suffix='.exe' if os.name=='nt' else ''
        subprocess.run(['rustc',str(project/'stub.rs'),'-o',str(project/'tools'/('cuobjdump'+suffix))],check=True)
        env=os.environ.copy();env.pop('PPF_ALLOW_FP64',None);env.pop('PPF_FORMAL_CUDA_GUARD',None)
        env['PATH']=str(project/'tools')+os.pathsep+env.get('PATH','');env['CARGO_TARGET_DIR']=str(project/'target')
        rows=[]
        for name,formal,bypass,wanted in [('development_empty',False,False,0),
            ('same_image_first_formal_empty',True,False,101),('formal_bypass_refused',True,True,101),
            ('formal_selector_failure_no_whole_fallback',True,False,101)]:
            runenv=env.copy()
            if formal:runenv['PPF_FORMAL_CUDA_GUARD']='1'
            if bypass:runenv['PPF_ALLOW_FP64']='1'
            if name=='formal_selector_failure_no_whole_fallback':runenv['PPF_MOCK_SELECTOR_FAIL']='1'
            result=subprocess.run(['cargo','check','--offline'],cwd=project,env=runenv,capture_output=True,text=True,errors='replace')
            text=result.stdout+result.stderr;(args.out/(name+'.log')).write_text(text,encoding='utf-8')
            assert result.returncode==wanted,(name,result.returncode,text)
            if name=='same_image_first_formal_empty':assert 'produced no readable SASS' in text,text
            if name=='formal_selector_failure_no_whole_fallback':assert 'formal CUDA SASS selector failed for sm_89' in text,text
            if bypass:assert 'cannot skip the device single-precision check' in text,text
            rows.append({'case':name,'exit_code':result.returncode,'expected_exit_code':wanted,'mock_tools':True})
        (args.out/'RESULT.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print('PASS development optional -> unchanged-image formal Cargo rerun/fail-closed; formal bypass refused (inert tools only)')

if __name__=='__main__':main()
