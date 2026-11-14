from pathlib import Path
import json,os,subprocess,tempfile
P=Path(__file__).resolve().parent;binary=P/'build/ms_q1_restore_audit';cache=Path('/home/lxq/graduate/data/multivector-colbertv2/chimera-build-cache/msmarco-v1-passage-full-c4194304-b4-owned-sorted-gather-v1');raw='/home/lxq/graduate/data/multivector-colbertv2/msmarco-v1-passage/documents.npy';inputs=[str(cache/'centroids.f32'),str(cache/'assignment.cagra'),raw,str(cache/'labels/0.u32')];env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1','LD_LIBRARY_PATH':'/home/lxq/miniconda3/envs/cuvs_benchmarks/lib:/usr/local/cuda-13.0/lib64'};env.pop('UMA_RUN_GPU_TESTS',None);cases=[]
def run(args):return subprocess.run([str(binary),*args],env=env,text=True,capture_output=True,timeout=10)
r=run(['--inspect-inputs',*inputs]);assert r.returncode==0 and json.loads(r.stdout)['GPU_used'] is False;cases.append('full source metadata CPU inspect accepted without array hash or CUDA context')
with tempfile.TemporaryDirectory(dir=P) as t:
 d=Path(t);small=d/'small';small.write_bytes(bytes(512));r=run([str(small),*inputs[1:],str(d/'out')]);assert r.returncode==2 and 'Expected full C4M' in r.stderr and not (d/'out').exists();cases.append('small centroid matrix rejected before CUDA')
 existing=d/'existing';existing.mkdir();sentinel=existing/'keep';sentinel.write_text('keep');r=run([*inputs,str(existing)]);assert r.returncode==2 and 'refuse existing' in r.stderr and sentinel.read_text()=='keep';cases.append('existing output rejected/preserved before CUDA')
 r=run([*inputs,str(d/'new-output')]);assert r.returncode==2 and 'UMA_RUN_GPU_TESTS=1' in r.stderr and not (d/'new-output').exists();cases.append('GPU opt-in rejected before resource/output construction')
 r=run(['bad','args']);assert r.returncode==2;cases.append('invalid CLI arity rejected')
result={'status':'PASS_CPU_CLI_METADATA_TESTS','GPU_used':False,'cases':cases,'count':len(cases)};(P/'cpu-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
