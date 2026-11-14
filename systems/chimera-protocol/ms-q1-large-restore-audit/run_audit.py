"""Plan by default. Exact-owned helper supervision; no whole-corpus rehash."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,subprocess
P=Path(__file__).resolve().parent
from supervision import supervise_child
CACHE=Path('/home/lxq/graduate/data/multivector-colbertv2/chimera-build-cache/msmarco-v1-passage-full-c4194304-b4-owned-sorted-gather-v1')
RAW=Path('/home/lxq/graduate/data/multivector-colbertv2/msmarco-v1-passage/documents.npy')
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stat(p):
 s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns)
def write(p,v):
 with open(p,'x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def main():
 a=argparse.ArgumentParser();a.add_argument('--run',action='store_true');a.add_argument('--approved-freeze-sha256');args=a.parse_args();freeze=P/'freeze.json'
 output=P/'gpu-audit';cmd=[str(P/'build/ms_q1_restore_audit'),str(CACHE/'centroids.f32'),str(CACHE/'assignment.cagra'),str(RAW),str(CACHE/'labels/0.u32'),str(output/'result')]
 if not args.run:print(json.dumps(dict(prepared_only=True,GPU_used=False,command=cmd,timeout_seconds=180,expected_peak_incremental_memory_GiB=8)));return
 assert args.approved_freeze_sha256==sha(freeze),'Exact reviewed freeze required';f=json.loads(freeze.read_text())
 for p,h in f['files_sha256'].items():assert sha(Path(p))==h,p
 binding=json.loads(Path(str(CACHE)+'.identity.json').read_text());assert sha(Path(str(CACHE)+'.identity.json'))==f['cache_binding_sha256']
 expected=binding['bindings']['input'];assert stat(RAW)=={k:expected[k] for k in ('device','inode','size','mtime_ns')};assert expected['sha256']=='3d83999077ca98e3fb24f0bf2b7aa41eeb607f30ecda9d20decf6aa9ed3da690'
 for lib in f['Q1_verified_libraries_stat_bound']:
  assert stat(Path(lib['path']))=={k:lib[k] for k in ('device','inode','size','mtime_ns')},lib['path']
 identities={str(p):stat(p) for p in (CACHE/'centroids.f32',CACHE/'assignment.cagra',RAW,CACHE/'labels/0.u32',CACHE/'state.txt')};assert sha(CACHE/'state.txt')==f['checkpoint_state_sha256']
 memory={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')};assert memory['MemAvailable']>=32*1024**3,'Insufficient MemAvailable'
 locks=subprocess.check_output(['lslocks','-n','-o','PID,MODE,PATH'],text=True);lease=[x for x in locks.splitlines() if '/tmp/uma-paper-device0.lock' in x];holders=[int(x.split()[0]) for x in lease if 'WRITE' in x and 'WRITE*' not in x];assert holders==[3228697],'Authorized offline holder changed'
 t=Path('/proc/3233295/stat').read_text();fields=t[t.rfind(')')+2:].split();assert fields[0]!='Z' and int(fields[19])==245395633,'LoTTE native identity no longer live'
 cuda_cache=P/'builds/cuda-cache';cuda_cache.mkdir(parents=True,exist_ok=True)
 output.mkdir(exist_ok=False);write(output/'prepared.json',dict(command=cmd,freeze_sha256=sha(freeze),input_identities=identities,launch_gate=dict(memory=memory,actual_holder=holders,lease_lines=lease,LoTTE_native_pid=3233295,LoTTE_native_start_ticks=245395633),timeout_seconds=180,memory_budget_GiB=8,source_full_sha_reused_stat_bound=True))
 import fcntl
 with open(output/'helper-output.lock','x') as lock,open(output/'helper.log','x') as log:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic()
  def poll(child):
   if time.monotonic()-started>175:raise TimeoutError('175s work +5s termination grace audit ceiling')
  def on_started(child):write(output/'process.json',dict(pid=child.pid))
  env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','UMA_RUN_GPU_TESTS':'1','CUDA_CACHE_PATH':str(cuda_cache),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','LD_LIBRARY_PATH':'/home/lxq/miniconda3/envs/cuvs_benchmarks/lib:/usr/local/cuda-13.0/lib64'}
  result=supervise_child(['taskset','-c','7',*cmd],env=env,log=log,output_lock=lock,exec_guard=P/'exec_guard',receipt_path=output/'supervision.json',write_receipt=write,on_started=on_started,on_poll=poll)
 assert result['returncode']==0 and result['requested_signal'] is None
 result_path=output/'result/result.json';r=json.loads(result_path.read_text());assert r['status']=='PASS_FULL_MS_Q1_LARGE_RESTORE';assert r['centroid_sha256']==f['expected_centroid_sha256'] and r['assignment_graph_sha256']==f['expected_graph_sha256'];assert r['q1_labels_sha256']==f['expected_labels_sha256'];assert all(stat(Path(p))==v for p,v in identities.items());write(output/'validation.json',dict(status='PASS_FULL_MS_Q1_LARGE_RESTORE_BOUND',result_sha256=sha(result_path),freeze_sha256=sha(freeze),input_stat_preserved=True,elapsed_seconds=time.monotonic()-started));print(str(output/'validation.json'))
if __name__=='__main__':main()
