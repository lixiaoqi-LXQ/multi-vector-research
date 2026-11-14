"""Serial root-authorized two frozen K10 formal points."""
from pathlib import Path
import os,sys,json,hashlib,fcntl,subprocess,time,signal,argparse
P=Path(__file__).resolve().parent;R=P.parent.parent;C=R/'plans/full-owned-device-v1/controls';sys.path.insert(0,str(C));from supervision import supervise_child
p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args();freeze=json.loads((P/'freeze.json').read_text())
def need(v,m):
 if not v:raise ValueError(m)
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,x):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(path)
for path,h in freeze['files_sha256'].items():need(sha(path)==h,'Frozen file changed: '+path)
if not a.run:print(json.dumps({'prepared_only':True,'GPU_used':False,'specs':1}));raise SystemExit
base=R/'runs/lotte-r100-bounded-neighbors-v1';base.mkdir(exist_ok=False);env=os.environ.copy();env.update(UMA_RUN_GPU_TESTS='1',CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH='/home/lxq/miniconda3/envs/cuvs_benchmarks/lib:/usr/local/cuda-13.0/lib64',OPENBLAS_NUM_THREADS='1',CUDA_CACHE_PATH=str(R/'builds/cuda-core10-finalists'));env.pop('GOMP_CPU_AFFINITY',None);state={'status':'running','driver_pid':os.getpid(),'accepted_runs':[],'formal_search_launched':False};write(P/'progress.json',state)
def execute(label,cmd,cpumode=False):
 folder=P/label;folder.mkdir(exist_ok=False);e=dict(env)
 if cpumode:e['CUDA_VISIBLE_DEVICES']='-1'
 with open(folder/'output.lock','a+') as lock,open(folder/'stdout.log','x') as log:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  def started(child):
   raw=Path(f'/proc/{child.pid}/stat').read_text();v=raw[raw.rfind(')')+2:].split();state.update(current=label,child_pid=child.pid,child_start_ticks=int(v[19]));write(P/'progress.json',state);write(folder/'process.json',{'pid':child.pid,'start_ticks':int(v[19]),'command':cmd});print(label,child.pid,flush=True)
  receipt=supervise_child(cmd,env=e,log=log,output_lock=lock,exec_guard=C/'exec_guard',receipt_path=folder/'supervision.json',write_receipt=write,on_started=started,terminate_grace_s=10)
 need(receipt['returncode']==0 and receipt['requested_signal'] is None and receipt['owned_child_terminal_and_reaped'],label+' failed; no retry or later work')
try:
 for spec in freeze['specs']:
  name=Path(spec).stem;out=base/name;execute(name+'-run',[str(R/'build-env/bin/python'),str(R/'run_search_grid_device_lock.py'),'--spec',spec,'--output',str(out),'--run']);execute(name+'-accept',[str(R/'build-env/bin/python'),str(R/'accept_run_device_lock.py'),str(out)],True);state['accepted_runs'].append(str(out));write(P/'progress.json',state)
 state.update(status='complete',current=None,child_pid=None,formal_search_launched=False);write(P/'progress.json',state);print('PASS3 bounded R100 neighbors',flush=True)
except BaseException as e:
 state.update(status='stopped_on_failure',failure_type=type(e).__name__,failure=str(e),automatic_retry=False);write(P/'progress.json',state);raise
