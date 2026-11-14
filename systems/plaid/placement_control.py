"""Same-index/parameter native vs GPU-payload placement control, after native tuning."""
import argparse,json,os,gc,hashlib,atexit,time,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import runner
from telemetry import Telemetry,memory_snapshot
from summarize import validate_result
p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--data',required=True);p.add_argument('--groundtruth',required=True);p.add_argument('--selection',required=True);p.add_argument('--target',default='R100>=0.9');p.add_argument('--output',required=True);p.add_argument('--threads',type=int,default=20);p.add_argument('--skip-full',action='store_true');p.add_argument('--resident-mode',choices=['gpu-code-payload','gpu-all-payload','gpu-resident'],default='gpu-code-payload');a=p.parse_args()
torch.set_num_threads(a.threads);root=Path(a.output);root.mkdir(parents=True,exist_ok=False);(root/'control-source.py').write_bytes(Path(__file__).read_bytes())
selection=Path(a.selection);selection_hash=hashlib.sha256(selection.read_bytes()).hexdigest();chosen=json.loads(selection.read_text())['selection'][a.target];assert chosen is not None,'Target had no calibration-qualified native configuration';cells,ndocs,threshold=chosen['point']
ctx={'stage':'before','child_pid':os.getpid(),'known_child_pids':[os.getpid()]};monitor=Telemetry(root/'nvml-1s.jsonl',ctx).start();atexit.register(monitor.close)
def state(stage):
    ctx['stage']=stage
    with open(root/'phase-memory.jsonl','a') as f:f.write(json.dumps({'stage':stage,**memory_snapshot(os.getpid())})+'\n')
idle=[];stable=0
for _ in range(60):
    value=int(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip());idle.append(value);stable=stable+1 if value<=5 else 0;time.sleep(1)
    if stable>=10:break
(root/'idle-before.json').write_text(json.dumps(idle));assert stable>=10,'GPU idle gate failed'
base=dict(index=a.index,data=a.data,groundtruth=a.groundtruth,device='cuda',rounds=1,warmup=10,threads=a.threads,helper_thread_ids=[monitor.thread.native_id],topk=100,query_rows=None,query_placement='host',ncells=cells,ndocs=ndocs,threshold=threshold,profile_range=False,nvtx_stages=False)
nq=len(np.load(Path(a.data)/'queries_ids.npy'));rows32=root/'rows32.npy';np.save(rows32,np.arange(32,dtype=np.int64));allrows=root/'all-rows.npy';np.save(allrows,np.arange(nq,dtype=np.int64))
def trial(placement,label,count,rounds,scorer):
    args=SimpleNamespace(**(base|dict(index_placement=placement,output=str(root/label),queries=count,rounds=rounds,query_rows=str(rows32) if count==32 else None,warmup=32 if rounds==3 else 10)))
    state(label);runner.search(args,scorer)
    validate_result(args.output,a.groundtruth,expected_rounds=rounds,expected_query_rows=str(rows32 if count==32 else allrows),topk=100,data=a.data,index=a.index,index_placement=placement)
    return [json.loads(x) for x in (Path(args.output)/'per-query.jsonl').read_text().splitlines()]
rows={}
for placement in ['native',a.resident_mode]:
    state('load-'+placement);scorer=runner.load_scorer(SimpleNamespace(**(base|dict(index_placement=placement))));rows[placement]=trial(placement,'qa32-'+placement,32,1,scorer);del scorer;gc.collect();torch.cuda.empty_cache()
error=max(abs(s-t) for x,y in zip(rows['native'],rows[a.resident_mode]) for s,t in zip(x['scores'],y['scores']));same=all(x['ids']==y['ids'] and x['query_id']==y['query_id'] for x,y in zip(rows['native'],rows[a.resident_mode]))
(root/'placement-correctness.json').write_text(json.dumps({'queries':32,'same_ids':same,'max_abs_score_error':error,'score_tolerance':1e-5,'moved_payloads':['codes','residuals']+(['ivf'] if a.resident_mode=='gpu-all-payload' else []),'retained_cpu_payloads':(['ivf'] if a.resident_mode!='gpu-all-payload' else [])+['doclens','offsets'],'selection_sha256':selection_hash,'parameters':[cells,ndocs,threshold]},indent=2));assert same and error<=1e-5,'Placement correctness failed'
if not a.skip_full:
    for placement in ['native',a.resident_mode]:
        state('load-'+placement);scorer=runner.load_scorer(SimpleNamespace(**(base|dict(index_placement=placement))));trial(placement,'paired3-'+placement,nq,3,scorer);del scorer;gc.collect();torch.cuda.empty_cache()
assert hashlib.sha256(selection.read_bytes()).hexdigest()==selection_hash
state('complete')
