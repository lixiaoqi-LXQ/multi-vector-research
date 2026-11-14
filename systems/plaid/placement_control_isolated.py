from child_supervision import SupervisedPopen,run as supervised_run
"""Placement control with a fresh process per mode; never double-reside UMA payloads."""
import argparse,atexit,json,os,subprocess,sys,time,hashlib
from pathlib import Path
import numpy as np
from future_telemetry import Telemetry,memory_snapshot
from summarize import validate_result
p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--data',required=True);p.add_argument('--groundtruth',required=True);p.add_argument('--selection',required=True);p.add_argument('--target',default='R100>=0.9');p.add_argument('--output',required=True);p.add_argument('--threads',type=int,default=20);p.add_argument('--cpus',default='0-19');p.add_argument('--resident-mode',choices=['gpu-code-payload','gpu-all-payload'],default='gpu-all-payload');p.add_argument('--skip-full',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parent;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'driver-source.py').write_bytes(Path(__file__).read_bytes());
for helper in ['child_supervision.py','child_entry.py','endpoint_search_runner.py','future_telemetry.py','process_inventory.py']:(out/('source-'+helper)).write_bytes((root/helper).read_bytes())
selection=Path(a.selection);selection_sha=hashlib.sha256(selection.read_bytes()).hexdigest();topk=int(a.target.split('>=')[0][1:]);point=json.loads(selection.read_text())['selection'][a.target]['point'];cells,ndocs,threshold=point
ctx={'stage':'before','child_pid':None,'known_child_pids':[]};monitor=Telemetry(out/'nvml-1s.jsonl',ctx).start();atexit.register(monitor.close)
allowed=[]
for token in a.cpus.split(','):
    if '-' in token:
        lo,hi=map(int,token.split('-'));allowed.extend(range(lo,hi+1))
    else:allowed.append(int(token))
assert a.threads<=len(allowed)
preference=[7,5,6,8,9,15,16,17,18,19,0,1,2,3,4,10,11,12,13,14]
places=','.join('{'+str(cpu)+'}' for cpu in preference if cpu in allowed)
env=dict(os.environ);env.pop('GOMP_CPU_AFFINITY',None);env.pop('PLAID_SEARCH_PHASE_MANIFEST',None);env.update(UMA_RUN_GPU_TESTS='1',OMP_DYNAMIC='FALSE',OMP_THREAD_LIMIT=str(a.threads),OPENBLAS_NUM_THREADS=str(a.threads),MKL_NUM_THREADS=str(a.threads),OMP_NUM_THREADS=str(a.threads),OMP_PLACES=places,OMP_PROC_BIND='true',UMA_SEARCH_CPU_BUDGET=a.cpus,PATH=str(root/'.venv/bin')+os.pathsep+env.get('PATH',''),PYTHONPATH=str(root/'faiss-gpu-env/lib/python3.13/site-packages'),LD_LIBRARY_PATH=str(root/'faiss-gpu-env/lib'),CUDA_HOME='/usr/local/cuda-13.0',TORCH_CUDA_ARCH_LIST='12.1',MAX_JOBS='2',TORCH_EXTENSIONS_DIR=str(root/'torch-extensions-cuda'))
def record(stage,**kw):
    ctx['stage']=stage
    with open(out/'phase-memory.jsonl','a') as f:f.write(json.dumps({'stage':stage,**kw,**memory_snapshot(ctx.get('child_pid'))})+'\n')
idle=[];stable=0
for _ in range(60):
    value=int(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip());idle.append(value);stable=stable+1 if value<=5 else 0;time.sleep(1)
    if stable>=10:break
(out/'idle-before.json').write_text(json.dumps(idle));assert stable>=10
nq=len(np.load(Path(a.data)/'queries_ids.npy'));rows32=out/'rows32.npy';np.save(rows32,np.arange(32,dtype=np.int64));allrows=out/'all-rows.npy';np.save(allrows,np.arange(nq,dtype=np.int64))
def run(mode,label,count,rounds):
    result=out/label;record('before-load-'+label)
    cmd=['taskset','-c',a.cpus,sys.executable,str(root/'endpoint_search_runner.py'),'search','--index',a.index,'--data',a.data,'--groundtruth',a.groundtruth,'--device','cuda','--index-placement',mode,'--query-placement','host','--queries',str(count),'--rounds',str(rounds),'--warmup','32','--threads',str(a.threads),'--topk',str(topk),'--ncells',str(cells),'--ndocs',str(ndocs),'--threshold',str(threshold),'--output',str(result)]
    if count==32:cmd+=['--query-rows',str(rows32)]
    with open(out/(label+'.log'),'w') as log:
        child=SupervisedPopen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);ctx['child_pid']=child.pid;ctx['known_child_pids'].append(child.pid);record('search-'+label);code=child.wait();ctx['child_pid']=None
    assert code==0,(label,code)
    assert not Path(f'/proc/{child.pid}').exists(),'Exited child still has proc owner'
    record('owner-released-'+label,exited_pid=child.pid,owner_proc_absent=True)
    validate_result(result,a.groundtruth,expected_rounds=rounds,expected_query_rows=str(rows32 if count==32 else allrows),topk=topk,data=a.data,index=a.index,index_placement=mode)
    return [json.loads(x) for x in (result/'per-query.jsonl').read_text().splitlines()]
qa={}
for mode in ['native',a.resident_mode]:qa[mode]=run(mode,'qa32-'+mode,32,1)
same=all(x['query_id']==y['query_id'] and x['ids']==y['ids'] for x,y in zip(qa['native'],qa[a.resident_mode]));error=max(abs(s-t) for x,y in zip(qa['native'],qa[a.resident_mode]) for s,t in zip(x['scores'],y['scores']))
(out/'placement-correctness.json').write_text(json.dumps({'queries':32,'same_ids':same,'max_abs_score_error':error,'score_tolerance':0.0,'parameters':point,'resident_mode':a.resident_mode,'selection_sha256':selection_sha,'separate_processes':True},indent=2));assert same and error==0.0
if not a.skip_full:
    for mode in ['native',a.resident_mode]:run(mode,'paired3-'+mode,nq,3)
    import itertools
    matched=0
    with open(out/'paired3-native'/'per-query.jsonl') as left,open(out/('paired3-'+a.resident_mode)/'per-query.jsonl') as right:
        for x,y in itertools.zip_longest(left,right):
            assert x is not None and y is not None,'Full paired record count mismatch'
            x,y=json.loads(x),json.loads(y)
            assert (x['round'],x['query_row'],x['query_id'],x['ids'],x['scores'])==(y['round'],y['query_row'],y['query_id'],y['ids'],y['scores']),'Full paired ID/score mismatch'
            matched+=1
    assert matched==nq*3
    (out/'full-placement-correctness.json').write_text(json.dumps({'queries':nq,'rounds':3,'matched_records':matched,'same_all_ids_and_scores_exact':True,'actual_topk':topk,'parameters':point,'threads':a.threads,'cpus':a.cpus,'boundary':'both modes endpoint_search_runner rank plus final topk slice inside timer','selection_sha256':selection_sha},indent=2))
assert hashlib.sha256(selection.read_bytes()).hexdigest()==selection_sha
record('complete')
