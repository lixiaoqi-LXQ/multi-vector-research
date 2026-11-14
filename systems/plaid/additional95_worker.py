"""One loaded official IndexScorer; independent calibration grid then frozen holdout."""
import argparse,itertools,json,random,os,hashlib,math,gc
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import endpoint_search_runner as runner
from summarize import validate_result
from future_telemetry import memory_snapshot
from identity import split_identity
p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--selection-plan',required=True);p.add_argument('--target',default='R100>=0.95');p.add_argument('--data',required=True);p.add_argument('--groundtruth',required=True);p.add_argument('--output',required=True);p.add_argument('--split',required=True);p.add_argument('--threads',type=int,default=20);p.add_argument('--quality-targets',default='10:.90,100:.90,10:.95,100:.95');p.add_argument('--screen-only',action='store_true');p.add_argument('--extend-high-recall',action='store_true');p.add_argument('--index-placement',choices=['native','gpu-code-payload','gpu-all-payload'],default='native');a=p.parse_args()
if a.index_placement!='native':assert hasattr(runner,'GPUAllPayloadIndexScorer'),'Full placement support has not yet been installed; do not run this mode during frozen100k control'
torch.set_num_threads(a.threads);root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
(root/'grid-worker-source.py').write_bytes(Path(__file__).read_bytes())
if os.environ.get('PLAID_SEARCH_PHASE_MANIFEST'):(root/'search-phase-transition.json').write_bytes(Path(os.environ['PLAID_SEARCH_PHASE_MANIFEST']).read_bytes())
(root/'split-identity.json').write_text(json.dumps(split_identity(a.split,a.data),indent=2))
calpath=Path(a.split)/'calibration-rows.npy';heldpath=Path(a.split)/'heldout-rows.npy';cal=np.load(calpath);held=np.load(heldpath)
nq=len(np.load(Path(a.data)/'queries_ids.npy'));assert set(cal).isdisjoint(held) and set(cal)|set(held)==set(range(nq))
allpath=root/'all-query-rows.npy';np.save(allpath,np.arange(nq,dtype=np.int64))
targets=[(int(t.split(':')[0]),float(t.split(':')[1])) for t in a.quality_targets.split(',')]
base=dict(index=a.index,data=a.data,groundtruth=a.groundtruth,device='cuda',queries=len(cal),rounds=1,warmup=5,threads=a.threads,topk=100,query_rows=str(calpath),query_placement='host',index_placement=a.index_placement)
def qualifies(r,k,t):
    if r.get('failed') or r.get(f'R{k}') is None:return False
    return int(round(r[f'R{k}']*len(cal)*k))>=math.ceil(Fraction(str(t))*len(cal)*k)
def state(stage,**kw):
    row={'stage':stage,**kw,**memory_snapshot(os.getpid())}
    (root/'grid-state.json').write_text(json.dumps(row,indent=2))
    with open(root/'phase-memory.jsonl','a') as f:f.write(json.dumps(row)+'\n')
state('loading-index');scorer=runner.load_scorer(SimpleNamespace(**base));state('index-loaded')
def _trial(point,label,rounds=1,full=False,topk=100):
    cells,ndocs,threshold=point;out=root/label
    args=SimpleNamespace(**(base|dict(ncells=cells,ndocs=ndocs,threshold=threshold,output=str(out),rounds=rounds,topk=topk,queries=nq if full else len(cal),query_rows=None if full else str(calpath),warmup=32 if full else 5)))
    runner.search(args,scorer)
    summary=validate_result(out,a.groundtruth,expected_rounds=rounds,expected_query_rows=str(allpath if full else calpath),topk=topk,data=a.data,index=a.index,heldout_rows=str(heldpath) if full else None,index_placement=a.index_placement)
    return {'point':point,'path':str(out),'mean_ms':float(np.mean([r['mean_ms'] for r in summary['rounds']])),'R10':summary['rounds'][0].get('recall@10'),'R100':summary['rounds'][0].get('recall@100'),'topk':topk}
def trial(point,label,rounds=1,full=False,topk=100):
    try:return _trial(point,label,rounds,full,topk)
    except torch.cuda.OutOfMemoryError as error:
        text=str(error)
    gc.collect();torch.cuda.empty_cache()
    failed={'point':point,'path':str(root/label),'topk':topk,'failed':'CUDA out of memory','error':text,'mean_ms':None,'R10':None,'R100':None,'valid_performance':False}
    (root/(label+'-failure.json')).write_text(json.dumps(failed,indent=2));return failed

plan=Path(a.selection_plan);declared=json.loads(plan.read_text());target=declared['targets'][a.target]
points=[tuple(r['point']) for r in target['calibration_repeat_shortlist']]
k=int(a.target.split('>=')[0][1:]);t=float(a.target.split('>=')[1]);targets=[(k,t)]
(root/'grid-definition.json').write_text(json.dumps({'precommitted_plan':str(plan),'precommitted_plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),'points':points,'targets':targets,'heldout_used':False},indent=2))
results=[]
for i,point in enumerate(points):
    state('calibration-finalists',done=i,total=len(points));r=trial(point,f'cal-k{k}-{i:02d}-c{point[0]}-d{point[1]}-t{point[2]}',rounds=3,topk=k);results.append(r)
(root/'calibration-results.json').write_text(json.dumps(results,indent=2))
repeats=results
(root/'calibration-finalists.json').write_text(json.dumps(repeats,indent=2));selected={}
for k,t in targets:
    qualified=sorted([r for r in repeats if r['topk']==k and qualifies(r,k,t)],key=lambda r:r['mean_ms'])
    selected[f'R{k}>={t}']=qualified[0] if qualified else None
frozen={'selection':selected,'heldout_used_for_selection':False,'definition_sha256':hashlib.sha256((root/'grid-definition.json').read_bytes()).hexdigest(),'calibration_results_sha256':hashlib.sha256((root/'calibration-results.json').read_bytes()).hexdigest(),'finalists_sha256':hashlib.sha256((root/'calibration-finalists.json').read_bytes()).hexdigest(),'timing_boundary':'preencoded host FP32 -> native rank -> final host slice topk inside timer; actual topk receipt'}
freeze=root/'selection-frozen.json';freeze.write_text(json.dumps(frozen,indent=2));freeze_hash=hashlib.sha256(freeze.read_bytes()).hexdigest()
if not a.screen_only:
    unique={(tuple(r['point']),r['topk']) for r in selected.values() if r is not None}
    for point,k in sorted(unique):
        state('fixed-full-validation',point=point,topk=k);r=trial(point,f'full-k{k}-c{point[0]}-d{point[1]}-t{point[2]}',rounds=3,full=True,topk=k)
        print('fixed-full',r,flush=True)
assert hashlib.sha256(freeze.read_bytes()).hexdigest()==freeze_hash
state('complete',selection_file=str(freeze),selection_sha256=freeze_hash)
