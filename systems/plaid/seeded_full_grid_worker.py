"""One loaded official IndexScorer; independent calibration grid then frozen holdout."""
import argparse,itertools,json,random,os,hashlib,math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import seeded_search_runner as runner
from summarize import validate_result
from telemetry import memory_snapshot
from identity import split_identity
p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--data',required=True);p.add_argument('--groundtruth',required=True);p.add_argument('--output',required=True);p.add_argument('--split',required=True);p.add_argument('--threads',type=int,default=20);p.add_argument('--quality-targets',default='10:.90,100:.90,10:.95,100:.95');p.add_argument('--screen-only',action='store_true');p.add_argument('--extend-high-recall',action='store_true');p.add_argument('--index-placement',choices=['native','gpu-code-payload','gpu-all-payload'],default='native');a=p.parse_args()
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
    return int(round(r[f'R{k}']*len(cal)*k))>=math.ceil(Fraction(str(t))*len(cal)*k)
def state(stage,**kw):
    row={'stage':stage,**kw,**memory_snapshot(os.getpid())}
    (root/'grid-state.json').write_text(json.dumps(row,indent=2))
    with open(root/'phase-memory.jsonl','a') as f:f.write(json.dumps(row)+'\n')
state('loading-index');scorer=runner.load_scorer(SimpleNamespace(**base));state('index-loaded')
def trial(point,label,rounds=1,full=False):
    cells,ndocs,threshold=point;out=root/label
    args=SimpleNamespace(**(base|dict(ncells=cells,ndocs=ndocs,threshold=threshold,output=str(out),rounds=rounds,queries=nq if full else len(cal),query_rows=None if full else str(calpath),warmup=32 if full else 5)))
    runner.search(args,scorer)
    summary=validate_result(out,a.groundtruth,expected_rounds=rounds,expected_query_rows=str(allpath if full else calpath),topk=100,data=a.data,index=a.index,heldout_rows=str(heldpath) if full else None,index_placement=a.index_placement)
    return {'point':point,'path':str(out),'mean_ms':float(np.mean([r['mean_ms'] for r in summary['rounds']])),'R10':summary['rounds'][0]['recall@10'],'R100':summary['rounds'][0]['recall@100']}
points=list(itertools.product([4,8,16,32,64],[1024,2048,4096,8192],[.45,.3,0.]));random.Random(42).shuffle(points)
(root/'grid-definition.json').write_text(json.dumps({'ncells':[4,8,16,32,64],'ndocs':[1024,2048,4096,8192],'threshold':[.45,.3,0.],'order_seed':42,'points':points,'targets':targets,'calibration_rows':cal.tolist(),'heldout_count':len(held),'selection':'Cal100 only; fastest qualified point after top3 candidates repeated3 rounds; freeze before full/heldout; holdout miss is reported without retuning','query_placement':'host FP32 -> timed H2D/half -> native rank -> host IDs/scores'},indent=2))
results=[]
for i,point in enumerate(points):
    state('grid-calibration',done=i,total=len(points));r=trial(point,f'cal-{i:03d}-c{point[0]}-d{point[1]}-t{point[2]}');results.append(r)
    (root/'calibration-results.json').write_text(json.dumps(results,indent=2));print('cal',i,r,flush=True)
# High recall extensions are calibration-only and stop after the requested targets
# have three qualifying candidates; no holdout result is inspected here.
if a.extend_high_recall:
    stages=[list(itertools.product([128],[1024,2048,4096,8192],[.45,.3,0.])),*[list(itertools.product([16,32,64,128],[d],[.3,0.])) for d in [16384,32768,65536]]]
    (root/'high-recall-stages.json').write_text(json.dumps({'stages':stages,'stop':'all requested targets have at least3 calibration-qualified points','heldout_used':False},indent=2))
    for stage_id,stage in enumerate(stages):
        if all(sum(qualifies(r,k,t) for r in results)>=3 for k,t in targets):break
        random.Random(42+stage_id).shuffle(stage)
        for point in stage:
            i=len(results);state('high-recall-calibration',stage_id=stage_id,done=i)
            r=trial(point,f'cal-{i:03d}-c{point[0]}-d{point[1]}-t{point[2]}');results.append(r)
            (root/'calibration-results.json').write_text(json.dumps(results,indent=2));print('extended-cal',i,r,flush=True)
shortlist=set()
for k,t in targets:
    qualified=sorted([r for r in results if qualifies(r,k,t)],key=lambda r:r['mean_ms'])[:3]
    shortlist.update(tuple(r['point']) for r in qualified)
repeats=[]
for i,point in enumerate(sorted(shortlist)):
    state('calibration-finalists',done=i,total=len(shortlist));repeats.append(trial(point,f'finalist-{i:02d}-c{point[0]}-d{point[1]}-t{point[2]}',rounds=3))
(root/'calibration-finalists.json').write_text(json.dumps(repeats,indent=2));selected={}
for k,t in targets:
    qualified=sorted([r for r in repeats if qualifies(r,k,t)],key=lambda r:r['mean_ms'])
    selected[f'R{k}>={t}']=qualified[0] if qualified else None
frozen={'selection':selected,'heldout_used_for_selection':False,'definition_sha256':hashlib.sha256((root/'grid-definition.json').read_bytes()).hexdigest(),'calibration_results_sha256':hashlib.sha256((root/'calibration-results.json').read_bytes()).hexdigest(),'finalists_sha256':hashlib.sha256((root/'calibration-finalists.json').read_bytes()).hexdigest()}
freeze=root/'selection-frozen.json';freeze.write_text(json.dumps(frozen,indent=2));freeze_hash=hashlib.sha256(freeze.read_bytes()).hexdigest()
if not a.screen_only:
    unique={tuple(r['point']) for r in selected.values() if r is not None}
    for point in sorted(unique):
        state('fixed-full-validation',point=point);r=trial(point,f'full-c{point[0]}-d{point[1]}-t{point[2]}',rounds=3,full=True)
        print('fixed-full',r,flush=True)
assert hashlib.sha256(freeze.read_bytes()).hexdigest()==freeze_hash
state('complete',selection_file=str(freeze),selection_sha256=freeze_hash)
