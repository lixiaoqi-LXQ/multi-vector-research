"""Native performance calibration reusing placement-invariant allbulk quality evidence."""
import argparse,json,random,math,hashlib
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import endpoint_search_runner as runner
from summarize import validate_result
from identity import split_identity
p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--data',required=True);p.add_argument('--groundtruth',required=True);p.add_argument('--split',required=True);p.add_argument('--quality-source',required=True);p.add_argument('--output',required=True);p.add_argument('--quality-targets',default='100:.90');p.add_argument('--threads',type=int,default=20);a=p.parse_args()
torch.set_num_threads(a.threads);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'source.py').write_bytes(Path(__file__).read_bytes());(out/'split-identity.json').write_text(json.dumps(split_identity(a.split,a.data),indent=2))
calpath=Path(a.split)/'calibration-rows.npy';cal=np.load(calpath);targets=[(int(v.split(':')[0]),float(v.split(':')[1])) for v in a.quality_targets.split(',')];assert len(targets)==1, 'Use one target per isolated native phase'
actual_topk=targets[0][0]
quality_source=Path(a.quality_source);reference=json.loads(quality_source.read_text())
def qualifies(r,k,t):return round(r[f'R{k}']*len(cal)*k)>=math.ceil(Fraction(str(t))*len(cal)*k)
points=[r for r in reference if any(qualifies(r,k,t) for k,t in targets)];random.Random(42).shuffle(points)
base=dict(index=a.index,data=a.data,groundtruth=a.groundtruth,device='cuda',queries=len(cal),rounds=1,warmup=5,threads=a.threads,topk=actual_topk,query_rows=str(calpath),query_placement='host',index_placement='native',profile_range=False,nvtx_stages=False)
(out/'state.json').write_text(json.dumps({'stage':'load-native','candidate_points':len(points)}));scorer=runner.load_scorer(SimpleNamespace(**base))
def trial(reference,label,rounds):
    point=reference['point'];cells,ndocs,threshold=point;result=out/label;args=SimpleNamespace(**(base|dict(ncells=cells,ndocs=ndocs,threshold=threshold,rounds=rounds,output=str(result))))
    runner.search(args,scorer);summary=validate_result(result,a.groundtruth,expected_rounds=rounds,expected_query_rows=str(calpath),topk=actual_topk,data=a.data,index=a.index,index_placement='native')
    observed=[json.loads(s) for s in (result/'per-query.jsonl').read_text().splitlines()];expected=[json.loads(s) for s in (Path(reference['path'])/'per-query.jsonl').read_text().splitlines()][:len(cal)]
    assert len(expected)==len(cal)
    for rr in range(rounds):
        for x,y in zip(observed[rr*len(cal):(rr+1)*len(cal)],expected):assert (x['query_row'],x['ids'],x['scores'])==(y['query_row'],y['ids'][:actual_topk],y['scores'][:actual_topk]),'Placement bitwise calibration mismatch'
    return {'point':point,'path':str(result),'reference_path':reference['path'],'mean_ms':float(np.mean([r['mean_ms'] for r in summary['rounds']])),'R10':summary['rounds'][0]['recall@10'],'R100':summary['rounds'][0].get('recall@100'),'bitwise_cal_agreement':True,'topk':actual_topk}
results=[]
for i,r in enumerate(points):
    (out/'state.json').write_text(json.dumps({'stage':'native-performance-calibration','done':i,'total':len(points)}));point=r['point'];results.append(trial(r,f'cal-{i:03d}-c{point[0]}-d{point[1]}-t{point[2]}',1));(out/'calibration-results.json').write_text(json.dumps(results,indent=2))
shortlist={tuple(r['point']) for k,t in targets for r in sorted([x for x in results if qualifies(x,k,t)],key=lambda x:x['mean_ms'])[:3]};finalists=[]
for i,point in enumerate(sorted(shortlist)):
    ref=next(r for r in reference if tuple(r['point'])==point);finalists.append(trial(ref,f'finalist-{i:02d}-c{point[0]}-d{point[1]}-t{point[2]}',3))
(out/'finalists.json').write_text(json.dumps(finalists,indent=2));selected={}
for k,t in targets:
    eligible=sorted([r for r in finalists if qualifies(r,k,t)],key=lambda x:x['mean_ms']);selected[f'R{k}>={t}']=eligible[0] if eligible else None
(out/'selection-frozen.json').write_text(json.dumps({'selection':selected,'quality_source':str(quality_source),'quality_source_sha256':hashlib.sha256(quality_source.read_bytes()).hexdigest(),'heldout_used_for_selection':False,'selection_rule':'native own latency on every quality-qualified point; fastest3 repeat calibration3 rounds; no heldout feedback','all_retained_cal_outputs_bitwise_equal_to_allbulk':True},indent=2));(out/'state.json').write_text(json.dumps({'stage':'complete'}))
