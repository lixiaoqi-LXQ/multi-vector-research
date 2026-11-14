import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
"""Two-target K100 calibration; one shared screening pass per placement; same accepted endpoint."""
import argparse,json,random,math,hashlib
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import endpoint_search_runner as runner
from summarize import validate_result
from identity import split_identity
p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--index-placement',choices=['native','gpu-all-payload'],required=True);p.add_argument('--data',required=True);p.add_argument('--groundtruth',required=True);p.add_argument('--quality-source',required=True);p.add_argument('--output',required=True);p.add_argument('--quality-targets',default='100:.90,100:.95');p.add_argument('--threads',type=int,default=20);a=p.parse_args()
torch.set_num_threads(a.threads);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'source.py').write_bytes(Path(__file__).read_bytes())
calpath=out/'all-query-rows.npy';cal=np.arange(2931,dtype=np.int64);np.save(calpath,cal);assert len(np.load(Path(a.data)/'queries_ids.npy'))==2931;targets=[(int(v.split(':')[0]),float(v.split(':')[1])) for v in a.quality_targets.split(',')];assert set(k for k,t in targets)=={100}, 'This first-delivery worker returnsTop100 only'
actual_topk=targets[0][0]
quality_source=Path(a.quality_source);reference=json.loads(quality_source.read_text())
def qualifies(r,k,t):return round(r[f'R{k}']*len(cal)*k)>=math.ceil(Fraction(str(t))*len(cal)*k)
points=reference;assert 1<=len(points)<=12
base=dict(index=a.index,data=a.data,groundtruth=a.groundtruth,device='cuda',queries=len(cal),rounds=1,warmup=5,threads=a.threads,topk=actual_topk,query_rows=str(calpath),query_placement='host',index_placement=a.index_placement,profile_range=False,nvtx_stages=False)
(out/'state.json').write_text(json.dumps({'stage':'load-native','candidate_points':len(points)}));scorer=runner.load_scorer(SimpleNamespace(**base))
def trial(reference,label,rounds):
    point=reference['point'];cells,ndocs,threshold=point;result=out/label;args=SimpleNamespace(**(base|dict(ncells=cells,ndocs=ndocs,threshold=threshold,rounds=rounds,output=str(result))))
    runner.search(args,scorer);summary=validate_result(result,a.groundtruth,expected_rounds=rounds,expected_query_rows=str(calpath),topk=actual_topk,data=a.data,index=a.index,index_placement=a.index_placement)
    return {'point':point,'path':str(result),'mean_ms':float(np.mean([r['mean_ms'] for r in summary['rounds']])),'R10':summary['rounds'][0]['recall@10'],'R100':summary['rounds'][0].get('recall@100'),'topk':actual_topk}
results=[]
for i,r in enumerate(points):
    (out/'state.json').write_text(json.dumps({'stage':'K100-lowcost-calibration','done':i,'total':len(points)}));point=r['point'];results.append(trial(r,f'cal-{i:03d}-c{point[0]}-d{point[1]}-t{point[2]}',1));(out/'calibration-results.json').write_text(json.dumps(results,indent=2));print('B1_SCREEN',i,results[-1],flush=True)
    if i==2:print('THREE_POINT_ESTIMATE',{'query_only_seconds':sum(x['mean_ms']*.1 for x in results),'estimated_screen_query_seconds':sum(x['mean_ms']*.1 for x in results)/3*len(points),'batch_qualification':'skip if B1screen below120s; initialization/load shared'},flush=True)
finalists=results;selected={}
for k,t in targets:
    eligible=sorted([r for r in results if qualifies(r,k,t)],key=lambda x:x['mean_ms']);selected[f'R{k}>={t}']=eligible[0] if eligible else None
(out/'selection-frozen.json').write_text(json.dumps({'selection':selected,'quality_source':str(quality_source),'quality_source_sha256':hashlib.sha256(quality_source.read_bytes()).hexdigest(),'heldout_used_for_selection':False,'selection_query_count':2931,'selection_rounds':1,'heldout_enabled':False,'selection_rule':'each placement independently selects fastest eligible full2931 single-round point; no heldout; frozen before formal3','lowcost_custom_points':True,'index_placement':a.index_placement,'actual_topk':100},indent=2));(out/'state.json').write_text(json.dumps({'stage':'complete'}))
