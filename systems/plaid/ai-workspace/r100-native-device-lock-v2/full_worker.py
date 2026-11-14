import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
"""Frozen K100 configurations, full2931 three rounds; no parameter selection/retuning."""
import argparse,hashlib,json,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import endpoint_search_runner as runner
from summarize import validate_result
p=argparse.ArgumentParser()
for key in ['index','data','groundtruth','selection','output']:p.add_argument('--'+key,required=True)
p.add_argument('--index-placement',choices=['native','gpu-all-payload'],required=True);a=p.parse_args();torch.set_num_threads(20)
root=Path(a.output);root.mkdir(parents=True,exist_ok=False);selection=Path(a.selection);sha=hashlib.sha256(selection.read_bytes()).hexdigest();chosen=json.loads(selection.read_text())
assert chosen['heldout_used_for_selection'] is False and chosen['actual_topk']==100 and chosen['index_placement']==a.index_placement
np.save(root/'all-query-rows.npy',np.arange(2931,dtype=np.int64));assert len(np.load(Path(a.data)/'queries_ids.npy'))==2931
base=dict(index=a.index,data=a.data,groundtruth=a.groundtruth,device='cuda',threads=20,index_placement=a.index_placement,query_placement='host',profile_range=False,nvtx_stages=False,topk=100)
scorer=runner.load_scorer(SimpleNamespace(**base));accepted={}
points=sorted({tuple(r['point']) for r in chosen['selection'].values() if r is not None})
for c,d,t in points:
 out=root/f'full-c{c}-d{d}-t{t}';args=SimpleNamespace(**base,ncells=c,ndocs=d,threshold=t,queries=2931,rounds=3,warmup=32,query_rows=None,output=str(out))
 runner.search(args,scorer)
 summary=validate_result(out,a.groundtruth,expected_rounds=3,expected_query_rows=str(root/'all-query-rows.npy'),topk=100,data=a.data,index=a.index,index_placement=a.index_placement)
 mean_recall=float(np.mean([r['recall@100'] for r in summary['rounds']]))
 mean_qps=float(np.mean([r['qps'] for r in summary['rounds']])) if 'qps' in summary['rounds'][0] else float(np.mean([1000/r['mean_ms'] for r in summary['rounds']]))
 accepted[str((c,d,t))]={'summary':str(out/'summary.json'),'result_sha256':summary['result_sha256'],'complete':summary['complete'],'mean_R100':mean_recall,'mean_QPS':mean_qps,'target_qualification':{k:mean_recall>=float(k.split('>=')[1]) for k,v in chosen['selection'].items() if v and tuple(v['point'])==(c,d,t)}}
assert hashlib.sha256(selection.read_bytes()).hexdigest()==sha
(root/'completion.json').write_text(json.dumps({'complete':True,'actual_topk':100,'expected_rounds':3,'queries_per_round':2931,'selection_sha256':sha,'accepted':accepted,'frozen_target_selection':chosen['selection'],'heldout_used_for_retuning':False},indent=2))
