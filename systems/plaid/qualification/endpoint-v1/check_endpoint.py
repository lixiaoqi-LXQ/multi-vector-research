"""Actual old/new endpoint32Q qualification; no throughput aggregation."""
import argparse,hashlib,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parents[1];sys.path.insert(0,str(R))
def verify():
 f=json.loads((HERE/'source-freeze.json').read_text())
 for group in ['source_sha256','existing_extensions_sha256']:
  for path,expected in f[group].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
 return f

def check(a):
 verify();assert os.environ.get('UMA_RUN_GPU_TESTS')=='1' and os.environ.get('PLAID_ENDPOINT_QA_AUTHORIZED')=='1'
 assert os.environ.get('OMP_NUM_THREADS')=='4' and os.environ.get('OMP_PLACES')=='{15},{16},{17},{18}'
 import numpy as np,torch
 from types import SimpleNamespace
 import endpoint_search_runner as new
 import seeded_search_runner as old
 torch.set_num_threads(4);runner=old if a.endpoint=='old' else new
 root=Path(a.output);assert not root.exists();rows=root.parent/'query-rows32.npy';rows.parent.mkdir(parents=True,exist_ok=True)
 if rows.exists():assert np.array_equal(np.load(rows),np.arange(32))
 else:np.save(rows,np.arange(32,dtype=np.int64))
 data='/home/lxq/graduate/data/multivector-colbertv2/lotte-pooled-dev';index='/home/lxq/graduate/data/multivector-colbertv2/indexes/plaid-lotte-prefix100000-b4-km20';gt='/home/lxq/graduate/data/multivector-colbertv2/groundtruth/lotte-prefix-100000'
 args=SimpleNamespace(index=index,data=data,groundtruth=gt,device='cuda',queries=32,rounds=1,warmup=2,threads=4,topk=a.topk,query_rows=str(rows),query_placement='host',index_placement=a.placement,ncells=4,ndocs=1024,threshold=.45,output=str(root),profile_range=False,nvtx_stages=False,qualification_only=True)
 runner.search(args)
 records=[json.loads(line) for line in (root/'per-query.jsonl').read_text().splitlines()];assert len(records)==32
 for i,row in enumerate(records):
  assert row['round']==0 and row['query_row']==i and len(row['ids'])==len(row['scores'])==a.topk
  assert len(set(row['ids']))==a.topk and all(0<=x<100000 for x in row['ids'])
  assert np.isfinite(row['scores']).all() and np.isfinite(row['elapsed_ms']) and row['elapsed_ms']>0
  assert all(x>=y for x,y in zip(row['scores'],row['scores'][1:]))
 identity=json.loads((root/'run-identity.json').read_text());assert identity['query_placement']=='host' and identity['search_init_seed']==123
 if a.endpoint=='new':
  assert 'final host top-k slice' in identity['timing_boundary']
  memory=json.loads((root/'phase-memory-counters.json').read_text());assert max(x.get('cuda_peak_allocated_bytes',0) for x in memory)<2*1024**3
 receipt={'complete':True,'diagnostic_only':True,'endpoint':a.endpoint,'actual_topk':a.topk,'placement':a.placement,'query_rows':list(range(32)),'rows':32,'timing_boundary':identity['timing_boundary'],'all_elapsed_finite_positive':True,'throughput_not_aggregated':True,'run_identity_sha256':hashlib.sha256((root/'run-identity.json').read_bytes()).hexdigest(),'source_freeze_sha256':hashlib.sha256((HERE/'source-freeze.json').read_bytes()).hexdigest()}
 (root/'endpoint-qa.json').write_text(json.dumps(receipt,indent=2))

def compare(a):
 import json
 root=Path(a.output);modes=['native','gpu-code-payload','gpu-all-payload'];reference=None;comparisons=[]
 for mode in modes:
  rows={}
  for label in ['old100','new100','new10']:
   path=root/f'{mode}-{label}';assert json.loads((path/'endpoint-qa.json').read_text())['complete'];records=[json.loads(line) for line in (path/'per-query.jsonl').read_text().splitlines()];assert len(records)==32
   rows[label]=[(r['query_row'],r['query_id'],r['ids'],r['scores']) for r in records]
  assert rows['old100']==rows['new100'],('old/new100 mismatch',mode)
  assert rows['new10']==[(r,q,ids[:10],scores[:10]) for r,q,ids,scores in rows['new100']],('K10 slice mismatch',mode)
  if reference is None:reference=rows
  else:assert reference==rows,('placement mismatch',mode)
  comparisons.append({'placement':mode,'old100_new100_all_ids_scores_exact':True,'new10_matches_new100_prefix10_exact':True,'queries':32})
 (root/'comparison.json').write_text(json.dumps({'complete':True,'bits':4,'index_docs':100000,'original_10k_indexes_missing_identity_chain':True,'comparisons':comparisons,'all_three_placements_all_results_exact':True,'qualification_only_no_QPS':True},indent=2))

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
 c=s.add_parser('check');c.add_argument('--endpoint',choices=['old','new'],required=True);c.add_argument('--topk',type=int,choices=[10,100],required=True);c.add_argument('--placement',choices=['native','gpu-code-payload','gpu-all-payload'],required=True);c.add_argument('--output',required=True)
 c=s.add_parser('compare');c.add_argument('--output',required=True);a=p.parse_args()
 if a.cmd=='check':check(a)
 else:compare(a)
if __name__=='__main__':main()
