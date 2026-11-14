"""ONE pending32-real-cal-query B1/B2/B4 correctness check; root slot required."""
import argparse,hashlib,json,os,sys
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--index-placement',choices=['native','gpu-all-payload'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert os.environ.get('UMA_RUN_GPU_TESTS')=='1' and os.environ.get('PLAID_CONCURRENT_CAL_QA_AUTHORIZED')=='1'
import numpy as np,torch
from types import SimpleNamespace
from colbert.infra import ColBERTConfig
from child_supervision import lease_fds
from concurrent_calibration import screen,payload_snapshot
import endpoint_search_runner as runner
assert lease_fds(),'Actual globalflock must be inherited; no cohort or formal-lock exception'
torch.set_num_threads(20);torch.manual_seed(123);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
D=Path('/home/lxq/graduate/data/multivector-colbertv2');scorer=runner.load_scorer(SimpleNamespace(index=str(D/'indexes/plaid-lotte-prefix100000-b4-km20'),device='cuda',index_placement=a.index_placement))
rows=np.load(D/'splits/lotte-calibration-seed42/calibration-rows.npy')[:32];queries=torch.from_numpy(np.load(D/'lotte-pooled-dev/queries.npy')[rows].copy()).float()
def config():
 c=ColBERTConfig(ncells=4,ndocs=1024,centroid_score_threshold=.45,query_maxlen=32,gpus=1);c.total_visible_gpus=1;return c
cfg=config();original=payload_snapshot(scorer)
with torch.inference_mode():
 for q in queries[:4]:scorer.rank(cfg,q.unsqueeze(0).cuda().half())
 baseline=[]
 for i,q in enumerate(queries):
  ids,scores=scorer.rank(cfg,q.unsqueeze(0).cuda().half());baseline.append({'query_index':i,'ids':ids[:10],'scores':scores[:10]})
torch.cuda.synchronize();results={}
for b in [2,4]:
 observed=screen(scorer,queries,config,concurrency=b,topk=10)
 assert observed['ordered_results']==baseline,('Concurrent stream output differs from unmodified B1',b)
 results[str(b)]=observed
assert original==payload_snapshot(scorer)
(out/'qualification.json').write_text(json.dumps({'complete':True,'scope':'one32realcalB1/B2/B4 correctness only; no fusedbatch or formalQPS','placement':a.index_placement,'cal_rows':rows.tolist(),'B1':baseline,'concurrent':results,'actual_topk':10,'B2_B4_all_IDs_scores_exact_B1':True,'source_sha256':{str(R/n):hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['concurrent_calibration.py','concurrent_calibration_qa.py','endpoint_search_runner.py']},'torch_peak_allocated':torch.cuda.max_memory_allocated(),'torch_peak_reserved':torch.cuda.max_memory_reserved()},indent=2))
