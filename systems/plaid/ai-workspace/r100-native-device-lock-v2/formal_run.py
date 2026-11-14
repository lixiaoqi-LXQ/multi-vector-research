import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
"""LoTTE formal-only first delivery: existing frozen R100 .90/.95 configurations, full3.
Caller must hold the existing global flock; no build or unrelated QA runs here.
"""
import argparse,atexit,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from child_supervision import SupervisedPopen,lease_fds
from future_telemetry import Telemetry,memory_snapshot
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();P=Path(__file__).resolve().parent;R=P.parent.parent;sys.path.insert(0,str(R));D=Path('/home/lxq/graduate/data/multivector-colbertv2');out=Path(a.output)
assert os.environ.get('PLAID_DEVICE_LOCK')==str(Path.home()/'device.lock'),'Require current device.lock protocol'
assert lease_fds(),'Core10 formal run requires actual inherited globalflock FD'
prepared=json.loads((P/'source-freeze.json').read_text())
for name,digest in prepared.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
out.mkdir(parents=True,exist_ok=False)
files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*P.glob('*.py'),P/'points.json',R/'endpoint_search_runner.py',R/'child_supervision.py',R/'child_entry.py',R/'identity.py',R/'summarize.py',R/'future_telemetry.py',R/'process_inventory.py']}
frozen=files;(out/'source-freeze.json').write_text(json.dumps(frozen,indent=2))
env=dict(os.environ);env.pop('GOMP_CPU_AFFINITY',None);env.pop('PLAID_SEARCH_PHASE_MANIFEST',None);env.update(UMA_RUN_GPU_TESTS='1',OMP_NUM_THREADS='20',OMP_THREAD_LIMIT='20',OMP_DYNAMIC='FALSE',OMP_PROC_BIND='true',OMP_PLACES='{7},{5},{6},{8},{9},{15},{16},{17},{18},{19},{0},{1},{2},{3},{4},{10},{11},{12},{13},{14}',OPENBLAS_NUM_THREADS='20',MKL_NUM_THREADS='20',UMA_SEARCH_CPU_BUDGET='0-19',PATH=str(R/'.venv/bin')+os.pathsep+env.get('PATH',''),PYTHONPATH=str(R/'faiss-gpu-env/lib/python3.13/site-packages'),LD_LIBRARY_PATH=str(R/'faiss-gpu-env/lib'),CUDA_HOME='/usr/local/cuda-13.0',TORCH_CUDA_ARCH_LIST='12.1',TORCH_EXTENSIONS_DIR=str(R/'torch-extensions-cuda'))
ctx={'stage':'idle','child_pid':None,'known_child_pids':[]};mon=Telemetry(out/'nvml-1s.jsonl',ctx).start();atexit.register(mon.close)
(out/'external-load-before.json').write_text(json.dumps({'time':time.time(),'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),'memory':memory_snapshot()},indent=2))
common=['--index',str(D/'indexes/plaid-lotte-full-b4-km4'),'--data',str(D/'lotte-pooled-dev'),'--groundtruth',str(D/'groundtruth/lotte-pooled-dev-full')]
receipts=[]
def run(cmd,label):
 for n,s in frozen.items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==s,n
 ctx['stage']=label;(out/(label+'-argv.json')).write_text(json.dumps({'argv':cmd,'environment':{k:env[k] for k in ['UMA_RUN_GPU_TESTS','OMP_NUM_THREADS','OMP_PLACES','OMP_PROC_BIND','UMA_SEARCH_CPU_BUDGET']}},indent=2))
 with open(out/(label+'.log'),'w') as log:
  child=SupervisedPopen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);ctx['child_pid']=child.pid;ctx['known_child_pids'].append(child.pid);code=child.wait();ctx['child_pid']=None
 receipt={'label':label,'pid':child.pid,'returncode':code,'terminal_reaped':not Path(f'/proc/{child.pid}').exists(),**memory_snapshot()};receipts.append(receipt);(out/'owner-receipts.json').write_text(json.dumps(receipts,indent=2));assert code==0 and receipt['terminal_reaped']
# Existing selections are frozen inputs. No calibration or selection occurs here.
calroot=Path('/home/lxq/graduate/ai-workspace/research/multivector/plaid/ai-workspace/r100-device-lock-v1/cal-output')
selection_receipts={m:{'path':str(calroot/(m+'-cal')/'selection-frozen.json'),'sha256':hashlib.sha256((calroot/(m+'-cal')/'selection-frozen.json').read_bytes()).hexdigest()} for m in ['native']}
for mode,record in selection_receipts.items():
 selected=json.loads(Path(record['path']).read_text());assert selected['heldout_used_for_selection'] is False and selected['actual_topk']==100 and selected['index_placement']==mode
 assert set(selected['selection'])=={'R100>=0.9','R100>=0.95'}
 for target,row in selected['selection'].items():assert row is not None and row['topk']==100 and row['R100']>=float(target.split('>=')[1])
(out/'both-placements-frozen-before-heldout.json').write_text(json.dumps(selection_receipts,indent=2))
for mode in ['native']:
 run(['taskset','-c','0-19',sys.executable,str(P/'full_worker.py'),*common,'--selection',selection_receipts[mode]['path'],'--index-placement',mode,'--output',str(out/(mode+'-full3'))],mode+'-full3')
ctx['stage']='complete';(out/'complete.json').write_text(json.dumps({'complete':True,'scope':'fullLoTTE4bit B1 actualTop100 R100targets.90/.95; maxQPS withincaleligible testedparametergrid','placements':['native'],'calibration100':'prior12pointcal frozen input only; no newcal or parameter change','both_placements_frozen_before_heldout':selection_receipts,'all_children_terminal_reaped':True,'no_builds_or_ablation_or_profile':True},indent=2));mon.close();atexit.unregister(mon.close)
