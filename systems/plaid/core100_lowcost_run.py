"""LoTTE first delivery: native/allpayload independently calibrate R10 .90/.95, then frozen full3.
Caller must hold the existing global flock; no build or unrelated QA runs here.
"""
import argparse,atexit,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from child_supervision import SupervisedPopen,lease_fds
from future_telemetry import Telemetry,memory_snapshot
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();R=Path(__file__).resolve().parent;D=Path('/home/lxq/graduate/data/multivector-colbertv2');out=Path(a.output)
assert lease_fds(),'Core10 formal run requires actual inherited globalflock FD'
out.mkdir(parents=True,exist_ok=False)
files=['core100-lowcost-points.json','core100_lowcost_run.py','core100_lowcost_cal_worker.py','core100_full_worker.py','endpoint_search_runner.py','identity.py','summarize.py','child_supervision.py','child_entry.py','future_telemetry.py','process_inventory.py']
frozen={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in files};(out/'source-freeze.json').write_text(json.dumps(frozen,indent=2))
for n in files:(out/('source-'+n)).write_bytes((R/n).read_bytes())
env=dict(os.environ);env.pop('GOMP_CPU_AFFINITY',None);env.pop('PLAID_SEARCH_PHASE_MANIFEST',None);env.update(UMA_RUN_GPU_TESTS='1',OMP_NUM_THREADS='20',OMP_THREAD_LIMIT='20',OMP_DYNAMIC='FALSE',OMP_PROC_BIND='true',OMP_PLACES='{7},{5},{6},{8},{9},{15},{16},{17},{18},{19},{0},{1},{2},{3},{4},{10},{11},{12},{13},{14}',OPENBLAS_NUM_THREADS='20',MKL_NUM_THREADS='20',UMA_SEARCH_CPU_BUDGET='0-19',PATH=str(R/'.venv/bin')+os.pathsep+env.get('PATH',''),PYTHONPATH=str(R/'faiss-gpu-env/lib/python3.13/site-packages'),LD_LIBRARY_PATH=str(R/'faiss-gpu-env/lib'),CUDA_HOME='/usr/local/cuda-13.0',TORCH_CUDA_ARCH_LIST='12.1',TORCH_EXTENSIONS_DIR=str(R/'torch-extensions-cuda'))
ctx={'stage':'idle','child_pid':None,'known_child_pids':[]};mon=Telemetry(out/'nvml-1s.jsonl',ctx).start();atexit.register(mon.close)
stable=0;idle=[]
for _ in range(60):
 u=int(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip());idle.append({'time':time.time(),'utilization':u});stable=stable+1 if u<=5 else 0;time.sleep(1)
 if stable>=10:break
(out/'idle-before.json').write_text(json.dumps(idle));assert stable>=10,'GPU must be idle10s and allindexbuilders terminal beforethiscoreformalrun'
common=['--index',str(D/'indexes/plaid-lotte-full-b4-km4'),'--data',str(D/'lotte-pooled-dev'),'--groundtruth',str(D/'groundtruth/lotte-pooled-dev-full'),'--split',str(D/'splits/lotte-calibration-seed42')]
receipts=[]
def run(cmd,label):
 for n,s in frozen.items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==s,n
 ctx['stage']=label;(out/(label+'-argv.json')).write_text(json.dumps({'argv':cmd,'environment':{k:env[k] for k in ['UMA_RUN_GPU_TESTS','OMP_NUM_THREADS','OMP_PLACES','OMP_PROC_BIND','UMA_SEARCH_CPU_BUDGET']}},indent=2))
 with open(out/(label+'.log'),'w') as log:
  child=SupervisedPopen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);ctx['child_pid']=child.pid;ctx['known_child_pids'].append(child.pid);code=child.wait();ctx['child_pid']=None
 receipt={'label':label,'pid':child.pid,'returncode':code,'terminal_reaped':not Path(f'/proc/{child.pid}').exists(),**memory_snapshot()};receipts.append(receipt);(out/'owner-receipts.json').write_text(json.dumps(receipts,indent=2));assert code==0 and receipt['terminal_reaped']
# Both placements freeze both target selections before any heldout validation.
quality=R/'core100-lowcost-points.json'
for mode in ['native','gpu-all-payload']:
 run(['taskset','-c','0-19',sys.executable,str(R/'core100_lowcost_cal_worker.py'),*common,'--quality-source',str(quality),'--quality-targets','100:.90,100:.95','--threads','20','--index-placement',mode,'--output',str(out/(mode+'-cal'))],mode+'-cal')
selection_receipts={m:{'path':str(out/(m+'-cal')/'selection-frozen.json'),'sha256':hashlib.sha256((out/(m+'-cal')/'selection-frozen.json').read_bytes()).hexdigest()} for m in ['native','gpu-all-payload']}
(out/'both-placements-frozen-before-heldout.json').write_text(json.dumps(selection_receipts,indent=2))
ctx['stage']='complete';(out/'complete.json').write_text(json.dumps({'complete':True,'scope':'fullLoTTE4bit B1 actualTop100 R100targets.90/.95; maxQPS withincaleligible testedparametergrid','placements':['native','gpu-all-payload'],'calibration100':'seed42;12K100 lowcost points sharedperplacement; top3perquality repeat3','both_placements_frozen_before_heldout':selection_receipts,'all_children_terminal_reaped':True,'no_builds_or_ablation_or_profile':True,'formal_not_run':True},indent=2));mon.close();atexit.unregister(mon.close)
