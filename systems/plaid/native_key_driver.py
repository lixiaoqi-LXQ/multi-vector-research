from child_supervision import SupervisedPopen,run as supervised_run
"""Isolated native keypoint calibration then same-parameter allbulk end-to-end control."""
import argparse,os,json,subprocess,sys,atexit,time
from pathlib import Path
from future_telemetry import Telemetry,memory_snapshot
p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--data',required=True);p.add_argument('--groundtruth',required=True);p.add_argument('--split',required=True);p.add_argument('--quality-source',required=True);p.add_argument('--output',required=True);p.add_argument('--target',default='100:.90');a=p.parse_args()
root=Path(__file__).parent;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'driver-source.py').write_bytes(Path(__file__).read_bytes());
for helper in ['child_supervision.py','child_entry.py','endpoint_search_runner.py','future_telemetry.py','process_inventory.py']:(out/('source-'+helper)).write_bytes((root/helper).read_bytes())
env=dict(os.environ);env.pop('GOMP_CPU_AFFINITY',None);env.update(UMA_RUN_GPU_TESTS='1',OMP_NUM_THREADS='20',OMP_THREAD_LIMIT='20',OMP_DYNAMIC='FALSE',OMP_PROC_BIND='true',OMP_PLACES='{7},{5},{6},{8},{9},{15},{16},{17},{18},{19},{0},{1},{2},{3},{4},{10},{11},{12},{13},{14}',OPENBLAS_NUM_THREADS='20',MKL_NUM_THREADS='20',UMA_SEARCH_CPU_BUDGET='0-19',PATH=str(root/'.venv/bin')+os.pathsep+env.get('PATH',''),PYTHONPATH=str(root/'faiss-gpu-env/lib/python3.13/site-packages'),LD_LIBRARY_PATH=str(root/'faiss-gpu-env/lib'),CUDA_HOME='/usr/local/cuda-13.0',TORCH_CUDA_ARCH_LIST='12.1',TORCH_EXTENSIONS_DIR=str(root/'torch-extensions-cuda'))
ctx={'stage':'before-native-cal','child_pid':None,'known_child_pids':[]};mon=Telemetry(out/'cal-nvml-1s.jsonl',ctx).start();atexit.register(mon.close)
stable=0
for _ in range(60):
    util=int(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip());stable=stable+1 if util<=5 else 0;time.sleep(1)
    if stable>=10:break
assert stable>=10
cmd=['taskset','-c','0-19',sys.executable,str(root/'native_key_worker.py'),'--index',a.index,'--data',a.data,'--groundtruth',a.groundtruth,'--split',a.split,'--quality-source',a.quality_source,'--output',str(out/'native-cal'),'--quality-targets',a.target,'--threads','20']
with open(out/'native-cal.log','w') as log:
    child=SupervisedPopen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);ctx['child_pid']=child.pid;ctx['known_child_pids'].append(child.pid);ctx['stage']='native-performance-calibration';code=child.wait();ctx['child_pid']=None
assert code==0 and not Path(f'/proc/{child.pid}').exists();(out/'native-owner-released.json').write_text(json.dumps({'pid':child.pid,'proc_absent':True,**memory_snapshot()},indent=2));mon.close()
k,t=a.target.split(':');selection=out/'native-cal/selection-frozen.json';chosen=json.loads(selection.read_text())['selection'][f'R{k}>={float(t)}']
if chosen is None:(out/'state.json').write_text(json.dumps({'stage':'complete','qualified_native_keypoint':False}));sys.exit()
with open(out/'paired-control.log','w') as log:
    supervised_run([sys.executable,str(root/'placement_control_isolated.py'),'--index',a.index,'--data',a.data,'--groundtruth',a.groundtruth,'--selection',str(selection),'--target',f'R{k}>={float(t)}','--output',str(out/'paired-control'),'--threads','20','--cpus','0-19','--resident-mode','gpu-all-payload'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
(out/'state.json').write_text(json.dumps({'stage':'complete','selection':str(selection)}))
