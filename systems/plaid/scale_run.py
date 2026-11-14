"""Preencoded official PLAID scale run; invoke under shared GPU flock after handoff."""
import argparse,json,os,subprocess,sys,time,atexit,hashlib
from telemetry import Telemetry,memory_snapshot
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--docs',type=int,required=True);p.add_argument('--bits',type=int,default=4);p.add_argument('--tag',required=True);p.add_argument('--groundtruth');p.add_argument('--centroids-from');p.add_argument('--build-only',action='store_true');p.add_argument('--screen-only',action='store_true');p.add_argument('--quality-targets',default='10:.90,100:.90,10:.95,100:.95');p.add_argument('--host-cpus',default='7');p.add_argument('--host-threads',type=int,default=1);p.add_argument('--run-label');a=p.parse_args()
cpu_count=sum((int(t.split('-')[1])-int(t.split('-')[0])+1) if '-' in t else 1 for t in a.host_cpus.split(','))
assert a.host_threads<=cpu_count, 'torch threads exceed bound CPU count'
root=Path(__file__).resolve().parent;data=Path('/home/lxq/graduate/data/multivector-colbertv2/lotte-pooled-dev');splits=Path('/home/lxq/graduate/data/multivector-colbertv2/splits/lotte-calibration-seed42')
index=Path('/home/lxq/graduate/data/multivector-colbertv2/indexes')/f'plaid-{a.tag}-b{a.bits}-km20'
job=root/'scale-runs'/(f'{a.tag}-b{a.bits}-km20'+('-'+a.run_label if a.run_label else ''));job.mkdir(parents=True,exist_ok=True)
for name in ['scale_run.py','telemetry.py','grid_worker.py','runner.py','identity.py','summarize.py']:(job/('source-'+name)).write_bytes((root/name).read_bytes())
(job/'driver-source-identity.json').write_text(json.dumps({name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['scale_run.py','telemetry.py','grid_worker.py','runner.py','identity.py','summarize.py']},indent=2))
env=dict(os.environ);env['PATH']=str(root/'.venv/bin')+os.pathsep+env.get('PATH','');env.update(PYTHONPATH=str(root/'faiss-gpu-env/lib/python3.13/site-packages'),LD_LIBRARY_PATH=str(root/'faiss-gpu-env/lib'),CUDA_HOME='/usr/local/cuda-13.0',TORCH_CUDA_ARCH_LIST='12.1',MAX_JOBS='2',TORCH_EXTENSIONS_DIR=str(root/'torch-extensions-cuda'))
context={'stage':'initializing','child_pid':None,'known_child_pids':[]}
monitor=Telemetry(job/'nvml-1s.jsonl',context).start();atexit.register(monitor.close)
def state(stage,**kw):
    context['stage']=stage
    (job/'state.json').write_text(json.dumps({'stage':stage,'time':time.time(),'args':vars(a),**kw,**memory_snapshot(context.get('child_pid'))},indent=2))
def run(cmd,log):
    with open(job/log,'w') as f:
        process=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        context['child_pid']=process.pid;context['known_child_pids'].append(process.pid)
        code=process.wait();context['child_pid']=None
        if code:raise subprocess.CalledProcessError(code,cmd)
idle=[];stable=0
for _ in range(60):
    value=int(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip())
    idle.append({'time':time.time(),'utilization_gpu':value});stable=stable+1 if value<=5 else 0
    time.sleep(1)
    if stable>=10:break
(job/'idle-before.json').write_text(json.dumps(idle));assert stable>=10,'GPU did not become idle for10s'
common=['--data',str(data),'--index',str(index)]
if not (index/'build-provenance.json').exists():
    documents=data/'documents.npy';documents=documents if documents.exists() else data/'documents.npy.partial'
    cmd=[sys.executable,str(root/'runner.py'),'build',*common,'--documents',str(documents),'--docs',str(a.docs),'--bits',str(a.bits),'--kmeans-iters','20','--build-device','cuda','--kmeans-device','cuda','--threads','4','--faiss-temp-mib','512']
    if a.centroids_from:cmd+=['--centroids-from',a.centroids_from]
    state('building');run(cmd,'build.log')
if a.build_only:state('build-complete');sys.exit()
assert a.groundtruth,'--groundtruth required to calibrate'
state('independent-grid-and-fixed-validation')
context['detail_path']=str(job/'grid'/'grid-state.json')
cmd=['taskset','-c',a.host_cpus,sys.executable,str(root/'grid_worker.py'),'--index',str(index),'--data',str(data),'--groundtruth',a.groundtruth,'--output',str(job/'grid'),'--split',str(splits),'--threads',str(a.host_threads),'--quality-targets',a.quality_targets]
if a.screen_only:cmd.append('--screen-only')
run(cmd,'grid-worker.log')
state('complete',frozen_selection=str(job/'grid'/'selection-frozen.json'))
