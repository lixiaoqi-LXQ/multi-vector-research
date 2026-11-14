from child_supervision import SupervisedPopen,run as supervised_run
"""Full-data official PLAID scale run (native default4 kmeans iterations); invoke under shared GPU flock after handoff; source frozen independently from the completed full4 phase."""
import argparse,json,os,subprocess,sys,time,atexit,hashlib
from future_telemetry import Telemetry,memory_snapshot
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--docs',type=int,required=True);p.add_argument('--kmeans-iters',type=int,default=4);p.add_argument('--data',default='/home/lxq/graduate/data/multivector-colbertv2/lotte-pooled-dev');p.add_argument('--split',default='/home/lxq/graduate/data/multivector-colbertv2/splits/lotte-calibration-seed42');p.add_argument('--bits',type=int,default=4);p.add_argument('--tag',required=True);p.add_argument('--groundtruth');p.add_argument('--centroids-from');p.add_argument('--codes-from4bit');p.add_argument('--existing-index');p.add_argument('--build-only',action='store_true');p.add_argument('--screen-only',action='store_true');p.add_argument('--quality-targets',default='10:.90,100:.90,10:.95,100:.95');p.add_argument('--host-cpus',default='0-19');p.add_argument('--host-threads',type=int,default=20);p.add_argument('--run-label');p.add_argument('--index-placement',choices=['native','gpu-code-payload','gpu-all-payload'],default='native');p.add_argument('--extend-high-recall',action='store_true');a=p.parse_args()
cpu_count=sum((int(t.split('-')[1])-int(t.split('-')[0])+1) if '-' in t else 1 for t in a.host_cpus.split(','))
assert a.host_threads<=cpu_count, 'torch threads exceed bound CPU count'
root=Path(__file__).resolve().parent;data=Path(a.data);splits=Path(a.split)
index=Path('/home/lxq/graduate/data/multivector-colbertv2/indexes')/f'plaid-{a.tag}-b{a.bits}-km{a.kmeans_iters}'
if a.existing_index:index=Path(a.existing_index)
job=root/'scale-runs'/(f'{a.tag}-b{a.bits}-km{a.kmeans_iters}'+('-'+a.index_placement if a.index_placement!='native' else '')+('-'+a.run_label if a.run_label else ''));job.mkdir(parents=True,exist_ok=False)
for name in ['future_phase_driver.py','future_telemetry.py','process_inventory.py','future_grid_worker.py','endpoint_search_runner.py','seeded_search_runner.py','runner.py','identity.py','summarize.py','reuse8_builder.py','reuse8_support.py','reuse8_validate_prefix.py','reuse8_preflight.py','child_supervision.py','child_entry.py']:(job/('source-'+name)).write_bytes((root/name).read_bytes())
(job/'driver-source-identity.json').write_text(json.dumps({name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['future_phase_driver.py','future_telemetry.py','process_inventory.py','future_grid_worker.py','endpoint_search_runner.py','seeded_search_runner.py','runner.py','identity.py','summarize.py','reuse8_builder.py','reuse8_support.py','reuse8_validate_prefix.py','reuse8_preflight.py','child_supervision.py','child_entry.py']},indent=2))
allowed=[]
for token in a.host_cpus.split(','):
    if '-' in token:
        lo,hi=map(int,token.split('-'));allowed.extend(range(lo,hi+1))
    else:allowed.append(int(token))
preference=[7,5,6,8,9,15,16,17,18,19,0,1,2,3,4,10,11,12,13,14]
places=[cpu for cpu in preference if cpu in allowed];assert len(places)==len(allowed)
env=dict(os.environ);env.pop('PLAID_SEARCH_PHASE_MANIFEST',None);env['UMA_RUN_GPU_TESTS']='1';env.pop('GOMP_CPU_AFFINITY',None);env.update(OMP_DYNAMIC='FALSE',OMP_THREAD_LIMIT=str(a.host_threads),OPENBLAS_NUM_THREADS=str(a.host_threads),MKL_NUM_THREADS=str(a.host_threads),OMP_NUM_THREADS=str(a.host_threads),OMP_PLACES=','.join('{'+str(cpu)+'}' for cpu in places),OMP_PROC_BIND='true',UMA_SEARCH_CPU_BUDGET=a.host_cpus);env['PATH']=str(root/'.venv/bin')+os.pathsep+env.get('PATH','');env.update(PYTHONPATH=str(root/'faiss-gpu-env/lib/python3.13/site-packages'),LD_LIBRARY_PATH=str(root/'faiss-gpu-env/lib'),CUDA_HOME='/usr/local/cuda-13.0',TORCH_CUDA_ARCH_LIST='12.1',MAX_JOBS='2',TORCH_EXTENSIONS_DIR=str(root/'torch-extensions-cuda'))
context={'stage':'initializing','child_pid':None,'known_child_pids':[]}
monitor=Telemetry(job/'nvml-1s.jsonl',context).start();atexit.register(monitor.close)
def state(stage,**kw):
    context['stage']=stage
    (job/'state.json').write_text(json.dumps({'stage':stage,'time':time.time(),'args':vars(a),**kw,**memory_snapshot(context.get('child_pid'))},indent=2))
def run(cmd,log):
    frozen=json.loads((job/'driver-source-identity.json').read_text())
    for name,expected in frozen.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected, 'Frozen future phase source changed: '+name
    (job/(log+'.argv.json')).write_text(json.dumps({'argv':cmd,'env':{k:env[k] for k in ['UMA_RUN_GPU_TESTS','OMP_NUM_THREADS','OMP_PLACES','OMP_PROC_BIND','UMA_SEARCH_CPU_BUDGET']}},indent=2))
    with open(job/log,'w') as f:
        process=SupervisedPopen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
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
if a.codes_from4bit:
    assert a.bits==8
    state('reuse-preflight-before-full-build')
    run(['taskset','-c',a.host_cpus,sys.executable,str(root/'reuse8_preflight.py'),'--source4',a.codes_from4bit,'--data',str(data),'--output',str(job/'reuse-preflight'),'--threads',str(a.host_threads)],'reuse-preflight.log')
if not (index/'build-provenance.json').exists():
    assert not a.existing_index, 'Existing index is incomplete'
    documents=data/'documents.npy';documents=documents if documents.exists() else data/'documents.npy.partial'
    cmd=[sys.executable,str(root/'runner.py'),'build',*common,'--documents',str(documents),'--docs',str(a.docs),'--bits',str(a.bits),'--kmeans-iters',str(a.kmeans_iters),'--build-device','cuda','--kmeans-device','cuda','--threads',str(a.host_threads),'--faiss-temp-mib','512']
    if a.centroids_from:cmd+=['--centroids-from',a.centroids_from]
    if a.codes_from4bit:
        assert a.bits==8 and a.groundtruth
        cmd=[sys.executable,str(root/'reuse8_builder.py'),'--codes-from',a.codes_from4bit,'--index',str(index),'--data',str(data),'--groundtruth',a.groundtruth,'--threads',str(a.host_threads)]
    state('building');run(['taskset','-c',a.host_cpus,*cmd],'build.log')
if a.codes_from4bit and not (job/'reuse-prefix-validation'/'validation.json').exists():
    state('reuse-prefix-correctness')
    run(['taskset','-c',a.host_cpus,sys.executable,str(root/'reuse8_validate_prefix.py'),'--source8',str(index),'--source4',a.codes_from4bit,'--data',str(data),'--output',str(job/'reuse-prefix-validation')],'reuse-prefix-validation.log')
if a.build_only:state('build-complete');sys.exit()
assert a.groundtruth,'--groundtruth required to calibrate'
state('independent-grid-and-fixed-validation')
context['detail_path']=str(job/'grid'/'grid-state.json')
cmd=['taskset','-c',a.host_cpus,sys.executable,str(root/'future_grid_worker.py'),'--index',str(index),'--data',str(data),'--groundtruth',a.groundtruth,'--output',str(job/'grid'),'--split',str(splits),'--threads',str(a.host_threads),'--quality-targets',a.quality_targets,'--index-placement',a.index_placement]
if a.screen_only:cmd.append('--screen-only')
if a.extend_high_recall:cmd.append('--extend-high-recall')
run(cmd,'grid-worker.log')
state('complete',frozen_selection=str(job/'grid'/'selection-frozen.json'))
