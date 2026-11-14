from child_supervision import SupervisedPopen,run as supervised_run
"""Cal-only native parameter and CPU organization selection, followed by paired endpoint evidence.
Invoke only in an authorized GPU window under the shared device flock.
"""
import argparse,atexit,hashlib,json,os,subprocess,sys,time,math
from fractions import Fraction
from pathlib import Path
from future_telemetry import Telemetry,memory_snapshot

ORGANIZATIONS={'all20':('0-19',20),'one7':('7',1),'fast4':('5-8',4),'fast10':('5-9,15-19',10)}
PREFERENCE=[7,5,6,8,9,15,16,17,18,19,0,1,2,3,4,10,11,12,13,14]
def organization_env(base,cpus,threads):
    allowed=[]
    for token in cpus.split(','):
        if '-' in token:
            lo,hi=map(int,token.split('-'));allowed.extend(range(lo,hi+1))
        else:allowed.append(int(token))
    assert len(allowed)==threads
    env=dict(base);env.pop('GOMP_CPU_AFFINITY',None);env.pop('PLAID_SEARCH_PHASE_MANIFEST',None)
    env.update(UMA_RUN_GPU_TESTS='1',OMP_NUM_THREADS=str(threads),OMP_THREAD_LIMIT=str(threads),OMP_DYNAMIC='FALSE',OMP_PROC_BIND='true',OMP_PLACES=','.join('{'+str(c)+'}' for c in PREFERENCE if c in allowed),OPENBLAS_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),UMA_SEARCH_CPU_BUDGET=cpus)
    return env

def main():
    p=argparse.ArgumentParser()
    for name in ['index','data','groundtruth','split','quality-source','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--target',default='100:.90');p.add_argument('--cal-only',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parent;out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    sources=['native_organization_driver.py','native_key_worker.py','endpoint_search_runner.py','placement_control_isolated.py','identity.py','summarize.py','future_telemetry.py','process_inventory.py','child_supervision.py','child_entry.py']
    frozen={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources}
    (out/'source-identity.json').write_text(json.dumps(frozen,indent=2))
    for name in sources:(out/('source-'+name)).write_bytes((root/name).read_bytes())
    base=dict(os.environ);base.update(PATH=str(root/'.venv/bin')+os.pathsep+base.get('PATH',''),PYTHONPATH=str(root/'faiss-gpu-env/lib/python3.13/site-packages'),LD_LIBRARY_PATH=str(root/'faiss-gpu-env/lib'),CUDA_HOME='/usr/local/cuda-13.0',TORCH_CUDA_ARCH_LIST='12.1',TORCH_EXTENSIONS_DIR=str(root/'torch-extensions-cuda'))
    ctx={'stage':'idle','child_pid':None,'known_child_pids':[]};mon=Telemetry(out/'nvml-1s.jsonl',ctx).start();atexit.register(mon.close)
    stable=0;idle=[]
    for _ in range(60):
        util=int(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip());idle.append({'time':time.time(),'util':util});stable=stable+1 if util<=5 else 0;time.sleep(1)
        if stable>=10:break
    (out/'idle-before.json').write_text(json.dumps(idle));assert stable>=10
    k,t=a.target.split(':');target=f'R{k}>={float(t)}'
    def run(cmd,env,label):
        for name,sha in frozen.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
        (out/(label+'-argv.json')).write_text(json.dumps({'argv':cmd,'environment':{x:env[x] for x in ['OMP_NUM_THREADS','OMP_PLACES','OMP_PROC_BIND','UMA_SEARCH_CPU_BUDGET','UMA_RUN_GPU_TESTS']}},indent=2))
        ctx['stage']=label
        with open(out/(label+'.log'),'w') as log:
            proc=SupervisedPopen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);ctx['child_pid']=proc.pid;ctx['known_child_pids'].append(proc.pid);code=proc.wait();ctx['child_pid']=None
        assert code==0,(label,code)
        assert not Path(f'/proc/{proc.pid}').exists()
        with open(out/'owner-release.jsonl','a') as f:f.write(json.dumps({'label':label,'pid':proc.pid,'proc_absent':True,**memory_snapshot()})+'\n')
    candidates=[];quality=a.quality_source
    for name,(cpus,threads) in ORGANIZATIONS.items():
        env=organization_env(base,cpus,threads)
        run(['taskset','-c',cpus,sys.executable,str(root/'native_key_worker.py'),'--index',a.index,'--data',a.data,'--groundtruth',a.groundtruth,'--split',a.split,'--quality-source',quality,'--output',str(out/name),'--quality-targets',a.target,'--threads',str(threads)],env,'cal-'+name)
        finals=json.loads((out/name/'finalists.json').read_text())
        for r in finals:
            if r.get('R'+k) is not None and round(r['R'+k]*100*int(k))>=math.ceil(Fraction(t)*100*int(k)):
                candidates.append(dict(r,organization=name,cpus=cpus,threads=threads))
        if name=='all20':
            # One shared parameter shortlist across the remaining 1/4/10-core organizations.
            eligible=sorted([r for r in candidates if r['organization']=='all20'],key=lambda r:r['mean_ms'])[:3]
            if not eligible:break
            shortlist=out/'organization-shared-shortlist.json';shortlist.write_text(json.dumps(eligible,indent=2));quality=str(shortlist)
    chosen=min(candidates,key=lambda r:r['mean_ms']) if candidates else None
    selection=out/'selection-frozen.json';selection.write_text(json.dumps({'selection':{target:chosen},'all_cal_candidates':candidates,'organizations':ORGANIZATIONS,'selection_rule':'20CPU independent cal over quality-qualified grid; same top3 eligible parameter configurations compared with 1/4/10 fast cores; fastest eligible finalist across organizations; freeze before any full/heldout','heldout_used_for_selection':False,'quality_source':a.quality_source},indent=2))
    if chosen and not a.cal_only:
        env=organization_env(base,chosen['cpus'],chosen['threads'])
        run([sys.executable,str(root/'placement_control_isolated.py'),'--index',a.index,'--data',a.data,'--groundtruth',a.groundtruth,'--selection',str(selection),'--target',target,'--output',str(out/'paired-control'),'--threads',str(chosen['threads']),'--cpus',chosen['cpus'],'--resident-mode','gpu-all-payload'],env,'paired-endpoint')
    ctx['stage']='complete';(out/'state.json').write_text(json.dumps({'stage':'complete','selection':str(selection),'paired':bool(chosen and not a.cal_only)},indent=2));mon.close()
if __name__=='__main__':main()
