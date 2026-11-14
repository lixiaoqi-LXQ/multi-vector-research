"""Root-authorized tiny offline-cohort correctness/tool-format qualification only."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,threading,time,uuid
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parent;BASE=ROOT.parent;D=P/'tiny-diagnostic-v1'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE/'chimera-checkpoint'));sys.path.insert(0,str(P/'controls'))
from run_search_grid_v2 import sha,check,check_thread_info,pagefault_evidence,need
from search_modes_v2 import verify_mode_log
from supervision import supervise_child
from run_checkpoint_build import exclusive_json
from child_supervision import SupervisedPopen
class DiagnosticPopen(SupervisedPopen):
 def terminate_and_reap(self,grace=25):return super().terminate_and_reap(grace)
ACTORS=[(3474856,245917949),(3482814,245934624),(3565623,246108224),(3495705,245958010)]
def read(p):return json.loads(Path(p).read_text())
def validate_freeze():
 f=read(D/'freeze.json')
 for p,h in f['files'].items():need(sha(p)==h,'Diagnostic dependency changed: '+p)
 rows=subprocess.check_output(['lslocks','-n','-o','PID,MODE,PATH'],text=True);holders=[int(x[0])for l in rows.splitlines()if len(x:=l.split())==3 and x[2]=='/tmp/uma-paper-device0.lock'and'*'not in x[1]];need(holders==[3474856],'Actual coarse guardian global holder changed')
 actual=[]
 for pid,ticks in ACTORS:
  st=Path(f'/proc/{pid}/stat').read_text().split();need(int(st[21])==ticks and st[2]not in ['Z','X'],'Offline cohort identity changed');actual.append(dict(pid=pid,start_ticks=ticks,state=st[2]))
 a=read(f['routing_audit']);need(sha(f['routing_audit'])==f['routing_audit_sha256']and a['final_cagra_file_sha256']==sha(Path(f['index'])/'centroids.carga'),'Tiny owned routing audit/graph changed');return f,actual
def command(plan,binary,out,profile=False):
 cmd=list(plan['command']);cmd[3]=str(binary);cmd[cmd.index('--warmup')+1]='32';cmd[cmd.index('--output')+1]=str(out)
 if profile:cmd+=['--profile-range']
 return cmd
def validate(stage,plan):
 p=D/stage;v=check(p/'rows.tsv',D/'gt32.npy',32,10000,100);v={k:x for k,x in v.items()if k not in ['qps','mean_ms','p50_ms','p99_ms']};v.update(actual_modes=verify_mode_log(p/'output.log',dict(algorithm_variant='full',kernel_mode='full')),OMP=check_thread_info(p/'output.log',plan['environment']),pagefaults=pagefault_evidence(p/'output.log'),performance_claim=False);need(v['pagefaults']['query_loop']['major_delta']==0,'Tiny query majorfault gate failed');exclusive_json(p/'validation.json',v)
def compare(a,b):
 import csv,numpy as np
 def values(p):
  with open(p)as f:rows=list(csv.DictReader(f,delimiter='\t'))
  return np.asarray([int(r['document_id'])for r in rows],dtype=np.int64),np.asarray([float(r['quantized_score'])for r in rows],dtype=np.float32)
 i,x=values(D/a/'rows.tsv');j,y=values(D/b/'rows.tsv');need(len(i)==3200 and np.array_equal(i,j)and np.array_equal(x.view(np.uint32),y.view(np.uint32)),'Ordered ID/F32 score bits differ');return dict(count=3200,ordered_IDs_exact=True,F32_score_bits_exact=True,ids_sha256=hashlib.sha256(i.tobytes()).hexdigest(),scores_f32_sha256=hashlib.sha256(x.tobytes()).hexdigest())
def qa(stage):
 if stage=='observation':need(read(D/'main/result.json')['passed']is True and(D/'main/validation.json').exists(),'Main QA must pass first')
 frozen,actors=validate_freeze();plan=read(D/'plan.json');out=D/stage;out.mkdir(exist_ok=False);cmd=command(plan,D/('main-search'if stage=='main'else'observation-search'),out/'rows.tsv');env=dict(os.environ,**plan['environment']);env.pop('GOMP_CPU_AFFINITY',None);timer=None;expired=threading.Event()
 with open(out/'output.lock','a')as lock,open(out/'output.log','x')as log:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  def started(child):
   nonlocal timer
   exclusive_json(out/'process.json',dict(pid=child.pid,start_ticks=int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]),command=cmd,actual_cohort=actors,timeout_s=170,cleanup_s=10,freeze_sha256=sha(D/'freeze.json'),diagnostic_only=True,performance_claim=False))
   def expire():
    if child.poll()is None:expired.set();os.kill(os.getpid(),signal.SIGTERM)
   timer=threading.Timer(170,expire);timer.start()
  try:r=supervise_child(cmd,env=env,log=log,output_lock=lock,exec_guard=BASE/'chimera-checkpoint/exec_guard',receipt_path=out/'supervision.json',write_receipt=exclusive_json,on_started=started,terminate_grace_s=10)
  finally:
   if timer:timer.cancel();timer.join()
 passed=r['returncode']==0 and r['requested_signal']is None and not expired.is_set();exclusive_json(out/'result.json',dict(passed=passed,supervision=r,performance_claim=False));need(passed,'Tiny QA failed; stop subsequent stages');validate(stage,plan)
 if stage=='observation':exclusive_json(D/'QA-acceptance.json',dict(status='PASS_TINY_MAIN_VS_OBSERVATION_32_IDS_F32_BITS',**compare('main','observation'),performance_claim=False))
 print(json.dumps(dict(stage=stage,passed=True,pid=r['child_pid'],terminal_reaped=True,performance_claim=False)),flush=True)
def trace():
 frozen,actors=validate_freeze();need(read(D/'QA-acceptance.json')['status']=='PASS_TINY_MAIN_VS_OBSERVATION_32_IDS_F32_BITS','QA first');plan=read(D/'plan.json');out=D/'nsys32';out.mkdir(exist_ok=False);cfgpath=out/'tree-config.json';lockpath=out/'diagnostic-output.lock';nonce=uuid.uuid4().hex;cmd=command(plan,D/'observation-search',out/'rows.tsv',True)
 cfg=dict(socket_name='chimera-tiny-diagnostic-'+nonce,token=nonce,lock_path=str(lockpath),engine_command=cmd,engine_binary=str(D/'observation-search'),engine_sha256=sha(D/'observation-search'),leaf_receipt=str(out/'leaf-lease.json'),tree_receipt=str(out/'tree-supervision.json'),timeout_s=175,cleanup_s=10)
 cfg['launch_command']=[plan['nsys'],'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-memory-usage=false','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','--output',str(out/'trace'),sys.executable,str(P/'leaf_exec.py'),str(cfgpath)];exclusive_json(cfgpath,cfg)
 env=dict(os.environ,**plan['environment']);env.pop('GOMP_CPU_AFFINITY',None);os.environ['PLAID_DEVICE_LOCK']=str(lockpath)
 with open(lockpath,'a')as lock,open(out/'output.log','x')as log:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);env['PLAID_DEVICE_LOCK']=str(lockpath)
  p=DiagnosticPopen([sys.executable,str(P/'tree_guard.py'),str(cfgpath),str(lock.fileno())],env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),));exclusive_json(out/'process.json',dict(pid=p.pid,start_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().split()[21]),command=cfg['launch_command'],actual_cohort=actors,diagnostic_output_lease=str(lockpath),global_device_lease_owned_by_coarse_guardian=3474856,not_a_formal_profile=True,timeout_s=175,cleanup_s=10,performance_claim=False))
  try:rc=p.wait(timeout=210)
  finally:
   if p.poll()is None:p.terminate_and_reap()
 r=read(out/'tree-supervision.json');need(rc==0 and r['all_owned_descendants_gone']and not r['timed_out'],'Tiny NSYS tree failed');leaf=read(out/'leaf-lease.json');need(len(r['lease_transfers'])==1 and r['lease_transfers'][0]['pid']==leaf['pid'],'Actual NSYS leaf diagnostic lock proof failed');validate('nsys32',plan);value=compare('observation','nsys32');exclusive_json(out/'execution.json',dict(status='TINY_TRACE_EXECUTED_AWAITING_FORMAT_PARSE',**value,trace_sha256=sha(out/'trace.nsys-rep'),tree_sha256=sha(out/'tree-supervision.json'),diagnostic_lease_only=True,performance_claim=False));print(json.dumps(dict(stage='nsys32',passed=True,guard_pid=p.pid,leaf_pid=leaf['pid'],all_owned_descendants_gone=True,performance_claim=False)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['main','observation','nsys32']);a=p.parse_args();qa(a.stage)if a.stage!='nsys32'else trace()
