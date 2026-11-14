"""Run only a root-authorized correctness helper under the held offline cohort lease."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
R=Path('/home/lxq/graduate/ai-workspace/research/multivector')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'chimera-checkpoint'))
from supervision import supervise_child
from run_checkpoint_build import exclusive_json

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase');p.add_argument('--guardian-pid',type=int);p.add_argument('--run',action='store_true');a=p.parse_args()
 plan=json.loads((ROOT/'gpu-qualification-plan.json').read_text());phase=plan['phases'][a.phase]
 if not a.run:print(json.dumps({'prepared_only':True,'GPU_used':False,'phase':phase},indent=2));return
 assert a.guardian_pid,'Current held coordinator guardian PID required'
 frozen=json.loads((ROOT/'freeze.json').read_text())
 for path,digest in frozen['files'].items():assert sha(ROOT/path)==digest,path
 for path,digest in frozen['external_files'].items():assert sha(path)==digest,path
 if phase.get('required_index_audit'):
  assert sha(phase['required_index_audit'])==phase['required_index_audit_sha256'],'Prefix routing audit receipt changed'
  audit=json.loads(Path(phase['required_index_audit']).read_text());cmd=phase['command']
  assert sha(Path(cmd[cmd.index('--index')+1])/'centroids.carga')==audit['final_cagra_file_sha256'],'Prefix route graph changed'
 rows=subprocess.check_output(['lslocks','-n','-o','PID,MODE,PATH'],text=True)
 holders=[int(v[0])for row in rows.splitlines()if len(v:=row.split())==3 and v[2]=='/tmp/uma-paper-device0.lock' and '*'not in v[1]]
 assert holders==[a.guardian_pid],f'Expected actual holder {a.guardian_pid}, observed {holders}'
 out=ROOT/'gpu-qualification'/a.phase;out.mkdir(parents=True,exist_ok=False)
 env=os.environ.copy();env.update(plan['environment']);env.pop('GOMP_CPU_AFFINITY',None)
 timed_out=threading.Event();timer=None
 with open(out/'output.lock','a')as lock,open(out/'output.log','x')as log:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  def started(child):
   nonlocal timer
   exclusive_json(out/'process.json',{'pid':child.pid,'command':phase['command'],'guardian_actual_holder':a.guardian_pid,'timeout_s':170,'terminate_grace_s':10,'binary_sha256':sha(phase['binary']),'freeze_sha256':sha(ROOT/'freeze.json'),'environment':plan['environment'],'construction_or_correctness_only':True,'performance_claim':False,'started_at_unix':time.time()})
   def expire():
    if child.poll()is None:timed_out.set();os.kill(os.getpid(),signal.SIGTERM)
   timer=threading.Timer(170,expire);timer.start()
  try:
   result=supervise_child(phase['command'],env=env,log=log,output_lock=lock,exec_guard=R/'chimera-checkpoint/exec_guard',receipt_path=out/'supervision.json',write_receipt=exclusive_json,on_started=started,terminate_grace_s=10)
  finally:
   if timer is not None:timer.cancel();timer.join()
 passed=result['returncode']==0 and result['requested_signal']is None and not timed_out.is_set()
 exclusive_json(out/'result.json',{'passed':passed,'timed_out':timed_out.is_set(),'supervision':result,'performance_claim':False})
 print(json.dumps({'phase':a.phase,'passed':passed,'child_pid':result['child_pid'],'returncode':result['returncode']}),flush=True)
 if not passed:raise SystemExit(1)
if __name__=='__main__':main()
