"""Explicit CPU-only acceptance under the existing exact-child supervisor."""
import argparse,fcntl,hashlib,json,os,signal,sys,threading,time
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'controls'))
from supervision import supervise_child
from run_checkpoint_build import exclusive_json

def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def need(v,m):
 if not v:raise ValueError(m)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['coarse','quantization','both'],default='coarse');p.add_argument('--out',required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
 manifest=P/'cpu-acceptance-tools.json';frozen=json.loads(manifest.read_text())
 phases=['coarse','quantization']if a.phase=='both'else[a.phase]
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',UMA_RUN_GPU_TESTS='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');env.pop('GOMP_CPU_AFFINITY',None)
 definitions={'coarse':('verify_complete.py','full-CPU-acceptance.json'),'quantization':('compare_quantized_rows.py','fine-coarse-quantization-byte-comparison.json')}
 commands={name:['taskset','-c','17,18','nice','-n','10','ionice','-c','2','-n','7',sys.executable,str(P/definitions[name][0]),'--run']for name in phases}
 if not a.run:print(json.dumps({'prepared_only':True,'GPU_used':False,'full_dataset_read':False,'commands':commands,'timeout_per_phase_s':1800,'cleanup_s':10,'tools_manifest_sha256':sha(manifest)},indent=2));return
 for path,h in frozen['files'].items():need(sha(path)==h,'Frozen acceptance dependency changed: '+path)
 # No payload reads before every required build has native terminal/reaped success.
 spec=json.loads((P/'spec.json').read_text());cmd=spec['command'];coarse=Path(cmd[cmd.index('--index')+1])
 paths=[coarse]
 if 'quantization'in phases:
  import compare_quantized_rows as comparator
  paths=[comparator.FINE,comparator.COARSE]
 for index in paths:
  build=json.loads((index/'build-provenance.json').read_text());need(build.get('complete')is True and build.get('returncode')==0,'Index save/build metadata incomplete: '+str(index));receipt=Path(build['run_receipt']);result=json.loads((receipt/'result.json').read_text());s=json.loads((receipt/'supervision.json').read_text());need(result.get('complete')is True and result.get('returncode')==0 and s.get('owned_child_terminal_and_reaped')and s.get('returncode')==0,'Actual builder did not finish/reap successfully: '+str(index))
 out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
 exclusive_json(out/'prepared.json',dict(commands=commands,GPU_used=False,performance_claim=False,tools_manifest_sha256=sha(manifest),environment={k:env[k]for k in ['CUDA_VISIBLE_DEVICES','UMA_RUN_GPU_TESTS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},started_at_unix=time.time()))
 for name in phases:
  tool,record=definitions[name];target=P/record;need(not target.exists(),'Keep existing receipt; do not repeat automatically: '+str(target));phase=out/name;phase.mkdir();timer=None;expired=threading.Event()
  with open(phase/'output.lock','a')as lock,open(phase/'output.log','x')as log:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   def started(child):
    nonlocal timer
    exclusive_json(phase/'process.json',dict(pid=child.pid,command=commands[name],timeout_s=1800,cleanup_s=10))
    def expire():
     if child.poll()is None:expired.set();os.kill(os.getpid(),signal.SIGTERM)
    timer=threading.Timer(1800,expire);timer.start()
   try:result=supervise_child(commands[name],env=env,log=log,output_lock=lock,exec_guard=P/'controls/exec_guard',receipt_path=phase/'supervision.json',write_receipt=exclusive_json,on_started=started,terminate_grace_s=10)
   finally:
    if timer is not None:timer.cancel();timer.join()
  passed=result['returncode']==0 and result['requested_signal']is None and not expired.is_set()and target.exists()
  exclusive_json(phase/'result.json',dict(passed=passed,timed_out=expired.is_set(),audit_receipt=str(target)if target.exists()else None,audit_receipt_sha256=sha(target)if target.exists()else None,validator_sha256=sha(P/tool),supervision_sha256=sha(phase/'supervision.json'),GPU_used=False,root_accepted=False))
  need(passed,'CPU acceptance failed; retain outputs and inspect before another attempt')
 exclusive_json(out/'result.json',dict(status='PASS_CPU_ONLY_AWAITING_SAVED_DATASET_AUDIT_AND_ROOT_ACCEPTANCE',GPU_used=False,root_accepted=False,phases={name:dict(result=str(out/name/'result.json'),sha256=sha(out/name/'result.json'))for name in phases},tools_manifest_sha256=sha(manifest)))
 print(json.dumps({'status':'PASS_CPU_ACCEPTANCE','out':str(out),'receipt_sha256':sha(out/'result.json'),'root_accepted':False}))
if __name__=='__main__':main()
