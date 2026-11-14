from child_supervision import SupervisedPopen,run as supervised_run
"""Run one future GPU command with source freeze, idle evidence and continuous telemetry.
Caller must hold the shared device flock and have the exclusive experiment window.
"""
import argparse,atexit,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from future_telemetry import Telemetry,memory_snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
 assert os.environ.get('UMA_RUN_GPU_TESTS')=='1'
 cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command;assert cmd
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parent
 names=['guarded_search_phase.py','additional95_worker.py','endpoint_search_runner.py','identity.py','summarize.py','future_telemetry.py','process_inventory.py','child_supervision.py','child_entry.py']
 frozen={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
 for n in names:(out/('source-'+n)).write_bytes((root/n).read_bytes())
 (out/'source-identity.json').write_text(json.dumps(frozen,indent=2));(out/'argv.json').write_text(json.dumps(cmd))
 ctx={'stage':'idle','child_pid':None,'known_child_pids':[]};monitor=Telemetry(out/'nvml-1s.jsonl',ctx).start();atexit.register(monitor.close)
 idle=[];stable=0
 for _ in range(60):
  u=int(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip());idle.append({'time':time.time(),'util':u});stable=stable+1 if u<=5 else 0;time.sleep(1)
  if stable>=10:break
 (out/'idle-before.json').write_text(json.dumps(idle));assert stable>=10
 for n,s in frozen.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==s
 ctx['stage']='guarded-search-phase'
 with open(out/'child.log','w') as log:
  child=SupervisedPopen(cmd,stdout=log,stderr=subprocess.STDOUT);ctx['child_pid']=child.pid;ctx['known_child_pids'].append(child.pid);code=child.wait();ctx['child_pid']=None
 assert code==0,code
 assert not Path(f'/proc/{child.pid}').exists()
 ctx['stage']='complete';(out/'completion.json').write_text(json.dumps({'complete':True,'pid':child.pid,'owner_absent':True,**memory_snapshot()},indent=2));monitor.close()
if __name__=='__main__':main()
