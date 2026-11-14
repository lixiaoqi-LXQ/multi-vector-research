"""Reviewed full offline build launcher; explicit --run, no automatic search."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,hashlib,signal
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--spec',default=str(P/'spec.json'));a=p.parse_args();spec_path=Path(a.spec).resolve();s=json.loads(spec_path.read_text())
if not a.run:print(json.dumps({'prepared_only':True,'spec':str(P/'spec.json'),'GPU_used':False}));raise SystemExit
for path,digest in {**s['control_file_sha256'],**s['launch_file_sha256']}.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
assert hashlib.sha256(Path(s['command'][s['command'].index('--provenance-json')+1]).read_bytes()).hexdigest()==s['provenance_sha256']
base=Path(s['base']);base.mkdir(exist_ok=False);import shutil
shutil.copy2(Path(__file__).resolve(),base/'frozen-launch_full.py');shutil.copy2(s['guardian_script'],base/'frozen-guard_full.py');shutil.copy2(spec_path,base/'frozen-spec.json');state=Path(s['job_state']);cmd=s['command']+['--run'];env=os.environ.copy();env.update(s['environment']);env.pop('GOMP_CPU_AFFINITY',None)
def ident(pid):
 q=Path(f'/proc/{pid}/stat').read_text();t=q[q.rfind(')')+2:].split();return {'pid':pid,'start_ticks':int(t[19])}
def write(x):
 tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));os.replace(tmp,state)
job={'status':'starting','launcher':ident(os.getpid()),'wrapper':None,'receipt_dir':cmd[cmd.index('--receipt-dir')+1],'binary':cmd[cmd.index('--binary')+1],'index':cmd[cmd.index('--index')+1],'command':cmd,'environment':s['environment'],'phase':'offline construction only','existing_GT_pid':s['existing_GT_pid']};write(job)
with open(base/'guardian.log','w') as log:guardian=subprocess.Popen([sys.executable,str(base/'frozen-guard_full.py'),'--state',str(state),'--lease-path',s['lease_path']],stdout=log,stderr=subprocess.STDOUT)
job['continuation_guardian_pid']=guardian.pid;write(job)
def owner():
 text=subprocess.check_output(['lslocks','-n','-o','PID,MODE,PATH'],text=True)
 return [int(row.split()[0]) for row in text.splitlines() if row.split() and row.split()[-1]==s['lease_path'] and '*' not in row.split()[1]]
def gt_live():
 expected=s['existing_GT_identity']
 try:
  q=ident(expected['pid']);cmdline=Path(f"/proc/{expected['pid']}/cmdline").read_bytes();raw=Path(f"/proc/{expected['pid']}/stat").read_text();st=raw[raw.rfind(')')+2:].split()[0]
  return q=={k:expected[k] for k in ('pid','start_ticks')} and st!='Z' and hashlib.sha256(cmdline).hexdigest()==expected['cmdline_sha256']
 except FileNotFoundError:return False
# The predecessor may authorize only its verified live GT cohort. Otherwise wait
# until our own successor actually owns the global lease before starting work.
while True:
 if guardian.poll() is not None:raise RuntimeError('Continuation guardian terminal before wrapper launch')
 holders=owner()
 if guardian.pid in holders:
  job['launch_lease']='successor guardian actually holding';break
 if gt_live() and holders==[s['existing_GT_guardian_pid']]:
  report=base/'guardian-state.json'
  if report.exists():
   info=json.loads(report.read_text())
   if info.get('pid')==guardian.pid and info.get('phase') in ('queued','holding'):job['launch_lease']='verified live GT + expected predecessor holds, successor queued';break
 time.sleep(.1)
write(job)
child=None;requested=None
old={}
def handler(sig,frame):
 global requested
 requested=sig
 if child is not None and child.poll() is None:child.terminate()
for sig in (signal.SIGINT,signal.SIGTERM):old[sig]=signal.signal(sig,handler)
try:
 with open(base/'driver.log','w') as log:
  if requested is not None:raise RuntimeError('Interrupted before wrapper launch')
  if guardian.poll() is not None:raise RuntimeError('Guardian died before wrapper launch')
  child=subprocess.Popen([str(Path(job['binary']).with_name('exec_guard')),str(os.getpid()),*cmd],env=env,stdout=log,stderr=subprocess.STDOUT)
  if requested is not None:child.terminate()
  job.update(status='running',wrapper=ident(child.pid));write(job);print(json.dumps({'launcher_pid':os.getpid(),'wrapper_pid':child.pid,'continuation_guardian_pid':guardian.pid,'job_state':str(state)}),flush=True)
  while child.poll() is None:
   if guardian.poll() is not None:
    job['lease_failure']='guardian exited during wrapper';child.terminate()
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:child.kill();child.wait()
    break
   time.sleep(.1)
  rc=child.wait();job.update(status='complete' if rc==0 else 'interrupted' if requested else 'failed',returncode=rc,finished_at=time.time());write(job)
finally:
 if child is not None and child.poll() is None:
  child.terminate()
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:child.kill();child.wait()
 for sig,oldhandler in old.items():signal.signal(sig,oldhandler)
print(json.dumps({'status':job['status'],'wrapper_returncode':job.get('returncode'),'formal_search_started':False}),flush=True)
raise SystemExit(job.get('returncode',1))
