from pathlib import Path
import os,sys,json,time,fcntl,subprocess,hashlib,shutil
P=Path(__file__).resolve().parent;R=P.parent;B=R.parent;sys.path.insert(0,str(R));from search_supervision_v2 import supervise_child
s=json.loads((P/'spec.json').read_text());run=Path(s['base']);state=Path(s['job_state']);cmd=s['command'];receipt=Path(cmd[cmd.index('--receipt-dir')+1]);C=Path(cmd[cmd.index('--binary')+1]).parent

def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def need(v,m):
 if not v:raise ValueError(m)
def ident(pid):
 raw=Path(f'/proc/{pid}/stat').read_text();v=raw[raw.rfind(')')+2:].split();return {'pid':pid,'start_ticks':int(v[19])}
def write(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
for p,h in s['control_file_sha256'].items():need(sha(p)==h,'Original immutable control changed: '+p)
need(sha(cmd[cmd.index('--provenance-json')+1])==s['provenance_sha256'],'Original provenance changed');need(not Path(cmd[cmd.index('--index')+1]).exists(),'Final index already exists');need(not receipt.exists(),'Attempt receipt exists');need(shutil.disk_usage(s['disk_path']).free>=s['minimum_disk_free_bytes'],'Insufficient disk');available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024;need(available>=s['minimum_memavailable_bytes'],'Insufficient MemAvailable')
run.mkdir(exist_ok=False);job={'status':'starting','driver':ident(os.getpid()),'phase':'exclusive preflight','current_child':None,'native':None,'index':cmd[cmd.index('--index')+1],'cache':cmd[cmd.index('--cache')+1],'receipt_dir':str(receipt),'binary':cmd[cmd.index('--binary')+1],'only_primary_4bit':True,'no_search':True};write(state,job);env=os.environ.copy();env.update(s['environment']);env.pop('GOMP_CPU_AFFINITY',None);lock=open('/home/lxq/device.lock','a+');guardian=None
try:
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  job.update(status='lock_busy',phase='device busy; no queued wait/no build launched');write(state,job);raise SystemExit(75)
 guardian=subprocess.Popen([sys.executable,str(P/'guard_fd.py'),'--fd',str(lock.fileno()),'--state',str(state)],pass_fds=(lock.fileno(),),stdin=subprocess.DEVNULL);job['guardian']=ident(guardian.pid);write(state,job)
 last=[0];offset=[0]
 def monitor(child):
  now=time.monotonic()
  if now-last[0]<1:return
  last[0]=now;nf=receipt/'process.json';target=child.pid
  if nf.exists():
   n=json.loads(nf.read_text())
   try:
    v=ident(n['pid']);args=Path(f"/proc/{n['pid']}/cmdline").read_bytes().decode(errors='replace')
    if job['binary'] in args and job['index'] in args:job['native']=v;target=n['pid'];write(state,job)
   except (FileNotFoundError,ProcessLookupError):pass
  sample={'time':time.time(),'phase':job['phase'],'pid':target}
  try:
   sample['process_memory']={k:val.strip() for line in Path(f'/proc/{target}/status').read_text().splitlines() for k,sep,val in [line.partition(':')] if k in ('VmRSS','VmHWM','VmSwap','Cpus_allowed_list')}
   sample['io']=Path(f'/proc/{target}/io').read_text()
  except (FileNotFoundError,PermissionError,ProcessLookupError) as e:sample['process_unavailable']=type(e).__name__
  sample['system_memory']={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.split(':')[0] in ('MemFree','MemAvailable','Cached')};f=receipt/'build.log'
  if f.exists():
   with f.open() as log:log.seek(offset[0]);new=log.read();offset[0]=log.tell()
   if new:
    sample['new_build_events']=new.splitlines();job['last_event']=new.splitlines()[-1];write(state,job)
  with (run/'telemetry.jsonl').open('a') as out:out.write(json.dumps(sample)+'\n')
 def execute(label,command,cpu=False):
  folder=run/label;folder.mkdir(exist_ok=False);e=dict(env)
  if cpu:e['CUDA_VISIBLE_DEVICES']='-1'
  job['phase']=label;write(state,job)
  with (folder/'stdout.log').open('x') as log:
   def started(c):job['current_child']=ident(c.pid);write(state,job);print(label,c.pid,flush=True)
   q=supervise_child(command,env=e,log=log,output_lock=lock,exec_guard=C/'exec_guard',receipt_path=folder/'supervision.json',write_receipt=write,on_started=started,on_poll=monitor,terminate_grace_s=15)
  job['current_child']=None;write(state,job);need(q['returncode']==0 and q['requested_signal'] is None,label+' failed; retained receipts');return folder
 pre=execute('cache-preflight',['taskset','-c','15',sys.executable,str(B/'chimera-ms-sorted-gather/resume-plan/check_stopped_cache.py'),'--spec',str(B/'chimera-ms-sorted-gather/resume-plan/spec.json'),'--hash-artifacts'],True);audit=json.loads((pre/'stdout.log').read_text());need(audit['status']=='PASS_READ_ONLY_STOPPED_COMMITTED_PREFIX','Cache preflight failed');write(run/'cache-validation.json',audit)
 execute('build-wrapper',cmd+['--run']);result=json.loads((receipt/'result.json').read_text());need(result['complete'] and result['returncode']==0,'Wrapper final save not complete');job['native']=None;job.update(status='build_saved',returncode=0);write(state,job)
 checked=execute('full-CPU-validation',['taskset','-c','0-19',sys.executable,str(P/'verify_complete.py'),'--run'],True)
 # The CPU gate expects the supervisor's completed-build terminal state.
 job.update(status='complete',phase='saved and fully CPU validated; pending root semantic audit review',returncode=0,finished_at=time.time(),complete_index=True);write(state,job)
except BaseException as error:
 if job['status']!='lock_busy':job.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',failure=str(error),finished_at=time.time());write(state,job)
 raise
finally:
 if guardian:
  try:guardian.wait(timeout=10)
  except subprocess.TimeoutExpired:guardian.terminate();guardian.wait(timeout=5)
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
print('COMPLETE MS primary4 saved andvalidated; lease released',flush=True)
