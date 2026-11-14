"""Queue a global lease continuation; never launches/kills GPU work."""
import argparse,os,time,json,fcntl
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--state',required=True);p.add_argument('--lease-path',default='/tmp/uma-paper-device0.lock');a=p.parse_args();state=Path(a.state);report=state.with_name('guardian-state.json')
def proc(pid):
 try:
  x=Path(f'/proc/{pid}/stat').read_text();s=x[x.rfind(')')+2:].split();return {'state':s[0],'start_ticks':int(s[19])}
 except FileNotFoundError:return None
def live(identity):
 if not identity:return False
 q=proc(identity['pid']);return q is not None and q['state']!='Z' and q['start_ticks']==identity['start_ticks']
def write(x):
 tmp=report.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));os.replace(tmp,report)
lock=open(a.lease_path,'a+');write({'pid':os.getpid(),'phase':'queued','time':time.time()});fcntl.flock(lock,fcntl.LOCK_EX);write({'pid':os.getpid(),'phase':'holding','time':time.time()})
try:
 while True:
  job=json.loads(state.read_text());wrapper=job.get('wrapper');launcher=job.get('launcher')
  # A missing wrapper publication is safe: launcher uses checked PDEATHSIG exec_guard.
  active=live(wrapper) or (wrapper is None and live(launcher))
  nativefile=Path(job['receipt_dir'])/'process.json';native=None
  if nativefile.exists():
   n=json.loads(nativefile.read_text());q=proc(n['pid'])
   if q and q['state']!='Z':
    try:cmd=Path(f"/proc/{n['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except FileNotFoundError:cmd=''
    if job['binary'] in cmd and job['index'] in cmd:native=n['pid'];active=True
  write({'pid':os.getpid(),'phase':'holding','wrapper':wrapper,'native_pid_if_live':native,'job_status':job.get('status'),'time':time.time()})
  if not active:break
  time.sleep(1)
finally:
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close();write({'pid':os.getpid(),'phase':'released','time':time.time()})
