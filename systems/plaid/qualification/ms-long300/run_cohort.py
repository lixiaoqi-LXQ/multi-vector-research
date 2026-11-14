"""Authorized diagnostic cohort, exact guard/supervision; never claims global lease ownership."""
import argparse,atexit,fcntl,hashlib,importlib.util,json,os,signal,sys,threading,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parents[1];sys.path.insert(0,str(R))
from process_inventory import process_identity
from cohort_supervision import supervise_child
import pynvml as nv
LOCK=Path('/tmp/uma-paper-device0.lock')
EXPECTED={3474856:245917949,3474837:245917939,3482814:245934624,3565623:246108224,3495705:245958010}
def write(path,value):path.write_text(json.dumps(value,indent=2))
def gate():
 actual={str(p):process_identity(p) for p in EXPECTED}
 for p,start in EXPECTED.items():
  if actual[str(p)] is None:
   assert p in [3565623,3495705],f'Required coarse holder cohort PID missing: {p}'
  else:assert actual[str(p)]['starttime_ticks']==start,(p,actual[str(p)])
 st=LOCK.stat();key=f'{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}'
 holders=[]
 for line in Path('/proc/locks').read_text().splitlines():
  f=line.split()
  if '->' not in f and len(f)>5 and f[1:4]==['FLOCK','ADVISORY','WRITE'] and f[5]==key:holders.append(int(f[4]))
 assert holders==[3474856],('Actual global WRITE holder changed',holders)
 return {'expected_pid_starttime':EXPECTED,'observed':actual,'global_lock_key':key,'actual_global_write_holders':holders,'authorization':'root explicit long300 diagnostic alongside offline build cohort; this process does not own global flock'}
def verify():
 freeze=json.loads((HERE/'source-freeze.json').read_text())
 for path,expected in freeze['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
 for path,expected in freeze['existing_extensions_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
 return freeze
class Monitor:
 def __init__(self,out,ctx):self.out=out;self.ctx=ctx;self.stop=threading.Event();self.rows=[];self.errors=[]
 def start(self):
  def loop():
   try:
    os.sched_setaffinity(threading.get_native_id(),{0});nv.nvmlInit();h=nv.nvmlDeviceGetHandleByIndex(0)
    with open(self.out/'monitor.jsonl','w',buffering=1) as log:
     while not self.stop.is_set():
      row={'time':time.time(),**self.ctx};ps=nv.nvmlDeviceGetComputeRunningProcesses(h);row['compute_processes']=[{'pid':int(p.pid),'usedGpuMemory':int(p.usedGpuMemory)} for p in ps];row['gpu_temperature']=int(nv.nvmlDeviceGetTemperature(h,nv.NVML_TEMPERATURE_GPU));row['gpu_utilization']=int(nv.nvmlDeviceGetUtilizationRates(h).gpu)
      row['own_gpu_used']=next((p['usedGpuMemory'] for p in row['compute_processes'] if p['pid']==row.get('child_pid')),0)
      row['MemAvailable']=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
      if row.get('child_pid'):
       try:row['child_RSS']=next(int(l.split()[1])*1024 for l in Path(f'/proc/{row["child_pid"]}/status').read_text().splitlines() if l.startswith('VmRSS:'))
       except (FileNotFoundError,StopIteration):pass
      self.rows.append(row);log.write(json.dumps(row)+'\n')
      if 2*1024**3<=row['own_gpu_used']<2**60:os.kill(os.getpid(),signal.SIGTERM)
      self.stop.wait(.5)
    nv.nvmlShutdown()
   except Exception as e:self.errors.append(repr(e))
  self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()
 def close(self):
  self.stop.set();self.thread.join(5);write(self.out/'monitor-summary.json',{'samples':len(self.rows),'errors':self.errors,'max_own_gpu_used':max((r['own_gpu_used'] for r in self.rows),default=0),'min_MemAvailable':min((r['MemAvailable'] for r in self.rows),default=None),'accounting':'NVML process used,torch peaks,RSS,MemAvailable separate; cohort contributes background memory','max_gap_s':max((b['time']-a['time'] for a,b in zip(self.rows,self.rows[1:])),default=0)})
def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--start',type=int,default=0);p.add_argument('--stop-after',type=int);a=p.parse_args()
 out=HERE/a.attempt;out.mkdir(parents=True,exist_ok=False);verify();write(out/'gate-before.json',gate());plan=json.loads((HERE/'launch-plan.json').read_text());write(out/'plan.json',plan);write(out/'source-freeze.json',json.loads((HERE/'source-freeze.json').read_text()))
 env=dict(os.environ);env.pop('GOMP_CPU_AFFINITY',None);env.pop('PLAID_SEARCH_PHASE_MANIFEST',None)
 for key,value in plan['gpu_environment'].items():
  if key=='PATH_PREFIX':env['PATH']=value+os.pathsep+env.get('PATH','')
  else:env[key]=value
 ctx={'stage':'before','child_pid':None};mon=Monitor(out,ctx);mon.start();atexit.register(mon.close)
 with open(out/'output.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);receipts=[]
  for i,step in enumerate(plan['commands']):
   if i<a.start:continue
   if a.stop_after is not None and i>a.stop_after:break
   verify();write(out/(step['name']+'-gate.json'),gate());ctx['stage']=step['name'];stepenv=dict(env);stepenv.update(step.get('env_override',{}));expired=[]
   timer=threading.Timer(170,lambda:(expired.append(True),os.kill(os.getpid(),signal.SIGTERM)));timer.daemon=True
   def started(child):ctx['child_pid']=child.pid;write(out/(step['name']+'-started.json'),{'pid':child.pid,'identity':process_identity(child.pid),'argv':step['argv'],'time':time.time()})
   with open(out/(step['name']+'.log'),'w') as log:
    timer.start()
    try:record=supervise_child(step['argv'],env=stepenv,log=log,output_lock=lock,exec_guard=HERE/'exec_guard',receipt_path=out/(step['name']+'-supervision.json'),write_receipt=write,on_started=started,terminate_grace_s=10)
    finally:timer.cancel()
   pid=record['child_pid'];ctx['child_pid']=None;assert record['owned_child_terminal_and_reaped'] and not Path(f'/proc/{pid}').exists()
   record.update(name=step['name'],step=i,timeout=bool(expired));receipts.append(record);write(out/'progress.json',receipts)
   if record['returncode']!=0 or expired:raise RuntimeError(('Step failed',record))
   # No per-mode process may retain GPU payload into the next mode.
   h=nv.nvmlDeviceGetHandleByIndex(0);assert pid not in [int(p.pid) for p in nv.nvmlDeviceGetComputeRunningProcesses(h)]
  write(out/'complete.json',{'complete':True,'steps':receipts,'all_owned_children_terminal_and_reaped':True,'GPU_global_lease_not_claimed':True})
 ctx['stage']='complete';mon.close();atexit.unregister(mon.close)
if __name__=='__main__':main()
