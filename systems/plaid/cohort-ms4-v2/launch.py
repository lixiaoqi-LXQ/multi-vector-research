"""Prepared only by default. --run requires root's explicit later launch authorization."""
import argparse,atexit,fcntl,hashlib,json,os,signal,sys,threading,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
from policy import verify,holder,matches,parse_ms_stage,resource_snapshot,stop_reason,source_stat
from supervision import supervise_child

def write(p,x):p.write_text(json.dumps(x,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--preflight',action='store_true');a=p.parse_args();spec=json.loads((HERE/'spec.json').read_text());freeze=json.loads((HERE/'freeze.json').read_text());assert hashlib.sha256((HERE/'spec.json').read_bytes()).hexdigest()==freeze['spec_sha256'];check=verify(spec)
 if not a.run:
  print(json.dumps(check,indent=2));return
 assert os.environ.get('PLAID_MS4_COHORT_ROOT_AUTHORIZED')=='1','Root must authorize this frozen fullMS4 launch; spec preparation is not launch permission'
 out=Path(spec['run_output']);out.mkdir(parents=True,exist_ok=False);write(out/'preflight.json',check);write(out/'spec.json',spec);env=dict(os.environ);env.update(spec['environment']);env['PATH']=spec['PATH_prefix']+os.pathsep+env.get('PATH','');env.pop('GOMP_CPU_AFFINITY',None);env.pop('PLAID_SEARCH_PHASE_MANIFEST',None)
 stop=threading.Event();records=[];reasons=[];child_pid=[None];monitor_errors=[];offset=[0];ms_identity=[None];last_stage=['unconfirmed'];pending_log=['']
 def request(reason):
  if not reasons:
   reasons.append(reason);write(out/'stop-reason.json',{'reason':reason,'time':time.time(),'action':'TERM only exact owned PLAID child; 30s grace then KILL/reap; preserve partial files; no automatic restart'})
   os.kill(os.getpid(),signal.SIGTERM)
 def monitor():
  try:
   os.sched_setaffinity(threading.get_native_id(),{0});import pynvml as nv;nv.nvmlInit();handle=nv.nvmlDeviceGetHandleByIndex(0);soft=0
   with open(out/'monitor.jsonl','w',buffering=1) as log:
    while not stop.is_set():
     snap=resource_snapshot(spec);available=snap['memory_bytes']['MemAvailable'];soft=soft+1 if available<spec['runtime_soft_MemAvailable_bytes'] else 0
     try:h=holder(spec);holder_ok=True
     except Exception as e:h={'rejected':str(e)};holder_ok=False
     # Exactly identify the MS native and consume every appended checkpoint marker.
     if not matches(spec['MS_native']):stage='MS_native_missing_or_changed'
     else:
      path=Path(spec['MS_assignment_log']);s=path.stat();identity=(s.st_dev,s.st_ino)
      if ms_identity[0] is not None and (identity!=ms_identity[0] or s.st_size<offset[0]):stage='MS_log_replaced_or_truncated'
      else:
       ms_identity[0]=identity
       with path.open() as f:f.seek(offset[0]);delta=f.read();offset[0]=f.tell()
       combined=pending_log[0]+delta
       lines=combined.split('\n');pending_log[0]=lines.pop();observed=parse_ms_stage('\n'.join(lines))
       if observed!='unconfirmed':last_stage[0]=observed
       stage=last_stage[0]
     pid=child_pid[0];anon=0;rss={}
     if pid:
      try:
       rss={l.split(':')[0]:int(l.split()[1])*1024 for l in Path(f'/proc/{pid}/status').read_text().splitlines() if l.startswith(('RssAnon:','RssFile:','VmRSS:','VmHWM:'))};anon=rss.get('RssAnon',0)
      except FileNotFoundError:pass
     raw=[{'pid':int(p.pid),'usedGpuMemory':int(p.usedGpuMemory)} for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)]
     row={'time':time.time(),**snap,'MS_stage':stage,'holder':h,'own_pid':pid,'own_RSS_separate_bytes':rss,'raw_NVML_compute_inventory':raw,'own_NVML_process_used':next((r['usedGpuMemory'] for r in raw if r['pid']==pid),0),'soft_low_count':soft};records.append(row);log.write(json.dumps(row)+'\n')
     reason=stop_reason(spec,available,soft,stage,holder_ok,snap['disk_free_bytes'],source_stat(spec['source_file'])==spec['source_stat'],anon,row['own_NVML_process_used'])
     if reason:request(reason);break
     stop.wait(.5)
   nv.nvmlShutdown()
  except BaseException as e:
   monitor_errors.append(repr(e))
   if not stop.is_set():request('safety_monitor_error:'+repr(e))
 with open(out/'output.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  thread=[None];timer=[None]
  def started(child):
   child_pid[0]=child.pid;write(out/'child-started.json',{'pid':child.pid,'argv':spec['command'],'time':time.time()});thread[0]=threading.Thread(target=monitor,daemon=True);thread[0].start();timer[0]=threading.Timer(spec['maximum_wall_seconds'],lambda:request('maximum_wall_time'));timer[0].daemon=True;timer[0].start()
  try:
   with open(out/'build.log','w') as log:receipt=supervise_child(spec['command'],env=env,log=log,output_lock=lock,exec_guard=HERE/'exec_guard',receipt_path=out/'supervision.json',write_receipt=write,on_started=started,terminate_grace_s=30)
  finally:
   stop.set()
   if timer[0]:timer[0].cancel()
   if thread[0]:thread[0].join(5)
  assert receipt['owned_child_terminal_and_reaped'] and not Path(f'/proc/{receipt["child_pid"]}').exists()
  finished={'returncode':receipt['returncode'],'terminal_and_reaped':True,'stop_reasons':reasons,'monitor_errors':monitor_errors,'partial_preserved':True,'formal_performance_time':False,'samples':len(records),'min_MemAvailable':min((r['memory_bytes']['MemAvailable'] for r in records),default=None),'max_own_RssAnon':max((r['own_RSS_separate_bytes'].get('RssAnon',0) for r in records),default=0),'max_own_NVML_process_used':max((r['own_NVML_process_used'] for r in records),default=0)}
  if receipt['returncode']==0 and not reasons:
   meta=json.loads((Path(spec['index'])/'metadata.json').read_text());inp=json.loads((Path(spec['index'])/'input-identity.json').read_text());assert meta['num_documents']==8841823 and meta['num_embeddings']==597909919 and meta['num_partitions']==262144 and meta['config']['nbits']==4
   assert inp['scanned_prefix_sha256']==spec['GT_scanned_payload_sha256'];assert source_stat(spec['source_file'])==spec['source_stat'];finished['complete']=True
  else:finished['complete']=False
  write(out/'completion.json',finished)
 if not finished['complete']:raise SystemExit(1)
if __name__=='__main__':main()
