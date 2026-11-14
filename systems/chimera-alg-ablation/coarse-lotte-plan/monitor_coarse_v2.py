"""Read-only1s offline construction telemetry. No CUDA APIs or process control."""
from pathlib import Path
import json,os,time,subprocess,hashlib,sys,signal
P=Path(__file__).resolve().parent;spec=json.loads((P/'spec.json').read_text());B=Path(spec['base']);S=Path(spec['job_state']);O=B/'telemetry-v2';O.mkdir(exist_ok=False);stop=False
for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda a,b:globals().__setitem__('stop',True))
def proc(pid,expected_start=None):
 result={'pid':pid}
 try:
  raw=Path(f'/proc/{pid}/stat').read_text();v=raw[raw.rfind(')')+2:].split();result.update(state=v[0],ppid=int(v[1]),start_ticks=int(v[19]),minor_faults=int(v[7]),major_faults=int(v[9]));result['identity_start_match']=expected_start is None or int(v[19])==expected_start
 except (FileNotFoundError,ProcessLookupError):result['terminal_or_gone']=True;return result
 except PermissionError as e:result['proc_access_error']=repr(e);return result
 try:
  for line in Path(f'/proc/{pid}/status').read_text().splitlines():
   key,_,value=line.partition(':')
   if key in ('VmRSS','VmHWM','VmSize','RssAnon','RssFile','VmSwap'):result[key+'_kB']=int(value.split()[0])
   elif key=='Cpus_allowed_list':result[key]=value.strip()
 except (FileNotFoundError,ProcessLookupError,PermissionError) as e:result['status_access_error']=repr(e)
 try:result['io']={k:int(value) for line in Path(f'/proc/{pid}/io').read_text().splitlines() for k,value in [line.split(':',1)]}
 except (FileNotFoundError,ProcessLookupError,PermissionError) as e:result['io_access_error']=repr(e)
 try:result['exe']=str(Path(f'/proc/{pid}/exe').resolve())
 except (FileNotFoundError,ProcessLookupError,PermissionError) as e:result['exe_access_error']=repr(e)
 group={}
 try:
  for q in Path(f'/proc/{pid}/task').iterdir():
   try:group[q.name]=next(x.split(':',1)[1].strip() for x in (q/'status').read_text().splitlines() if x.startswith('Cpus_allowed_list:'))
   except (FileNotFoundError,ProcessLookupError,PermissionError,StopIteration):pass
 except (FileNotFoundError,ProcessLookupError,PermissionError) as e:result['threads_access_error']=repr(e)
 result['thread_allowed_cpus']=group
 return result
def command(cmd):
 try:
  q=subprocess.run(cmd,text=True,capture_output=True,timeout=2);return {'returncode':q.returncode,'stdout':q.stdout.strip(),'stderr':q.stderr.strip()}
 except subprocess.TimeoutExpired:return {'timeout_s':2}
started=time.time();old=None;offset=0;pending='';native_start=None;count=0;gaps=[];min_available=None;max_rss=0;max_hwm=0;max_anon=0;status='monitor_stopped';last=None
(O/'launch.json').write_text(json.dumps({'pid':os.getpid(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'start_unix':started,'target_period_s':1,'CPU_only_read_observer':True,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'GT_identity':spec['existing_GT_identity'],'notes':['CPU process fault counters; not CUDA page faults','MemAvailable/RSS/GPU memory are separate views and must not be summed','Event time is collection time, host ranges are not GPU-exclusive timing']},indent=2))
with open(O/'samples.jsonl','w') as out,open(O/'checkpoint-events.jsonl','w') as events:
 while not stop:
  mono=time.monotonic();now=time.time();job=json.loads(S.read_text());nfile=Path(job['receipt_dir'])/'process.json';native=None
  if nfile.exists():
   n=json.loads(nfile.read_text());native=proc(n['pid'],native_start)
   if native_start is None and 'start_ticks' in native:native_start=native['start_ticks']
  wrapper=job.get('wrapper');wp=proc(wrapper['pid'],wrapper['start_ticks']) if wrapper else None;gtid=spec['existing_GT_identity'];gt=proc(gtid['pid'],gtid['start_ticks'])
  try:gt['cmdline_sha_matches']=hashlib.sha256(Path(f"/proc/{gtid['pid']}/cmdline").read_bytes()).hexdigest()==gtid['cmdline_sha256']
  except (FileNotFoundError,ProcessLookupError,PermissionError):gt['cmdline_sha_matches']=False
  extra_cohort=[];cohort_path=B/'additional-cohort.json'
  if cohort_path.exists():
   for identity in json.loads(cohort_path.read_text()).get('processes',[]):
    entry=proc(identity['pid'],identity['start_ticks'])
    try:entry['cmdline_sha_matches']=hashlib.sha256(Path(f"/proc/{identity['pid']}/cmdline").read_bytes()).hexdigest()==identity['cmdline_sha256']
    except (FileNotFoundError,ProcessLookupError,PermissionError):entry['cmdline_sha_matches']=False
    entry['role']=identity.get('role');extra_cohort.append(entry)
  mem={k:int(v.split()[0]) for line in Path('/proc/meminfo').read_text().splitlines() for k,v in [line.split(':',1)] if k in ('MemTotal','MemAvailable','MemFree','Buffers','Cached','SwapTotal','SwapFree')};min_available=min(mem['MemAvailable'],min_available or mem['MemAvailable'])
  if native:max_rss=max(max_rss,native.get('VmRSS_kB',0));max_hwm=max(max_hwm,native.get('VmHWM_kB',0));max_anon=max(max_anon,native.get('RssAnon_kB',0))
  gap=None if old is None else mono-old
  if gap is not None and gap>1.5:gaps.append({'sample':count,'gap_s':gap,'time':now})
  old=mono;gpu=command(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,clocks.mem,temperature.gpu','--format=csv,noheader']);apps=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']);locks=command(['lslocks','-n','-o','PID,MODE,PATH']);locks['stdout']='\n'.join(line for line in locks.get('stdout','').splitlines() if '/tmp/uma-paper-device0.lock' in line)
  sample={'unix_time':now,'monotonic':mono,'sample_start_gap_s':gap,'collection_elapsed_s':time.monotonic()-mono,'job_status':job.get('status'),'native':native,'wrapper':wp,'GT':gt,'additional_cohort':extra_cohort,'system_memory_kB':mem,'gpu_view':gpu,'compute_apps':apps,'device_lock_view':locks,'memory_note':'Separate memory views; do not sum'};out.write(json.dumps(sample)+'\n');out.flush();last=sample;count+=1
  logfile=Path(job['receipt_dir'])/'build.log'
  if logfile.exists():
   with logfile.open() as f:f.seek(offset);text=f.read();new_offset=f.tell()
   data=pending+text;lines=data.split('\n');pending=lines.pop()
   for line in lines:
    if '[BUILD_CHECKPOINT]' in line or '[BUILD_PHASE]' in line:events.write(json.dumps({'collected_unix_time':now,'source_log_offset_end':new_offset,'initial_existing_log_read':offset==0,'line':line})+'\n');events.flush()
   offset=new_offset
  native_alive=native and native.get('state') not in (None,'Z') and native.get('identity_start_match',False);wrapper_alive=wp and wp.get('state') not in (None,'Z') and wp.get('identity_start_match',False)
  if job.get('status') in ('complete','failed','interrupted','checkpoint_stopped') and not native_alive and not wrapper_alive:status='job_and_owned_processes_terminal';break
  time.sleep(max(0,1-(time.monotonic()-mono)))
(O/'termination.json').write_text(json.dumps({'pid':os.getpid(),'status':status,'stop_signal_requested':stop,'samples':count,'elapsed_s':time.time()-started,'gaps_over_1p5s':gaps,'min_MemAvailable_kB':min_available,'max_native_VmRSS_kB':max_rss,'max_native_VmHWM_kB':max_hwm,'max_native_RssAnon_kB':max_anon,'last_sample':last,'collected_end_unix':time.time()},indent=2))
