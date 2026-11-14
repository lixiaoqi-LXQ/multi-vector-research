"""One-second NVML telemetry; unsupported UMA fields are recorded once and omitted."""
import json,os,threading,time
import pynvml as nv
from process_inventory import classify_compute_pids
def memory_snapshot(pid=None):
    values={}
    for line in open('/proc/meminfo'):
        key,value=line.split(':',1)
        if key in ['MemTotal','MemAvailable','MemFree','Cached','SReclaimable','Shmem']:values[key]=int(value.split()[0])*1024
    result={'time':time.time(),'system_memory_bytes':values}
    if pid is not None:
        try:
            for line in open(f'/proc/{pid}/status'):
                if line.startswith(('VmRSS:','VmHWM:')):result[line.split(':')[0]+'_bytes']=int(line.split()[1])*1024
        except FileNotFoundError:pass
    return result

class Telemetry:
    def __init__(self,path,context):
        self.path=path;self.context=context;self.stop_event=threading.Event();self.unsupported={};self.errors=[];self.foreign=set();self.thread=None;self.sample_count=0;self.max_gap_s=0;self.last_sample=None;self.memory_by_stage={}
    def start(self):
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start();return self
    def _run(self):
        try:
            os.sched_setaffinity(threading.get_native_id(),{0})
            nv.nvmlInit();h=nv.nvmlDeviceGetHandleByIndex(0)
            def field(name,fn):
                if name in self.unsupported:return None
                try:return fn()
                except nv.NVMLError as e:
                    self.unsupported[name]=str(e);return None
            with open(self.path,'w',buffering=1) as f:
                while not self.stop_event.is_set():
                    begin=time.monotonic();ctx=dict(self.context);row={'time':time.time(),**ctx}
                    if ctx.get('detail_path'):
                        try:row['grid_detail']=json.loads(open(ctx['detail_path']).read())
                        except (FileNotFoundError,json.JSONDecodeError):pass
                    meminfo={}
                    for line in open('/proc/meminfo'):
                        key,value=line.split(':',1)
                        if key in ['MemTotal','MemAvailable','MemFree','Cached','SReclaimable','Shmem']:meminfo[key]=int(value.split()[0])*1024
                    row['system_memory_bytes']=meminfo
                    rss=None
                    if ctx.get('child_pid'):
                        try:
                            for line in open(f"/proc/{ctx['child_pid']}/status"):
                                if line.startswith('VmRSS:'):rss=int(line.split()[1])*1024;break
                        except FileNotFoundError:pass
                    if rss is not None:row['child_rss_bytes']=rss
                    phase=row.get('grid_detail',{}).get('stage',ctx.get('stage','unknown'))
                    aggregate=self.memory_by_stage.setdefault(phase,{'min_MemAvailable':meminfo['MemAvailable'],'min_MemFree':meminfo['MemFree'],'max_child_RSS':0,'samples':0})
                    aggregate['min_MemAvailable']=min(aggregate['min_MemAvailable'],meminfo['MemAvailable']);aggregate['min_MemFree']=min(aggregate['min_MemFree'],meminfo['MemFree']);aggregate['max_child_RSS']=max(aggregate['max_child_RSS'],rss or 0);aggregate['samples']+=1
                    funcs={
                        'temperature_c':lambda:int(nv.nvmlDeviceGetTemperature(h,nv.NVML_TEMPERATURE_GPU)),
                        'sm_clock_mhz':lambda:int(nv.nvmlDeviceGetClockInfo(h,nv.NVML_CLOCK_SM)),
                        'mem_clock_mhz':lambda:int(nv.nvmlDeviceGetClockInfo(h,nv.NVML_CLOCK_MEM)),
                        'power_mw':lambda:int(nv.nvmlDeviceGetPowerUsage(h)),
                        'pstate':lambda:int(nv.nvmlDeviceGetPerformanceState(h)),
                        'utilization':lambda:{'gpu':int((r:=nv.nvmlDeviceGetUtilizationRates(h)).gpu),'memory':int(r.memory)},
                        'compute_pids':lambda:[int(p.pid) for p in nv.nvmlDeviceGetComputeRunningProcesses(h)],
                        'memory_bytes':lambda:{'used':int((m:=nv.nvmlDeviceGetMemoryInfo(h)).used),'total':int(m.total),'free':int(m.free)},
                    }
                    for name,fn in funcs.items():
                        value=field(name,fn)
                        if value is not None:row[name]=value
                    inventory=classify_compute_pids(row.get('compute_pids',[]))
                    row.update(inventory);self.foreign.update(inventory['unattributed_compute_pids'])
                    f.write(json.dumps(row)+'\n');self.sample_count+=1
                    now=time.monotonic()
                    if self.last_sample is not None:self.max_gap_s=max(self.max_gap_s,now-self.last_sample)
                    self.last_sample=now
                    self.stop_event.wait(max(0,1-(time.monotonic()-begin)))
            nv.nvmlShutdown()
        except Exception as e:self.errors.append(repr(e))
    def close(self):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=5)
        self.path.with_suffix('.status.json').write_text(json.dumps({'interval_seconds':1,'sample_count':self.sample_count,'maximum_gap_seconds':self.max_gap_s,'monitor_cpu_affinity':[0],'system_memory_by_stage_bytes':self.memory_by_stage,'memory_accounting':'System MemAvailable/MemFree and process RSS are separate observations; never add RSS and CUDA allocation counters as physical consumption.','unsupported_fields':self.unsupported,'errors':self.errors,'unattributed_compute_pids_observed':sorted(self.foreign),'inventory_interpretation':'raw compute_pids records residency. Own live descendant PID+starttime attribution includes nested drivers; unattributed inventory alone is not foreign active computation.'},indent=2))
