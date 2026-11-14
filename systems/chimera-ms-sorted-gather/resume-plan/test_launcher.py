"""Actual private launch/guardian code; fake CPU wrapper, private locks only."""
from pathlib import Path
import ctypes,hashlib,json,os,signal,subprocess,sys,time,shutil
import preflight
P=Path(__file__).resolve().parent;T=P/'cpu-launcher-tests';T.mkdir(exist_ok=False)
# Reap only descendants of this isolated test supervisor.
assert ctypes.CDLL(None).prctl(36,1,0,0,0)==0
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def waitfor(test,timeout=8):
 start=time.monotonic()
 while time.monotonic()-start<timeout:
  try:
   value=test()
   if value:return value
  except (FileNotFoundError,json.JSONDecodeError):pass
  time.sleep(.02)
 raise TimeoutError('fixture condition timed out')
def reap(pid):
 try:os.waitpid(pid,0)
 except ChildProcessError:pass
records=[]
for name,valid,killguard,rc in [('valid_cohort_Q1_stop3',True,False,3),('invalid_cohort_waits_for_own_lease',False,False,0),('guardian_dead_no_wrapper',False,True,0)]:
 d=T/name;d.mkdir();base=d/'run';lock=d/'device.lock';ready=d/'holder.ready'
 for n in ['launch_full.py','guard_full.py']:shutil.copy2(P/n,d/n)
 # Actual cohort identity logic; resource/input tests are separately exercised by test_preflight.
 (d/'preflight.py').write_text((P/'preflight.py').read_text()+"\n# CPU-fixture-only injected resource/input validators:\nverify=lambda spec: None\nverify_resources=lambda spec: None\n")
 c=d/'controls';c.mkdir();shutil.copy2('/home/lxq/graduate/ai-workspace/research/multivector/chimera-ms-sorted-gather/q1-plan/controls/exec_guard',c/'exec_guard');(c/'gpu_build').write_text('never executed CPU fake binary identity')
 fake=d/'fake_wrapper.py';fake.write_text("import sys,time,json\nfrom pathlib import Path\na=sys.argv;receipt=Path(a[a.index('--receipt-dir')+1]);receipt.mkdir();(receipt/'started').write_text('CPU fixture only');time.sleep(.12);raise SystemExit(int(a[a.index('--fake-rc')+1]))\n")
 holder=subprocess.Popen([sys.executable,'-c',"import fcntl,time,sys;from pathlib import Path;f=open(sys.argv[1],'a+');fcntl.flock(f,fcntl.LOCK_EX);Path(sys.argv[2]).write_text('ready');time.sleep(30)",str(lock),str(ready)])
 lotte=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);gt=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);waitfor(ready.exists)
 identities=[{k:preflight.process(p.pid)[k] for k in ('pid','start_ticks','cmdline_sha256')} for p in [holder,lotte,gt]]
 cohort=json.loads(json.dumps(identities))
 if not valid:cohort[1]['start_ticks']+=1
 provenance=d/'provenance.json';provenance.write_text('{}');spec={'base':str(base),'job_state':str(base/'job-state.json'),'command':[sys.executable,str(fake),'--binary',str(c/'gpu_build'),'--index',str(d/'index'),'--receipt-dir',str(d/'receipt'),'--provenance-json',str(provenance),'--fake-rc',str(rc)],'environment':{'CUDA_VISIBLE_DEVICES':'-1','UMA_RUN_GPU_TESTS':'0'},'predecessor_holder_pid':holder.pid,'predecessor_native_identity':identities[1],'authorized_offline_cohort':cohort,'guardian_script':str(d/'guard_full.py'),'lease_path':str(lock),'control_file_sha256':{str(c/n):sha(c/n) for n in ['exec_guard','gpu_build']},'launch_file_sha256':{str(d/n):sha(d/n) for n in ['launch_full.py','guard_full.py','preflight.py']},'provenance_sha256':sha(provenance)};(d/'spec.json').write_text(json.dumps(spec))
 log=open(d/'launcher.log','w');proc=subprocess.Popen([sys.executable,str(d/'launch_full.py'),'--run'],stdout=log,stderr=subprocess.STDOUT)
 gpid=None
 try:
  job=waitfor(lambda:json.loads((base/'job-state.json').read_text()).get('continuation_guardian_identity'));gpid=job['pid'];waitfor(lambda:json.loads((base/'guardian-state.json').read_text())['phase']=='queued')
  if valid:
   code=proc.wait(timeout=8);state=json.loads((base/'job-state.json').read_text());assert code==3 and state['status']=='checkpoint_stopped' and state['intentional_checkpoint_stop'] is True and (d/'receipt/started').exists()
   assert json.loads((base/'guardian-state.json').read_text())['phase']=='queued'
   owners=subprocess.check_output(['lslocks','-n','-o','PID,MODE,PATH'],text=True);assert any(row.split()[-1]==str(lock) and int(row.split()[0])==holder.pid and '*' not in row.split()[1] for row in owners.splitlines());assert preflight.process(gpid)['start_ticks']==job['start_ticks']
   os.kill(gpid,signal.SIGTERM);reap(gpid);guardian_cancelled_queued=True
  else:
   time.sleep(.2);assert json.loads((base/'job-state.json').read_text())['wrapper'] is None and not (d/'receipt/started').exists()
   if killguard:
    os.kill(gpid,signal.SIGTERM);code=proc.wait(timeout=8);assert code!=0 and not (d/'receipt/started').exists();reap(gpid);guardian_cancelled_queued=True
   else:
    holder.terminate();holder.wait();code=proc.wait(timeout=8);assert code==0 and json.loads((base/'job-state.json').read_text())['launch_lease']=='successor guardian actually holding';waitfor(lambda:json.loads((base/'guardian-state.json').read_text())['phase']=='released');reap(gpid);guardian_cancelled_queued=False
  records.append({'case':name,'passed':True,'launcher_returncode':code,'guardian_precisely_cancelled_while_queued':guardian_cancelled_queued,'GPU_used':False})
 finally:
  if proc.poll() is None:proc.terminate();proc.wait(timeout=5)
  log.close()
  for p in [holder,lotte,gt]:
   if p.poll() is None:p.terminate()
   p.wait()
  if gpid and preflight.process(gpid) is not None:
   try:os.kill(gpid,signal.SIGTERM)
   except ProcessLookupError:pass
   reap(gpid)
(P/'cpu-launcher-validation.json').write_text(json.dumps({'CPU_only':True,'GPU_used':False,'tests':records,'production_launch_sha256':sha(P/'launch_full.py'),'production_guard_sha256':sha(P/'guard_full.py'),'resource_input_checks_separately_tested':True},indent=2)+'\n');print(json.dumps(records,indent=2))
