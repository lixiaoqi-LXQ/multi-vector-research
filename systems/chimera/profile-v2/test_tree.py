"""CPU fake-profiler tree/lease tests; no CUDA or NSYS process."""
import ctypes,fcntl,hashlib,json,os,signal,subprocess,sys,time,uuid
from pathlib import Path
P=Path(__file__).resolve().parent;out=P/('cpu-tree-fixtures-'+uuid.uuid4().hex[:8]);out.mkdir(exist_ok=False)
ctypes.CDLL(None).prctl(36,1,0,0,0)
engine=out/'engine.py';engine.write_text("import os,signal,sys,time\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(str(os.getpid()))\nif sys.argv[2]=='short':time.sleep(.1)\nelse:\n signal.signal(signal.SIGTERM,signal.SIG_IGN)\n while True:time.sleep(.02)\n")
profiler=out/'fake-nsys.py';profiler.write_text("import subprocess,sys\np=subprocess.Popen(sys.argv[1:],close_fds=True,start_new_session=True)\nsys.exit(p.wait())\n")
driver=out/'driver.py';driver.write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0,sys.argv[1])\nfrom child_supervision import SupervisedPopen\np=SupervisedPopen([sys.executable,sys.argv[2],sys.argv[3],sys.argv[4]])\nsys.exit(p.wait())\n")
def waitfile(p):
 for _ in range(500):
  if p.exists()and p.stat().st_size:return
  time.sleep(.01)
 raise AssertionError(str(p))
def unlocked(path):
 fd=os.open(path,os.O_RDWR)
 try:
  try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);return True
  except BlockingIOError:return False
 finally:os.close(fd)
checks=[]
for mode in ['normal','timeout','TERM','parent-KILL']:
 d=out/mode;d.mkdir();lock=d/'device.lock';fd=os.open(lock,os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(fd,fcntl.LOCK_EX);nonce=uuid.uuid4().hex
 command=[sys.executable,str(engine),str(d/'ready'),'short'if mode=='normal'else'long']
 cfg=dict(socket_name='chimera-profile-test-'+nonce,token=nonce,lock_path=str(lock),engine_command=command,engine_binary=sys.executable,engine_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),leaf_receipt=str(d/'leaf.json'),tree_receipt=str(d/'tree.json'),timeout_s=1 if mode=='timeout'else 30,cleanup_s=.3)
 cfgpath=d/'config.json';cfg['launch_command']=[sys.executable,str(profiler),sys.executable,str(P/'leaf_exec.py'),str(cfgpath)];cfgpath.write_text(json.dumps(cfg))
 env=dict(os.environ,PLAID_DEVICE_LOCK=str(lock),CUDA_VISIBLE_DEVICES='-1',UMA_RUN_GPU_TESTS='0')
 with open(d/'output.log','x')as log:
  p=subprocess.Popen([sys.executable,str(driver),str(P/'controls'),str(P/'tree_guard.py'),str(cfgpath),str(fd)],env=env,pass_fds=(fd,),stdout=log,stderr=subprocess.STDOUT);os.close(fd)
  waitfile(d/'ready');leaf=int((d/'ready').read_text());assert not unlocked(lock),'Lock not held by actual engine';receipt=json.loads((d/'leaf.json').read_text());assert receipt['pid']==leaf and receipt['lease_inheritable']
  # Engine is in a different session and fake NSYS closed all inherited FDs.
  if mode in ['TERM','parent-KILL']:os.kill(p.pid,signal.SIGTERM if mode=='TERM'else signal.SIGKILL)
  p.wait(timeout=10);waitfile(d/'tree.json');r=json.loads((d/'tree.json').read_text())
  for _ in range(300):
   try:pid,status=os.waitpid(-1,os.WNOHANG)
   except ChildProcessError:pid=0
   if not Path(f'/proc/{leaf}').exists()and unlocked(lock):break
   time.sleep(.01)
  assert not Path(f'/proc/{leaf}').exists()and unlocked(lock)
  assert r['all_owned_descendants_gone']and len(r['lease_transfers'])==1 and not r['lease_server_errors']
  assert mode!='timeout'or r['timed_out']
  checks.append(dict(case=mode,driver_returncode=p.returncode,engine_pid=leaf,NSYS_executed=False,GPU_used=False,escaped_session_cleaned=True,actual_engine_had_held_lease=True,lock_released_after_tree_gone=True,tree_receipt_sha256=hashlib.sha256((d/'tree.json').read_bytes()).hexdigest()))
sys.path.insert(0,str(P/'controls'));from child_supervision import SupervisedPopen
prior=os.environ.get('UMA_RUN_GPU_TESTS');os.environ['UMA_RUN_GPU_TESTS']='1';os.environ['PLAID_DEVICE_LOCK']=str(out/'no-lease.lock')
try:
 try:SupervisedPopen([sys.executable,'-c','pass'])
 except RuntimeError:checks.append(dict(case='GPU-without-inherited-lease-rejected',GPU_used=False))
 else:raise AssertionError('GPU without lease accepted')
finally:
 if prior is None:os.environ.pop('UMA_RUN_GPU_TESTS',None)
 else:os.environ['UMA_RUN_GPU_TESTS']=prior
result=dict(status='PASS_CPU_PROFILE_TREE_AND_LEASE',checks=checks,GPU_used=False,NSYS_executed=False)
(out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
