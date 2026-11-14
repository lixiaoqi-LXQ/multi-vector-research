"""CPU-only subprocess/lease proof for future drivers; no CUDA imports."""
import ctypes,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
R=Path(__file__).parent;O=R/'child-supervision-cpu-check';O.mkdir(exist_ok=True)
ctypes.CDLL(None).prctl(36,1,0,0,0) # Test process reaps adopted orphan descendants.
leaf=O/'leaf.py';leaf.write_text("import os,signal,time,sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(str(os.getpid()))\ndef done(*a):\n time.sleep(.5)\n sys.exit(0)\nsignal.signal(signal.SIGTERM,done)\nwhile True:time.sleep(.05)\n")
driver=O/'driver.py';driver.write_text("import sys,os\nfrom pathlib import Path\nsys.path.insert(0,sys.argv[1])\nfrom child_supervision import SupervisedPopen\np=SupervisedPopen([sys.executable,sys.argv[2],sys.argv[3]])\nPath(sys.argv[4]).write_text(str(p.pid))\np.wait()\n")
def waitfile(p):
 for _ in range(200):
  if p.exists() and p.read_text():return int(p.read_text())
  time.sleep(.01)
 raise AssertionError(str(p))
def unlocked(lock):
 fd=os.open(lock,os.O_RDWR)
 try:
  try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);return True
  except BlockingIOError:return False
 finally:os.close(fd)
checks=[]
for label,kill in [('TERM',signal.SIGTERM),('parent-death-KILL',signal.SIGKILL)]:
 lock=O/(label+'.lock');fd=os.open(lock,os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(fd,fcntl.LOCK_EX)
 ready=O/(label+'.leafpid');pidfile=O/(label+'.pid');ready.unlink(missing_ok=True);pidfile.unlink(missing_ok=True)
 env=dict(os.environ,PLAID_DEVICE_LOCK=str(lock),CUDA_VISIBLE_DEVICES='',UMA_RUN_GPU_TESTS='')
 p=subprocess.Popen([sys.executable,str(driver),str(R),str(leaf),str(ready),str(pidfile)],pass_fds=(fd,),env=env)
 os.close(fd);child=waitfile(ready);assert waitfile(pidfile)==child
 assert not unlocked(lock),'Lease missing while child runs'
 os.kill(p.pid,kill);time.sleep(.1);assert not unlocked(lock),'Lease released before child exits'
 p.wait(timeout=10)
 # Under SIGKILL, subreaper test owns the leaf; TERM driver reaps it itself.
 if kill==signal.SIGKILL:
  for _ in range(200):
   got,_=os.waitpid(child,os.WNOHANG)
   if got:break
   time.sleep(.02)
  else:raise AssertionError('Parent-death leaf survived')
 assert not Path(f'/proc/{child}').exists(),'Child unreaped'
 assert unlocked(lock),'Lease retained after child reap'
 checks.append({'case':label,'child_pid':child,'driver_code':p.returncode,'lease_held_until_child_exit':True,'child_absent':True,'lease_released_after_reap':True})
# Exception before wait must still reap at interpreter atexit.
exc=O/'exception-driver.py';exc.write_text(driver.read_text().replace('p.wait()', 'raise RuntimeError("injected failure")'))
lock=O/'exception.lock';fd=os.open(lock,os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(fd,fcntl.LOCK_EX)
ready=O/'exception.leafpid';pidfile=O/'exception.pid';ready.unlink(missing_ok=True);pidfile.unlink(missing_ok=True)
env=dict(os.environ,PLAID_DEVICE_LOCK=str(lock),CUDA_VISIBLE_DEVICES='',UMA_RUN_GPU_TESTS='')
with open(O/'exception.log','w') as log:
 p=subprocess.Popen([sys.executable,str(exc),str(R),str(leaf),str(ready),str(pidfile)],pass_fds=(fd,),env=env,stderr=log)
 os.close(fd);child=waitfile(pidfile);p.wait(timeout=10)
assert p.returncode!=0 and not Path(f'/proc/{child}').exists() and unlocked(lock)
checks.append({'case':'exception-before-wait','child_absent':True,'lease_released_after_reap':True})
(O/'receipt.json').write_text(json.dumps({'cpu_only':True,'checks':checks},indent=2));print(json.dumps(checks))
