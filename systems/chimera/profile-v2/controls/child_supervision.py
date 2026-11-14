"""Exact owned-child termination/reaping; preserve inherited flock fd through child tree."""
import atexit,os,signal,subprocess,sys,threading,time
from pathlib import Path
_ACTIVE=set()
class SupervisorTermination(SystemExit):pass
def _terminate(signum,frame):raise SupervisorTermination(128+signum)
def lease_fds():
    path=Path(os.environ.get('PLAID_DEVICE_LOCK','/tmp/uma-paper-device0.lock'))
    if not path.exists():return ()
    st=path.stat();found=[]
    for item in Path('/proc/self/fd').iterdir():
        try:
            fd=int(item.name);fs=os.fstat(fd)
            if fd>2 and (fs.st_dev,fs.st_ino)==(st.st_dev,st.st_ino):found.append(fd)
        except (OSError,ValueError):pass
    return tuple(found)
class SupervisedPopen(subprocess.Popen):
    def __init__(self,args,**kwargs):
        assert not kwargs.get('shell'), 'No shell child supervision'
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM,_terminate);signal.signal(signal.SIGHUP,_terminate)
        inherited_lease=lease_fds()
        if (kwargs.get('env') or os.environ).get('UMA_RUN_GPU_TESTS')=='1' and not inherited_lease:
            raise RuntimeError('GPU child requires inherited shared device flock FD; invoke under flock without --close')
        passes=set(kwargs.pop('pass_fds',()))|set(inherited_lease)
        kwargs['pass_fds']=tuple(sorted(passes));kwargs['start_new_session']=True
        self.lease_fds=tuple(sorted(passes))
        cmd=[sys.executable,str(Path(__file__).with_name('child_entry.py')),str(os.getpid()),*map(str,args)]
        super().__init__(cmd,**kwargs);_ACTIVE.add(self)
    def terminate_and_reap(self,grace=10):
        if self.poll() is None:
            try:os.killpg(self.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:subprocess.Popen.wait(self,timeout=grace)
            except subprocess.TimeoutExpired:
                try:os.killpg(self.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                subprocess.Popen.wait(self)
        else:subprocess.Popen.wait(self)
        _ACTIVE.discard(self)
    def wait(self,timeout=None):
        try:
            code=super().wait(timeout=timeout);_ACTIVE.discard(self);return code
        except subprocess.TimeoutExpired:raise
        except BaseException:
            # Ignore a second TERM during exact-child cleanup, then restore handler.
            old=signal.signal(signal.SIGTERM,signal.SIG_IGN)
            try:self.terminate_and_reap()
            finally:signal.signal(signal.SIGTERM,old)
            raise
    def __exit__(self,*exc):
        if exc[0]:self.terminate_and_reap()
        return super().__exit__(*exc)
@atexit.register
def _cleanup():
    for child in list(_ACTIVE):child.terminate_and_reap()
def run(args,*,check=False,**kwargs):
    proc=SupervisedPopen(args,**kwargs)
    try:code=proc.wait()
    finally:
        if proc.poll() is None:proc.terminate_and_reap()
    if check and code:raise subprocess.CalledProcessError(code,args)
    return subprocess.CompletedProcess(args,code)
