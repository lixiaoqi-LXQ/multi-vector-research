"""Linux exec shim: parent-death TERM and race check without unsafe preexec_fn."""
import ctypes,os,signal,sys
expected=int(sys.argv[1]);libc=ctypes.CDLL(None,use_errno=True)
if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG')
if os.getppid()!=expected:os.kill(os.getpid(),signal.SIGTERM)
os.execvpe(sys.argv[2],sys.argv[2:],os.environ)
