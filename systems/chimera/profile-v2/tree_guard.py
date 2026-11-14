"""Subreaper for one owned profiler tree; inherited lease held through descendant cleanup."""
import array,ctypes,json,os,signal,socket,struct,subprocess,sys,threading,time
from pathlib import Path
cfg=json.loads(Path(sys.argv[1]).read_text());lease=int(sys.argv[2]);libc=ctypes.CDLL(None,use_errno=True)
if libc.prctl(36,1,0,0,0):raise OSError(ctypes.get_errno(),'CHILD_SUBREAPER')
requested=None
def stop(signum,frame):
 global requested
 requested=signum
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
def identity(pid):
 try:
  s=Path(f'/proc/{pid}/stat').read_text();r=s[s.rfind(')')+2:].split();return(int(r[1]),int(r[19]))
 except(OSError,ValueError):return None
def descendants():
 rows={}
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   v=identity(int(p.name))
   if v:rows[int(p.name)]=v
 found={os.getpid()};changed=True
 while changed:
  new={pid for pid,(ppid,ticks)in rows.items()if ppid in found}-found;changed=bool(new);found|=new
 return {pid:rows[pid][1]for pid in found if pid!=os.getpid()}
transfers=[];server_errors=[];done=threading.Event();server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.bind('\0'+cfg['socket_name']);server.listen(1);server.settimeout(.1)
def serve():
 while not done.is_set():
  try:conn,_=server.accept()
  except socket.timeout:continue
  except OSError:return
  try:
   conn.settimeout(3);pid,uid,gid=struct.unpack('3i',conn.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12));token=conn.recv(256).decode();assert uid==os.getuid()and token==cfg['token']and pid in descendants(),'Lease request is not this owned tree'
   conn.sendmsg([b'L'],[(socket.SOL_SOCKET,socket.SCM_RIGHTS,array.array('i',[lease]))]);transfers.append(dict(pid=pid,start_ticks=identity(pid)[1],same_open_file_description=True))
  except BaseException as e:server_errors.append(str(e))
  finally:conn.close()
t=threading.Thread(target=serve,daemon=True);t.start();start=time.monotonic();child=None;error=None;timed_out=False;signals=[];reaped=[]
def drain():
 while True:
  try:pid,status=os.waitpid(-1,os.WNOHANG)
  except ChildProcessError:return
  if pid==0:return
  code=os.waitstatus_to_exitcode(status);reaped.append(dict(pid=pid,returncode=code))
  if child is not None and pid==child.pid:child.returncode=code
def signal_tree(sig):
 for pid,ticks in descendants().items():
  try:
   fd=libc.pidfd_open(pid,0)
   if fd<0:
    if ctypes.get_errno()==3:continue
    raise OSError(ctypes.get_errno(),'pidfd_open')
   v=identity(pid)
   if v and v[1]==ticks:
    if libc.pidfd_send_signal(fd,sig,None,0)!=0 and ctypes.get_errno()!=3:raise OSError(ctypes.get_errno(),'pidfd_send_signal')
    signals.append(dict(pid=pid,start_ticks=ticks,signal=sig))
  finally:
   if 'fd'in locals()and fd>=0:os.close(fd);del fd
try:
 child=subprocess.Popen(cfg['launch_command'],env=os.environ.copy(),close_fds=True)
 while child.returncode is None and requested is None:
  drain()
  if time.monotonic()-start>=cfg['timeout_s']:timed_out=True;break
  time.sleep(.02)
except BaseException as e:error=repr(e)
finally:
 # Includes normal NSYS exit with detached helper descendants; never release early.
 signal_tree(signal.SIGTERM);deadline=time.monotonic()+cfg.get('cleanup_s',10)
 while descendants()and time.monotonic()<deadline:drain();time.sleep(.02)
 signal_tree(signal.SIGKILL)
 while descendants():drain();time.sleep(.02)
 drain();done.set();server.close();t.join(timeout=1)
 result=dict(returncode=None if child is None else child.returncode,requested_signal=requested,timed_out=timed_out,error=error,all_owned_descendants_gone=not descendants(),reaped=reaped,signals=signals,lease_transfers=transfers,lease_server_errors=server_errors,subreaper=True,elapsed_s=time.monotonic()-start)
 with open(cfg['tree_receipt'],'x')as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
raise SystemExit(0 if result['returncode']==0 and requested is None and not timed_out and error is None and len(transfers)==1 and not server_errors else 1)
