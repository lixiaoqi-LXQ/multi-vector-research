"""Receive the held device lease across profiler FD sanitization, then exec GPU leaf."""
import array,ctypes,fcntl,hashlib,json,os,signal,socket,sys
from pathlib import Path
config=json.loads(Path(sys.argv[1]).read_text());parent=os.getppid()
libc=ctypes.CDLL(None,use_errno=True)
if libc.prctl(1,signal.SIGKILL,0,0,0):raise OSError(ctypes.get_errno(),'PDEATHSIG')
if os.getppid()!=parent:raise RuntimeError('Leaf parent changed before exec')
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);s.connect('\0'+config['socket_name']);s.sendall(config['token'].encode())
message,ancillary,flags,address=s.recvmsg(1,socket.CMSG_SPACE(array.array('i').itemsize));s.close();fds=array.array('i')
for level,kind,data in ancillary:
 if level==socket.SOL_SOCKET and kind==socket.SCM_RIGHTS:fds.frombytes(data[:len(data)-len(data)%fds.itemsize])
assert message==b'L' and len(fds)==1,'Missing single held lease descriptor'
fd=fds[0];actual=os.fstat(fd);expected=Path(config['lock_path']).stat();assert(actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino),'Wrong lease inode'
fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);os.set_inheritable(fd,True)
binary=Path(config['engine_binary']);assert hashlib.sha256(binary.read_bytes()).hexdigest()==config['engine_sha256'],'Engine changed before exec'
record=dict(pid=os.getpid(),parent_pid=parent,start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),lease_fd=fd,lease_device=actual.st_dev,lease_inode=actual.st_ino,lease_inheritable=os.get_inheritable(fd),lease_transfer='SCM_RIGHTS same open file description',command=config['engine_command'],engine_sha256=config['engine_sha256'])
with open(config['leaf_receipt'],'x')as f:json.dump(record,f,indent=2);f.flush();os.fsync(f.fileno())
os.execvpe(config['engine_command'][0],config['engine_command'],os.environ)
