"""CPU read-only committed-prefix audit after exact native termination."""
from pathlib import Path
import argparse,json,fcntl,hashlib,os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--spec',default=str(P/'spec.json'));p.add_argument('--hash-artifacts',action='store_true');a=p.parse_args();s=json.loads(Path(a.spec).read_text());cmd=s['command'];cache=Path(cmd[cmd.index('--cache')+1]);index=Path(cmd[cmd.index('--index')+1]);locks=[]
for path in (cache/'.lock',Path(str(index)+'.checkpoint-output.lock')):
 f=open(path,'rb');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
def sha(path):
 with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def st(path):
 q=path.stat();return (q.st_dev,q.st_ino,q.st_size,q.st_mtime_ns,q.st_ctime_ns)
state=cache/'state.txt';before=st(state);digest=sha(state);t=state.read_text().split();assert t[0]=='CHIMERA_ASSIGNMENT_CHECKPOINT_V1' and t[1]==s['checkpoint_identity_sha256'];assert list(map(int,t[3:12]))==[597909919,128,4194304,3,65536,42,42,1,150];done=int(t[16]);nchunks=int(t[18]);assert len(t)==19+3*nchunks and 65536<=done<=597909919
artifacts={}
for name,bi,hi in [('centroids.f32',12,13),('assignment.cagra',14,15)]:
 path=cache/name;b=st(path);assert b[2]==int(t[bi]);expected=s['verified_cache_artifacts'][name];assert b[:4]==tuple(expected[k] for k in ('device','inode','size','mtime_ns'))
 if a.hash_artifacts:assert sha(path)==t[hi]
 assert st(path)==b;artifacts[name]=dict(bytes=b[2],sha256=t[hi],full_sha_rechecked=a.hash_artifacts)
chunks=[];position=0;combined=hashlib.sha256();committed=set()
for i in range(nchunks):
 start,count=int(t[19+i*3]),int(t[20+i*3]);h=t[21+i*3];assert start==position and count==min(65536,597909919-start);path=cache/'labels'/f'{start}.u32';before_chunk=st(path);data=path.read_bytes();assert len(data)==count*4 and hashlib.sha256(data).hexdigest()==h;v=np.frombuffer(data,dtype='<u4');assert np.all(v<4194304);assert st(path)==before_chunk;combined.update(data);position+=count;committed.add(path.name);chunks.append(dict(start=start,count=count,sha256=h))
assert position==done and st(state)==before and sha(state)==digest
if done==597909919:assert t[17]==combined.hexdigest()
else:assert t[17]=='-'
orphan=[str(p) for p in (cache/'labels').iterdir() if p.name not in committed];report=dict(status='PASS_READ_ONLY_STOPPED_COMMITTED_PREFIX',completed_tokens=done,total_tokens=597909919,checkpoint_identity_sha256=t[1],state_sha256=digest,checked_committed_chunks=nchunks,artifacts=artifacts,uncommitted_files_preserved=orphan,final_index_exists=index.exists(),both_native_cache_and_output_locks_exclusively_observed_free=True);print(json.dumps(report,indent=2))
