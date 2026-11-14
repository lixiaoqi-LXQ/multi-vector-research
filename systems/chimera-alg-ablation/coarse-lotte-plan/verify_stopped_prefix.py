"""Read-only verification of a deliberately stopped coarse checkpoint, including every committed label."""
import argparse,fcntl,hashlib,json,os,struct,subprocess,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
def need(v,m):
 if not v:raise ValueError(m)
def sha(p):
 with open(p,'rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stat(p):
 s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
 if not a.run:print(json.dumps({'prepared_only':True,'GPU_used':False,'cache_read':False}));return
 begin=time.time();spec=json.loads((P/'spec.json').read_text());cmd=spec['command'];cache=Path(cmd[cmd.index('--cache')+1]);index=Path(cmd[cmd.index('--index')+1]);run=Path(spec['job_state']).parent;receipt=Path(cmd[cmd.index('--receipt-dir')+1]);job=json.loads((run/'job-state.json').read_text());supervision=json.loads((receipt/'supervision.json').read_text());result=json.loads((receipt/'result.json').read_text());need(job['status']=='interrupted'and job['returncode']==143,'Not the requested stopped coarse attempt');need(supervision['owned_child_terminal_and_reaped']and supervision['returncode']==-15 and supervision['requested_signal']==15,'Native TERM/reap evidence missing');need(result['status']=='interrupted'and result['complete']is False,'Stopped partial must remain incomplete')
 actors={str(pid):not Path(f'/proc/{pid}').exists()for pid in [3474837,3474859,3482814,3474856,3542509]};need(all(actors.values()),'Owned coarse actors or observer still exist')
 locks=[]
 for path in [cache/'.lock',Path(str(index)+'.checkpoint-output.lock')]:
  with open(path,'rb')as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(dict(path=str(path),exclusive_nonblocking_acquired_then_released=True))
 need(not index.exists(),'Stopped assignment unexpectedly has a final index');statepath=cache/'state.txt';before=stat(statepath);state=statepath.read_text().splitlines();need(state[0]=='CHIMERA_ASSIGNMENT_CHECKPOINT_V1','Bad cache schema');need(list(map(int,state[3].split()))==[266205513,128,131072,3,65536,42,42,1,150],'Cache settings changed')
 identity_path=Path(str(cache)+'.identity.json');identity=json.loads(identity_path.read_text());canonical=json.dumps(identity['bindings'],sort_keys=True,separators=(',',':')).encode();need(hashlib.sha256(canonical).hexdigest()==identity['identity_sha256']==state[1]==result['identity_sha256'],'Immutable cache/source/binary/environment binding mismatch');need(identity['identity_sha256']==json.loads((receipt/'prepared.json').read_text())['identity_sha256'],'Prepared attempt identity mismatch')
 lenspath=Path(cmd[cmd.index('--doclens')+1]);lens=np.fromfile(lenspath,dtype='<i4');need(lens[0]==2428854 and len(lens)==2428855 and np.all(lens[1:]>0)and int(lens[1:].sum(dtype=np.int64))==266205513,'Input doclens shape changed');need(hashlib.sha256(lens[1:].tobytes()).hexdigest()==state[2],'Cache doclens payload SHA changed')
 payloads={}
 for name,line in [('centroids.f32',4),('assignment.cagra',5)]:
  size,h=state[line].split();q=cache/name;first=stat(q);need(first['size']==int(size)and sha(q)==h and stat(q)==first,'Cache matrix/assignment graph size or fullSHA mismatch');payloads[name]=dict(first,sha256=h)
 completed,digest,batches=state[6].split();completed,batches=int(completed),int(batches);need(0<completed<266205513 and len(state)==7+batches,'Not a well formed partial prefix');need(digest=='-','Unexpected partial cache digest marker');cursor=0;full=hashlib.sha256();lo=131072;hi=-1;committed=[]
 for line in state[7:]:
  start,count,h=line.split();start,count=int(start),int(count);need(start==cursor and count==min(65536,266205513-start),'Committed prefix gap/count mismatch');q=cache/'labels'/f'{start}.u32';first=stat(q);payload=q.read_bytes();need(len(payload)==count*4 and hashlib.sha256(payload).hexdigest()==h and stat(q)==first,'Committed label chunk size/SHA changed');values=np.frombuffer(payload,dtype='<u4');need(np.all(values<131072),'Committed centroid ID outside C131072');lo=min(lo,int(values.min()));hi=max(hi,int(values.max()));cursor+=count;full.update(payload);committed.append(dict(start=start,count=count,sha256=h,stat=first))
 need(cursor==completed,'Prefix count disagrees with state');need(stat(statepath)==before,'Checkpoint state changed during verification')
 listed={f'{x["start"]}.u32'for x in committed};extras=[]
 for q in (cache/'labels').iterdir():
  if q.name not in listed:extras.append(dict(name=q.name,stat=stat(q),sha256=sha(q),committed=False,retained=True))
 guardian=json.loads((run/'guardian-state.json').read_text());need(guardian['phase']=='released','Coarse guardian did not release');telemetry=json.loads((run/'telemetry-v2/termination.json').read_text());rows=subprocess.check_output(['lslocks','-n','-o','PID,MODE,PATH'],text=True);device=[l.strip()for l in rows.splitlines()if'/tmp/uma-paper-device0.lock'in l];need(any(l.startswith('3557276 ')and'WRITE'in l and'WRITE*'not in l for l in device),'Fine successor global holder not observed')
 result=dict(status='PASS_STOPPED_COARSE_ALL_COMMITTED_PREFIX_LABELS',GPU_used=False,read_only_cache=True,resume_or_GPU_launched=False,final_index_complete=False,reason='User requires one simultaneous index build; preserve coarse partial while fine LoTTE continues',tokens_total=266205513,committed_tokens=completed,remaining_tokens=266205513-completed,committed_batches=batches,last_committed_start=committed[-1]['start'],last_committed_count=committed[-1]['count'],all_committed_chunks_SHA_range_continuity_verified=True,computed_complete_prefix_labels_sha256=full.hexdigest(),partial_state_digest_marker=digest,label_range=[lo,hi],identity_sha256=identity['identity_sha256'],cache_identity_file_sha256=sha(identity_path),cache_state_sha256=sha(statepath),cache_state_stat_before_after=before,cache_matrix_and_assignment_graph=payloads,committed_chunks=committed,uncommitted_extra_files=extras,all_partial_material_retained=True,owned_actor_absence=actors,coarse_locks_free=locks,global_lock_view=device,coarse_guardian_release=guardian,observer_termination_receipt_sha256=sha(run/'telemetry-v2/termination.json'),observer_termination=telemetry,existing_telemetry_gap_receipt_sha256=sha(P/'telemetry-transition.json'),supervision_receipt_sha256=sha(receipt/'supervision.json'),result_receipt_sha256=sha(receipt/'result.json'),validator_sha256=sha(Path(__file__)),elapsed_s=time.time()-begin)
 target=P/'stopped-prefix-CPU-validation.json'
 with open(target,'x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:result[k]for k in ['status','committed_tokens','remaining_tokens','committed_batches','computed_complete_prefix_labels_sha256','label_range','uncommitted_extra_files','elapsed_s']}));print(json.dumps({'receipt':str(target),'sha256':sha(target)}))
if __name__=='__main__':main()
