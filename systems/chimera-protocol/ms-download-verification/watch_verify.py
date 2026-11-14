#!/usr/bin/env python3
"""Observe one pinned producer; verify its completed file without modifying arrays."""
import datetime,hashlib,json,os,stat,sys,tempfile,time,traceback
from pathlib import Path
ROOT=Path('/home/lxq/graduate/data/multivector-colbertv2/msmarco-v1-passage')
WORK=Path(__file__).resolve().parent
EVIDENCE=Path('/home/lxq/graduate/data/memory/multivector-msmarco-full-array')
PID=2512667;TOTAL=153064939392;EXPECTED='3d83999077ca98e3fb24f0bf2b7aa41eeb607f30ecda9d20decf6aa9ed3da690'
REPO='tuskanny/ms_marco_colbertv2';REV='f8540899613ccc13049ac87db1cc4041a1dc4e31'
STATE=WORK/'state.json';FINAL=ROOT/'documents.npy';PARTIAL=ROOT/'documents.npy.partial'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,value):
 fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
  directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(directory)
  finally:os.close(directory)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def proc():
 try:
  text=Path(f'/proc/{PID}/stat').read_text();fields=text[text.rfind(')')+2:].split();return {'state':fields[0],'starttime_ticks':fields[19],'cmdline':Path(f'/proc/{PID}/cmdline').read_bytes().replace(b'\0',b' ').decode().strip()}
 except FileNotFoundError:return None
def need(v,message):
 if not v:raise ValueError(message)
def snap(s):return {'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
def check_provenance(m):
 need(m['repository']==REPO and m['revision']==REV,'Pinned repository/revision changed')
 for name,detail in m['files'].items():need(detail['url']==f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{name}','Pinned URL changed: '+name)
 d=m['files']['documents.npy'];need(d['bytes']==TOTAL and d['lfs_sha256']==EXPECTED,'Pinned expected full-array identity changed')
def main():
 os.sched_setaffinity(0,{2});EVIDENCE.mkdir(parents=True,exist_ok=True)
 initial=proc();need(initial is not None and initial['state']!='Z','Expected original producer already absent at watcher startup')
 need('fetch_array.py' in initial['cmdline'] and str(ROOT) in initial['cmdline'],'Producer command identity mismatch')
 base={'watcher_pid':os.getpid(),'producer_pid':PID,'producer_identity':initial,'cpu_affinity':[2],'started_utc':utc(),'expected_bytes':TOTAL,'expected_sha256':EXPECTED,'gpu_used':False,'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 check_provenance(json.loads((ROOT/'download-provenance.json').read_text()));atomic(STATE,base|{'status':'WAITING_SAME_PRODUCER'})
 print(json.dumps(base|{'status':'WAITING_SAME_PRODUCER'}),flush=True)
 while True:
  current=proc();same=current is not None and current['starttime_ticks']==initial['starttime_ticks'];terminal=not same or current['state']=='Z'
  if FINAL.exists():
   need(FINAL.stat().st_size==TOTAL,'Final array wrong size')
   need(not PARTIAL.exists(),'Both final and partial array names exist')
   if terminal:break
  elif terminal:raise RuntimeError('Same producer terminated without completed final array')
  atomic(STATE,base|{'status':'WAITING_SAME_PRODUCER','observed_utc':utc(),'producer_state':current['state'] if same else 'terminal','partial_bytes':PARTIAL.stat().st_size if PARTIAL.exists() else None,'final_exists':FINAL.exists()})
  time.sleep(10)
 fd=os.open(FINAL,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
 try:
  before=os.fstat(fd);need(stat.S_ISREG(before.st_mode) and before.st_size==TOTAL,'Final is not expected regular file')
  need(snap(FINAL.stat())==snap(before),'Final inode differs from opened file')
  start=time.monotonic();h=hashlib.sha256();done=0;last_report=start
  atomic(STATE,base|{'status':'HASHING_CORE2','hash_started_utc':utc(),'file_before':snap(before),'bytes_hashed':0})
  while True:
   block=os.read(fd,16*1024*1024)
   if not block:break
   h.update(block);done+=len(block)
   if time.monotonic()-last_report>=10:
    atomic(STATE,base|{'status':'HASHING_CORE2','file_before':snap(before),'bytes_hashed':done,'observed_utc':utc()});last_report=time.monotonic()
  after=os.fstat(fd);need(done==TOTAL and snap(before)==snap(after)==snap(FINAL.stat()),'Array identity/stat changed across full hash')
  actual=h.hexdigest();need(actual==EXPECTED,'FULL ARRAY SHA256 MISMATCH: '+actual)
 finally:os.close(fd)
 evidence=base|{'status':'VERIFIED_COMPLETE','producer_terminal_before_hash':True,'verified_utc':utc(),'actual_sha256':actual,'actual_bytes':done,'file_before':snap(before),'file_after':snap(after),'hash_seconds':time.monotonic()-start,'array_modified':False}
 ep=EVIDENCE/('documents-full-sha256-'+str(os.getpid())+'.json');need(not ep.exists(),'Evidence path exists');atomic(ep,evidence)
 # Reload immediately before update; preserve all unrelated fields and all pinned URLs.
 pp=ROOT/'download-provenance.json';meta=json.loads(pp.read_text());check_provenance(meta);atomic(EVIDENCE/('download-provenance-before-'+str(os.getpid())+'.json'),meta)
 updated=json.loads(json.dumps(meta));detail=updated['files']['documents.npy'];detail.update(sha256=actual,verified_full_sha256=actual,status='verified complete local file',verified_utc=evidence['verified_utc'],verification_evidence=str(ep))
 check_provenance(updated);need(snap(FINAL.stat())==snap(after),'Array changed before provenance publication');atomic(pp,updated)
 reread=json.loads(pp.read_text());need(reread==updated,'Published provenance mismatch');atomic(EVIDENCE/('download-provenance-after-'+str(os.getpid())+'.json'),updated)
 evidence['provenance_updated']=True;evidence['provenance_sha256']=hashlib.sha256(pp.read_bytes()).hexdigest();atomic(ep,evidence);atomic(STATE,evidence)
 print(json.dumps(evidence),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as error:
  failure={'status':'FAILED','watcher_pid':os.getpid(),'producer_pid':PID,'failed_utc':utc(),'error':str(error),'traceback':traceback.format_exc(),'array_modified':False,'other_processes_stopped':False}
  atomic(STATE,failure);atomic(EVIDENCE/('FAILED-'+str(os.getpid())+'.json'),failure);print(json.dumps(failure),flush=True);raise
