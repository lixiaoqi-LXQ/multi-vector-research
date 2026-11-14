"""Read-only /proc snapshots. Unreadable fields are unavailable, never inferred exits."""
from pathlib import Path
import hashlib

def failure(errors,path,exc):errors.append(dict(path=str(path),exception=type(exc).__name__,errno=getattr(exc,'errno',None),message=str(exc)))
def proc(pid,expected_start=None):
 root=Path(f'/proc/{pid}');r={'pid':pid,'unavailable':[]};errors=r['unavailable']
 try:
  text=(root/'stat').read_text();v=text[text.rfind(')')+2:].split();r.update(state=v[0],ppid=int(v[1]),start_ticks=int(v[19]),minor_faults=int(v[7]),major_faults=int(v[9]));r['identity_start_match']=expected_start is None or int(v[19])==expected_start
 except (OSError,ValueError,IndexError,UnicodeError) as e:failure(errors,root/'stat',e)
 try:
  for line in (root/'status').read_text().splitlines():
   key,_,value=line.partition(':')
   if key in ('VmRSS','VmHWM','VmSize','RssAnon','RssFile','VmSwap'):r[key+'_kB']=int(value.split()[0])
   elif key=='Cpus_allowed_list':r[key]=value.strip()
 except (OSError,ValueError,IndexError,UnicodeError) as e:failure(errors,root/'status',e)
 try:r['io']={k:int(v) for line in (root/'io').read_text().splitlines() for k,v in [line.split(':',1)]}
 except (OSError,ValueError,UnicodeError) as e:failure(errors,root/'io',e)
 try:r['exe']=str((root/'exe').resolve(strict=True))
 except OSError as e:failure(errors,root/'exe',e)
 group={}
 try:
  for q in (root/'task').iterdir():
   try:group[q.name]=next(x.split(':',1)[1].strip() for x in (q/'status').read_text().splitlines() if x.startswith('Cpus_allowed_list:'))
   except (OSError,StopIteration,UnicodeError) as e:failure(errors,q/'status',e)
 except OSError as e:failure(errors,root/'task',e)
 r['thread_allowed_cpus']=group;r['observation_status']='partial_unavailable' if errors else 'available';return r

def bind_cmdline(record,expected):
 path=Path(f"/proc/{record['pid']}/cmdline")
 try:record['cmdline_sha_matches']=hashlib.sha256(path.read_bytes()).hexdigest()==expected
 except OSError as e:record['cmdline_sha_matches']=None;failure(record.setdefault('unavailable',[]),path,e)
 return record
