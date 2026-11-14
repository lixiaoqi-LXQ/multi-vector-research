"""Live /proc descendants with starttime identities; no NVML/GPU calls."""
import os
from pathlib import Path

def process_identity(pid):
 try:
  data=Path(f'/proc/{pid}/stat').read_text();fields=data.rsplit(')',1)[1].split()
  return {'pid':int(pid),'ppid':int(fields[1]),'starttime_ticks':int(fields[19]),'state':fields[0]}
 except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError,IndexError):return None

def live_tree(root_pid=None):
 root_pid=os.getpid() if root_pid is None else root_pid;pending=[int(root_pid)];result={}
 while pending:
  pid=pending.pop();before=process_identity(pid)
  if before is None or pid in result:continue
  children=set()
  try:
   for path in Path(f'/proc/{pid}/task').glob('*/children'):
    try:children.update(map(int,path.read_text().split()))
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
  except (FileNotFoundError,ProcessLookupError):continue
  after=process_identity(pid)
  if after is None or before['starttime_ticks']!=after['starttime_ticks']:continue
  result[pid]=after
  for child in children:
   row=process_identity(child)
   if row and row['ppid']==pid:pending.append(child)
 return result

def classify_compute_pids(pids,root_pid=None):
 own=live_tree(root_pid);attribution=[]
 for pid in pids:
  current=process_identity(pid);observed=own.get(pid)
  match=current is not None and observed is not None and current['starttime_ticks']==observed['starttime_ticks']
  attribution.append({'pid':int(pid),'classification':'own_live_tree' if match else 'unattributed_inventory','process_identity':current})
 return {'own_live_tree':list(own.values()),'compute_pid_attribution':attribution,'unattributed_compute_pids':[x['pid'] for x in attribution if x['classification']=='unattributed_inventory']}
