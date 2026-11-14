"""Small CPU-only full-MS Q1 startup checks. No CUDA calls/full-array read."""
from pathlib import Path
import json,os,hashlib,shutil,subprocess,time

def need(v,msg):
 if not v:raise ValueError(msg)
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def process(pid):
 try:
  t=Path(f'/proc/{pid}/stat').read_text();parts=t[t.rfind(')')+2:].split();return {'pid':pid,'start_ticks':int(parts[19]),'cmdline_sha256':hashlib.sha256(Path(f'/proc/{pid}/cmdline').read_bytes()).hexdigest(),'state':parts[0]}
 except FileNotFoundError:return None
def cohort_live(spec):
 for e in spec['authorized_offline_cohort']:
  now=process(e['pid'])
  if now is None or now['state']=='Z' or any(now[k]!=e[k] for k in ('pid','start_ticks','cmdline_sha256')):return False
 return True
def verify_resources(spec):
 mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:','MemAvailable:'))};free=shutil.disk_usage(spec['disk_path']).free
 need(mem['MemAvailable']>=spec['minimum_memavailable_bytes'],'MemAvailable below Q1 safety threshold')
 need(free>=spec['minimum_disk_free_bytes'],'Disk free below Q1 safety threshold')
 return {'time_unix':time.time(),'memory':mem,'disk_free_bytes':free,'new_job_nonreclaimable_budget_bytes':spec['estimated_new_job_nonreclaimable_budget_bytes']}
def verify(spec):
 resources=verify_resources(spec)
 need(cohort_live(spec),'Authorized original offline cohort identities no longer live; refresh review before launch')
 for p,digest in {**spec['control_file_sha256'],**spec['launch_file_sha256']}.items():need(sha(p)==digest,'Control SHA mismatch: '+p)
 command=spec['command'];prov=Path(command[command.index('--provenance-json')+1]);need(sha(prov)==spec['provenance_sha256'],'Provenance mismatch');m=json.loads(prov.read_text());receipt=Path(m['input_derivation_receipt']);need(sha(receipt)==m['input_derivation_receipt_sha256'],'Verified full source receipt changed');verified=json.loads(receipt.read_text());raw=Path(command[command.index('--input')+1]);s=raw.stat();observed={'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
 need(verified['actual_sha256']==m['input_sha256']=='3d83999077ca98e3fb24f0bf2b7aa41eeb607f30ecda9d20decf6aa9ed3da690' and observed==verified['file_after'],'Raw full source identity differs from verified source')
 need(sha(command[command.index('--doclens')+1])==m['doclens_sha256'],'Doclens SHA mismatch')
 for option in ('--index','--spill-full-codes','--receipt-dir'):need(not Path(command[command.index(option)+1]).exists(),'Fresh resume output exists: '+option)
 need('--resume' in command and '--stop-after-batches' not in command,'Whole resume must remove stop and explicitly resume')
 need(command[command.index('--clusters')+1]=='4194304' and command[command.index('--ex-bits')+1]=='3','Full-MS dimensions changed')
 cache=Path(command[command.index('--cache')+1]);binding=Path(str(cache)+'.identity.json');need(sha(binding)==spec['cache_binding_sha256'],'Q1 immutable binding changed')
 bound=json.loads(binding.read_text());need(bound['identity_sha256']==spec['checkpoint_identity_sha256'],'Checkpoint identity changed')
 for entry in spec['same_Q1_control_identity']:
  file=Path(entry['path']);st=file.stat();need([st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns]==[entry[k] for k in ('device','inode','size','mtime_ns')],'Original Q1 control inode/stat changed: '+str(file))
 need(sha(cache/'state.txt')==spec['expected_checkpoint_state_sha256'],'Prepared checkpoint prefix changed; refresh resume plan')
 for name,expected in spec['verified_cache_artifacts'].items():
  file=cache/name;st=file.stat();need([st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns]==[expected[k] for k in ('device','inode','size','mtime_ns')],'Audited Q1 cache artifact stat changed: '+name)
 need(sha(cache/'labels/0.u32')==spec['expected_labels_sha256'],'Q1 first-label identity changed')
 audit=Path(spec['large_restore_audit_validation']);need(sha(audit)==spec['large_restore_audit_validation_sha256'],'Large restore validation changed');need(json.loads(audit.read_text())['status']=='PASS_FULL_MS_Q1_LARGE_RESTORE_BOUND','Large restore audit not accepted')
 need(sha(Path(spec['large_restore_audit_result']))==spec['large_restore_audit_result_sha256'],'Large restore result changed')

 return {'status':'PASS_PREPARED_CPU_ONLY','GPU_used':False,'full_input_hashed':False,'source_full_bytes':s.st_size,'cohort':spec['authorized_offline_cohort'],'resources':resources}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();print(json.dumps(verify(json.loads(Path(a.spec).read_text())),indent=2))
