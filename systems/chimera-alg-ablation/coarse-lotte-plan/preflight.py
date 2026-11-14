"""Small CPU-only full-LoTTE C131072 construction startup checks. No CUDA calls/full-array read."""
from pathlib import Path
import json,os,hashlib,shutil,subprocess,time,ast,struct,array,sys

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
 need(mem['MemAvailable']>=spec['minimum_memavailable_bytes'],'MemAvailable below coarse full-build safety threshold')
 need(free>=spec['minimum_disk_free_bytes'],'Disk free below coarse full-build safety threshold')
 return {'time_unix':time.time(),'memory':mem,'disk_free_bytes':free,'new_job_nonreclaimable_budget_bytes':spec['estimated_new_job_nonreclaimable_budget_bytes']}
def verify(spec):
 resources=verify_resources(spec)
 need(cohort_live(spec),'Authorized original offline cohort identities no longer live; refresh review before launch')
 for p,digest in {**spec['control_file_sha256'],**spec['launch_file_sha256']}.items():need(sha(p)==digest,'Control SHA mismatch: '+p)
 command=spec['command'];prov=Path(command[command.index('--provenance-json')+1]);need(sha(prov)==spec['provenance_sha256'],'Provenance mismatch');m=json.loads(prov.read_text());receipt=Path(m['input_derivation_receipt']);need(sha(receipt)==m['input_derivation_receipt_sha256'],'Verified full source receipt changed');verified=json.loads(receipt.read_text());raw=Path(command[command.index('--input')+1]);s=raw.stat();observed={'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
 need(verified['complete'] is True and verified['source_file_sha256']==m['input_sha256']=='d1e02723ade72901d1931fa002cf013a351bbeb9c24ce50e394db1ca5b305538' and verified['tokens_checked']==266205513 and verified['dimensions']==128 and verified['source_revision']=='e3745bd22a3be48d167c438fc8a7a839b86860b1','LoTTE complete source integrity receipt differs')
 reference=Path(spec['source_sha_reference']);need(sha(reference)==spec['source_sha_reference_sha256'],'Accepted live full-LoTTE source SHA reference changed');raw_id=json.loads(reference.read_text())['bindings']['input']
 need(raw_id['sha256']==m['input_sha256'] and {k:observed[k] for k in ('device','inode','mtime_ns')}=={k:raw_id[k] for k in ('device','inode','mtime_ns')} and observed['bytes']==raw_id['size']==68148611456,'Raw source stat differs from the complete measured SHA binding')
 with raw.open('rb') as f:
  need(f.read(6)==b'\x93NUMPY','Invalid NPY signature');version=tuple(f.read(2));width=2 if version==(1,0) else 4 if version in ((2,0),(3,0)) else 0;need(width>0,'Unsupported NPY version');length=int.from_bytes(f.read(width),'little');need(0<length<1048576,'Invalid NPY header size');header=ast.literal_eval(f.read(length).decode('latin1'))
 need(header['shape']==(266205513,128) and header['descr'] in ('<u2','<f2') and header['fortran_order'] is False,'Full LoTTE FP16 NPY shape/order/type changed')
 need(sha(command[command.index('--doclens')+1])==m['doclens_sha256'],'Doclens SHA mismatch')
 with open(command[command.index('--doclens')+1],'rb') as f:
  count=struct.unpack('<i',f.read(4))[0];lens=array.array('i');lens.fromfile(f,count)
  if sys.byteorder!='little':lens.byteswap()
  need(not f.read(1),'Trailing doclens bytes')
 need(count==2428854 and min(lens)==3 and max(lens)==180 and sum(lens)==266205513,'Full LoTTE doclens count/order-scale changed')
 need(sha(command[command.index('--doclens')+1])==m['doclens_sha256'],'Doclens SHA mismatch')
 for option in ('--cache','--index','--spill-full-codes','--receipt-dir'):need(not Path(command[command.index(option)+1]).exists(),'Coarse full-build fresh path exists: '+option)
 need(not Path(command[command.index('--cache')+1]+'.identity.json').exists(),'Fresh cache binding sidecar exists')
 need(command[command.index('--clusters')+1]=='131072' and command[command.index('--ex-bits')+1]=='3' and '--stop-after-batches' not in command and '--resume' not in command,'Coarse full-LoTTE construction parameters changed or stopped early')
 return {'status':'PASS_PREPARED_CPU_ONLY','GPU_used':False,'full_input_hashed':False,'source_full_bytes':s.st_size,'cohort':spec['authorized_offline_cohort'],'resources':resources}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();print(json.dumps(verify(json.loads(Path(a.spec).read_text())),indent=2))
