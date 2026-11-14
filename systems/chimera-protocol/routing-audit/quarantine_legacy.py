"""Metadata-only scientific quarantine; preserve six payloads and prior receipts."""
import json,hashlib,os,tempfile,time
from pathlib import Path
R=Path('/home/lxq/graduate/ai-workspace/research/multivector');source=R/'chimera/review/centroid-ownership/affected-indexes.json';doc=json.loads(source.read_text());NAMES=['centroids.carga','cluster_1bit.bin','doc_1bit.bin','doc_full.bin','doclens.bin','ivf.bin']
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stat(p):
 s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
records=[]
for entry in doc['indexes']:
 index=Path(entry['index'])
 if not all((index/n).is_file() for n in NAMES):records.append({'index':str(index),'status':'SKIPPED_NOT_COMPLETE_SIX_PAYLOAD_INDEX'});continue
 before={n:stat(index/n) for n in NAMES};receipts={p.name:sha(p) for p in index.glob('*.json') if p.name!='ROUTING_DATASET_UNVERIFIED.json'}
 mark={'schema_version':1,'status':'ROUTING_DATASET_UNVERIFIED','formal_E2E_comparison_allowed':False,'reason':doc['reason'],'affected_inventory':str(source),'affected_inventory_sha256':sha(source),'centroids_cagra_sha256':sha(index/'centroids.carga'),'payload_stat_before':before,'preexisting_receipt_sha256':receipts,'required_requalification':'saved rotated centroid dataset semantic audit against accepted fixed source; new owned-device primary indexes use separate paths','created_unix':time.time()}
 target=index/'ROUTING_DATASET_UNVERIFIED.json'
 if target.exists():
  previous=json.loads(target.read_text());assert previous['status']==mark['status'] and previous['centroids_cagra_sha256']==mark['centroids_cagra_sha256'];status='REUSED_EXISTING_QUARANTINE'
 else:
  fd,tmp=tempfile.mkstemp(prefix='.routing-quarantine-',dir=index)
  try:
   with os.fdopen(fd,'w') as f:json.dump(mark,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
   os.link(tmp,target);directory=os.open(index,os.O_RDONLY|os.O_DIRECTORY);os.fsync(directory);os.close(directory)
  finally:os.unlink(tmp)
  status='QUARANTINED_METADATA_ONLY'
 assert {n:stat(index/n) for n in NAMES}==before
 assert all(sha(index/n)==digest for n,digest in receipts.items())
 records.append({'index':str(index),'status':status,'marker_sha256':sha(target),'six_payload_stats_unchanged':True,'prior_receipt_sha256_unchanged':True,'centroids_cagra_sha256':mark['centroids_cagra_sha256']})
out=R/'chimera-protocol/routing-audit/quarantine-receipt.json';assert not out.exists();out.write_text(json.dumps({'status':'COMPLETE','GPU_used':False,'source_inventory_sha256':sha(source),'records':records},indent=2)+'\n');print(json.dumps(records,indent=2))
