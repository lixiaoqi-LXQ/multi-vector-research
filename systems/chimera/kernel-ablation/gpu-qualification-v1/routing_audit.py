"""CPU-only semantic routing-dataset acceptance guard for future Chimera runs."""
import hashlib,json,re
from pathlib import Path
MARKER='ROUTING_DATASET_UNVERIFIED.json'
def need(value,message):
 if not value:raise ValueError(message)
def sha(path):
 with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def reject_quarantined(index):
 marker=Path(index)/MARKER
 need(not marker.exists() and not marker.is_symlink(),'Routing dataset is quarantined/unverified: '+str(marker))
def verify_routing_dataset(spec,index_identity):
 index=Path(spec['index']);reject_quarantined(index)
 path=spec.get('routing_dataset_audit_receipt');digest=spec.get('routing_dataset_audit_sha256')
 need(isinstance(path,str) and isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest),'Require pinned saved routing dataset semantic audit receipt')
 path=Path(path).resolve(strict=True);need(sha(path)==digest,'Routing dataset audit receipt SHA mismatch');audit=json.loads(path.read_text())
 need(audit.get('status')=='PASS_POST_ENCODE_SAVE_DATASET','Routing dataset semantic audit did not pass')
 need(audit.get('rows')==index_identity['n_clusters'] and audit.get('dimension')==index_identity['paddedD'],'Routing dataset audit shape mismatch')
 need(audit.get('rotation_seed')==42 and audit.get('rotator')=='FhtKacRotator','Routing dataset audit rotation mismatch')
 expected=audit.get('expected_rotated_dataset_sha256');actual=audit.get('actual_saved_dataset_sha256')
 need(isinstance(expected,str) and re.fullmatch('[0-9a-f]{64}',expected) and expected==actual and audit.get('float32_bit_mismatches')==0,'Saved routing dataset differs from expected rotated centroids')
 need(audit.get('dataset_memory_type')==2,'Saved routing dataset was not audited via device-owned loaded index')
 graph=index_identity['files']['centroids.carga']['sha256'];need(audit.get('final_cagra_file_sha256')==graph,'Routing dataset audit is for a different centroids.carga')
 # A derived8 index can reuse this receipt because its routing graph bytes are identical.
 inherited=None
 build=index/'build-provenance.json'
 if build.exists():
  meta=json.loads(build.read_text())
  if meta.get('fresh_8bit_build') is False:
   source=Path(meta['source4_index']).resolve(strict=True);reject_quarantined(source)
   a=(source/'centroids.carga').stat();b=(index/'centroids.carga').stat()
   need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns),'Derived8 must share the same source4 routing graph inode')
   inherited=str(source)
 return {'verified':True,'receipt':str(path),'receipt_sha256':digest,'centroids_cagra_sha256':graph,'expected_rotated_dataset_sha256':expected,'source4_inherited_from':inherited,'guard_sha256':sha(__file__)}
