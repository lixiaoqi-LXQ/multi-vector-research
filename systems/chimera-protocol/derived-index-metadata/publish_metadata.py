#!/usr/bin/env python3
"""Verify an accepted requantized8 index and optionally publish metadata only."""
import argparse,hashlib,json,os,re,struct,sys,tempfile
from pathlib import Path
FILES=('centroids.carga','cluster_1bit.bin','doc_1bit.bin','doc_full.bin','doclens.bin','ivf.bin')
FIXED=('centroids.carga','cluster_1bit.bin','doclens.bin','ivf.bin')
OP='CPU requantize from fixed 4-bit index to 8-bit'
def need(v,msg):
 if not v:raise ValueError(msg)
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stat4(p):
 s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
def read(p):return json.loads(p.read_text())
def identity(index):
 with open(index/'doc_1bit.bin','rb') as f:h=struct.unpack('<5Q',f.read(40))
 files={}
 for n in FILES:
  p=index/n;before=stat4(p);digest=sha(p);need(before==stat4(p),'Payload changed during hashing: '+n)
  files[n]={'size':before[2],'mtime_ns':before[3],'inode':before[1],'device':before[0],'sha256':digest}
 return dict(zip(('N','D','n_clusters','ex_bits','paddedD'),h),total_bits=h[3]+1,files=files)
def prepare(index,source4_receipt,expected_requant_sha,qualification,expected_qualification_sha):
 index=Path(index).resolve(strict=True);source4_receipt=Path(source4_receipt).resolve(strict=True);qualification=Path(qualification).resolve(strict=True)
 need(not (index/'build-provenance.json').exists(),'Refusing existing build-provenance.json')
 rp=index/'requantize-receipt.json';need(sha(rp)==expected_requant_sha,'Requantize receipt SHA mismatch');r=read(rp)
 need(r['schema']==1 and r['operation']==OP and r['fresh_8bit_build'] is False,'Unsupported derived operation')
 need(sha(qualification)==expected_qualification_sha,'Qualification receipt SHA mismatch');q=read(qualification)
 need(q['status']=='PASS_CPU_BYTE_QUALIFICATION' and q['binary_sha256']==r['tool_sha256'],'Tool is not the qualified binary')
 need(all(q.get(k) is True for k in ('source4_all_six_payloads_preserved','rotator_onebit_codes_onebit_factors_preserved','raw_sha_mutation_rejected','existing_output_rejected','source_and_accepted_output_preserved_after_negative_tests')),'Incomplete qualification evidence')
 need(sha(source4_receipt)==r['source_build_receipt_sha256'],'Source4 build receipt SHA mismatch');s=read(source4_receipt)
 need(s.get('complete') is True and s['total_bits']==4 and s['ex_bits']==3,'Source4 build not complete or wrong bits')
 need(s.get('fresh_8bit_build') is not False,'Source4 must be original build provenance')
 source=Path(r['inputs']['doc_1bit.bin']['path']).resolve(strict=True).parent
 source_id=identity(source);need(source_id['ex_bits']==3,'Source4 header bits mismatch')
 for n in FILES:
  recorded=r['inputs'][n];p=source/n;need(Path(recorded['path']).resolve(strict=True)==p,'Source payload path mismatch: '+n)
  need(source_id['files'][n]['sha256']==recorded['sha256']==s['index_identity']['files'][n]['sha256'],'Source4 payload SHA mismatch: '+n)
  need(stat4(p)==recorded['stat'],'Source4 stat identity changed: '+n)
 raw=Path(r['inputs']['raw_fp16_npy']['path']).resolve(strict=True);need(raw==Path(s['vector_input']).resolve(strict=True),'Raw input differs from original source4 build')
 before=stat4(raw);need(before==r['inputs']['raw_fp16_npy']['stat'],'Raw stat identity changed')
 original_raw=s['input_files']['vector_input'];need(before==[original_raw['device'],original_raw['inode'],original_raw['size'],original_raw['mtime_ns']],'Raw differs from original source4 build stat identity')
 need(sha(raw)==r['inputs']['raw_fp16_npy']['sha256'],'Raw SHA mismatch');need(stat4(raw)==before,'Raw changed during hashing')
 current=identity(index);need(current['ex_bits']==7 and current['D']==current['paddedD']==128,'Derived header incompatible')
 need((current['N'],current['D'],current['n_clusters'],current['paddedD'])==(source_id['N'],source_id['D'],source_id['n_clusters'],source_id['paddedD']),'Source/derived header dimensions mismatch')
 need(s['tokens']==current['N'] and s['n_clusters']==current['n_clusters'],'Source build dimensions mismatch')
 val=r['validation'];need(val['all_rows_bitwise_source_verified'] is True and val['gpu_used'] is False and (val['tokens'],val['dimension'],val['source_bits'],val['destination_bits'])==(current['N'],current['D'],4,8),'Missing all-row invariant verification')
 need(current['files']['doc_full.bin']['size']==current['N']*128,'Invalid full8 payload size')
 with open(index/'doclens.bin','rb') as f:
  count=struct.unpack('<i',f.read(4))[0];lens=f.read();need(len(lens)==count*4 and count==s['docs'],'Document count mismatch');need(sys.byteorder=='little','Native index ABI requires little-endian host');lengths=memoryview(lens).cast('i')
 need(all(x>0 for x in lengths) and sum(lengths)==current['N'],'Document lengths mismatch')
 for n in FILES:
  need(current['files'][n]['sha256']==r['outputs'][n]['sha256'] and current['files'][n]['size']==r['outputs'][n]['bytes'],'Derived payload SHA/size mismatch: '+n)
 for n in FIXED:need(stat4(index/n)[:2]==stat4(source/n)[:2],'Reused file is not source hardlink: '+n)
 src_identity=s['source_identity'];need(all(isinstance(src_identity[k],str) for k in ('repository','revision','documents_url')),'Incomplete source identity')
 need(re.fullmatch('[0-9a-f]{40}',src_identity['revision']) is not None,'Source revision must be fixed SHA');need(src_identity['documents_url']==f"https://huggingface.co/datasets/{src_identity['repository']}/resolve/{src_identity['revision']}/documents.npy",'Unpinned source identity')
 result={'schema_version':1,'complete':True,'operation':OP,'fresh_8bit_build':False,'docs':count,'tokens':current['N'],'dimension':current['D'],'n_clusters':current['n_clusters'],'ex_bits':7,'total_bits':8,'source_identity':src_identity,'index_identity':current,'source4_index':str(source),'source4_build_receipt':str(source4_receipt),'source4_build_receipt_sha256':sha(source4_receipt),'requantize_receipt':str(rp),'requantize_receipt_sha256':expected_requant_sha,'qualification_receipt':str(qualification),'qualification_receipt_sha256':expected_qualification_sha,'requantize_tool_sha256':r['tool_sha256'],'raw_fp16_sha256':r['inputs']['raw_fp16_npy']['sha256'],'metadata_adapter_sha256':sha(__file__),'timing_scope':'CPU-derived requantization; fresh8 build time is not measured','seconds_requantize_and_validate':r['seconds_requantize_and_validate']}
 # Verify identities again before returning/publishing; only metadata may be added.
 for directory,ident in [(source,source_id),(index,current)]:
  for n,detail in ident['files'].items():need(stat4(directory/n)==[detail['device'],detail['inode'],detail['size'],detail['mtime_ns']],'Input changed during metadata verification')
 need(sha(rp)==expected_requant_sha and sha(source4_receipt)==r['source_build_receipt_sha256'],'Receipt changed during verification')
 return result

def publish(index,metadata):
 index=Path(index);target=index/'build-provenance.json';need(not target.exists(),'Refusing existing build-provenance.json')
 for n,d in metadata['index_identity']['files'].items():need(stat4(index/n)==[d['device'],d['inode'],d['size'],d['mtime_ns']],'Payload changed before metadata publication')
 need(sha(index/'requantize-receipt.json')==metadata['requantize_receipt_sha256'],'Derived receipt changed before publication')
 fd,tmp=tempfile.mkstemp(prefix='.derived-metadata-',dir=index)
 try:
  with os.fdopen(fd,'w') as f:json.dump(metadata,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  os.link(tmp,target) # Atomic exclusive publication; never overwrite existing metadata.
  directory=os.open(index,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(directory)
  finally:os.close(directory)
 finally:
  os.unlink(tmp)

def main():
 p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--source4-receipt',required=True);p.add_argument('--requantize-receipt-sha256',required=True);p.add_argument('--qualification',required=True);p.add_argument('--qualification-sha256',required=True);p.add_argument('--preview',required=True);p.add_argument('--publish',action='store_true');a=p.parse_args()
 m=prepare(a.index,a.source4_receipt,a.requantize_receipt_sha256,a.qualification,a.qualification_sha256)
 preview=Path(a.preview);need(not preview.exists(),'Refusing existing preview');preview.write_text(json.dumps(m,indent=2)+'\n')
 if a.publish:publish(a.index,m)
 print(json.dumps({'published':a.publish,'preview':str(preview),'index':a.index}))
if __name__=='__main__':main()
