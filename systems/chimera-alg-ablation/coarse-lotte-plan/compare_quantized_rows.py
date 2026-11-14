"""Direct CPU all-byte fine/coarse global quantization comparison (header-aware)."""
from pathlib import Path
import argparse,hashlib,json,struct,tempfile,time
P=Path(__file__).resolve().parent
FINE=Path('/home/lxq/graduate/data/multivector-colbertv2/indexes/chimera/lotte-pooled-dev-full-c2097152-b4-owned-device-checkpoint-v1')
COARSE=Path('/home/lxq/graduate/data/multivector-colbertv2/indexes/chimera/lotte-pooled-dev-full-c131072-b4-fig8-coarse-v1')
def need(value,message):
 if not value:raise ValueError(message)
def stat(p):
 s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns)
def compare_stream(a,b,skip=0):
 before=[stat(a),stat(b)];need(before[0]['size']==before[1]['size'],'Global quantization file lengths differ');digests=[hashlib.sha256(),hashlib.sha256()];body=hashlib.sha256();compared=0
 with a.open('rb')as x,b.open('rb')as y:
  for f,h in zip([x,y],digests):h.update(f.read(skip))
  while True:
   u=x.read(8<<20);v=y.read(8<<20);need(u==v,'Global quantization row bytes differ: '+a.name)
   if not u:break
   digests[0].update(u);digests[1].update(v);body.update(u);compared+=len(u)
 need([stat(a),stat(b)]==before,'Files changed during direct byte comparison')
 return {'all_compared_bytes_equal':True,'compared_bytes':compared,'header_bytes_excluded':skip,'equal_body_sha256':body.hexdigest(),'fine_fullfile_sha256':digests[0].hexdigest(),'coarse_fullfile_sha256':digests[1].hexdigest(),'before_after_unchanged':before}
def compare(fine,coarse):
 with (fine/'doc_1bit.bin').open('rb')as f:ah=f.read(40)
 with (coarse/'doc_1bit.bin').open('rb')as f:bh=f.read(40)
 a=struct.unpack('<5Q',ah);b=struct.unpack('<5Q',bh);need(a[0:2]==b[0:2]and a[3:]==b[3:]and a[0]>0 and a[1]==a[4]==128 and a[3]==3,'Fine/coarse dimension/bitwidth differs')
 need(ah[:16]==bh[:16]and ah[24:]==bh[24:],'Index headers differ beyond cluster count field')
 n=a[0];need((fine/'doc_full.bin').stat().st_size==64*n and(coarse/'doc_full.bin').stat().st_size==64*n,'Fullcode file byte count differs from4bit representation')
 need((fine/'doc_1bit.bin').stat().st_size-40-24*n==64 and(coarse/'doc_1bit.bin').stat().st_size-40-24*n==64,'Unexpected serialized rotator layout')
 output={name:compare_stream(fine/name,coarse/name,40 if name=='doc_1bit.bin'else 0)for name in ['doc_1bit.bin','doc_full.bin','doclens.bin']}
 return {'fine_header':a,'coarse_header':b,'allowed_header_difference':'only size_t num_clusters atbytes16..23','all_rotator_onebit_codes_factors_fullcodes_doclens_bytes_exact':True,'files':output}
def verify():
 started=time.time();metas=[json.loads((p/'build-provenance.json').read_text())for p in [FINE,COARSE]]
 for i,(path,meta)in enumerate(zip([FINE,COARSE],metas)):
  need(meta['complete']is True and(meta['docs'],meta['tokens'],meta['ex_bits'])==(2428854,266205513,3),'Full fine/coarse index is incomplete or wrong dataset')
  need(meta['n_clusters']==([2097152,131072][i]),'Fine/coarse cluster count is wrong')
 need(metas[0]['source_identity']==metas[1]['source_identity'],'Fine/coarse source revisions differ')
 result=compare(FINE,COARSE)
 need(result['fine_header'][2]==2097152 and result['coarse_header'][2]==131072,'Actual header cluster counts differ from declared fine/coarse indexes')
 for name,detail in result['files'].items():
  need(detail['fine_fullfile_sha256']==metas[0]['index_identity']['files'][name]['sha256']and detail['coarse_fullfile_sha256']==metas[1]['index_identity']['files'][name]['sha256'],'Read file bytes no longer match accepted build hashes')
 return {'status':'PASS_FULL_FINE_COARSE_QUANTIZATION_ALL_BYTES','GPU_used':False,'performance_claim':False,'fine_index':str(FINE),'coarse_index':str(COARSE),'comparison':result,'elapsed_seconds':time.time()-started,'scope':'global token-row representation fixed while routing geometry/CR vary; clustered payload/IVF/graphs intentionally rebuilt','validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
def cpu_test():
 root=P/'cpu-fixtures';root.mkdir(exist_ok=True)
 with tempfile.TemporaryDirectory(dir=root)as t:
  base=Path(t);fine=base/'fine';coarse=base/'coarse';fine.mkdir();coarse.mkdir();n=11
  body=bytes(range(64))+bytes((i%251 for i in range(24*n)));full=bytes((i%251 for i in range(64*n)));lens=struct.pack('<4i',3,1,3,7)
  for path,c in [(fine,5),(coarse,3)]:
   (path/'doc_1bit.bin').write_bytes(struct.pack('<5Q',n,128,c,3,128)+body);(path/'doc_full.bin').write_bytes(full);(path/'doclens.bin').write_bytes(lens)
  compare(fine,coarse);rejected=0
  for name in ['doc_1bit.bin','doc_full.bin','doclens.bin']:
   p=coarse/name;original=p.read_bytes();changed=bytearray(original);changed[-1]^=1;p.write_bytes(changed)
   try:compare(fine,coarse)
   except ValueError:rejected+=1
   p.write_bytes(original)
  p=coarse/'doc_1bit.bin';original=p.read_bytes();changed=bytearray(original);changed[0]^=1;p.write_bytes(changed)
  try:compare(fine,coarse)
  except ValueError:rejected+=1
  need(rejected==4,'Corrupt quantization/header fixtures accepted')
 return {'status':'PASS_CPU_HEADER_AWARE_QUANTIZATION_COMPARATOR','valid_header_C_difference_accepted':True,'corrupt_payload_or_other_header_cases_rejected':4,'GPU_used':False,'full_N_IO_executed':False}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--cpu-test',action='store_true');a=p.parse_args()
 if a.cpu_test:print(json.dumps(cpu_test()));raise SystemExit
 if not a.run:print(json.dumps({'prepared_only':True,'GPU_used':False,'full_N_IO_executed':False,'fine':str(FINE),'coarse':str(COARSE),'expected_compared_bytes':24*266205513+64+64*266205513+9715420}));raise SystemExit
 target=P/'fine-coarse-quantization-byte-comparison.json';need(not target.exists(),'Retain previous comparison receipt');result=verify();target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status']}))
