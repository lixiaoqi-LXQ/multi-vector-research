"""Verify pinned file SHA and every expanded FP32 value outside QPS windows."""
from pathlib import Path
import numpy as np,json,hashlib,time,os
root=Path('/home/lxq/graduate/data/multivector-colbertv2')
src=root/'lotte-pooled-dev';dst=root/'chimera-inputs/lotte-pooled-dev-full'
p=json.loads((src/'download-provenance.json').read_text());m=json.loads((dst/'manifest.json').read_text())
sp=src/'documents.npy';rp=dst/'embeddings.bin';n,d=m['tokens'],m['dimension']
assert m['complete'] and sp.stat().st_size==p['files']['documents.npy']['bytes'] and rp.stat().st_size==8+n*d*4
s=np.load(sp,mmap_mode='r',allow_pickle=False);assert s.shape==(n,d) and s.dtype==np.uint16
s=s.view(np.float16)
with open(rp,'rb') as f:rh=f.read(8)
assert np.array_equal(np.frombuffer(rh,dtype='<i4'),[n,d])
r=np.memmap(rp,dtype='<f4',mode='r',offset=8,shape=(n,d))
source_offset=sp.stat().st_size-n*d*2
with open(sp,'rb') as f:header=f.read(source_offset)
hs=hashlib.sha256(header);hr=hashlib.sha256(rh);started=time.time()
for begin in range(0,n,131072):
 end=min(n,begin+131072);raw=np.ascontiguousarray(s[begin:end]);actual=np.ascontiguousarray(r[begin:end]);expected=raw.astype(np.float32)
 if not np.isfinite(expected).all():raise RuntimeError(f'Nonfinite source vector near{begin}')
 if not np.array_equal(actual.view(np.uint32),expected.view(np.uint32)):raise RuntimeError(f'Expansion mismatch near{begin}')
 hs.update(raw.tobytes());hr.update(actual.tobytes())
 if begin%(131072*128)==0:print(json.dumps({'tokens_checked':end,'total_tokens':n,'elapsed_s':time.time()-started}),flush=True)
source_sha=hs.hexdigest();raw_sha=hr.hexdigest();assert source_sha==p['files']['documents.npy']['lfs_sha256'],(source_sha,p['files']['documents.npy'])
receipt={'complete':True,'source_file_sha256':source_sha,'fp32_file_sha256':raw_sha,'tokens_checked':n,'dimensions':d,'float_values_bitwise_verified':n*d,'all_source_values_finite':True,'elapsed_s':time.time()-started,'source_revision':p['revision'],'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
a=Path('/home/lxq/graduate/ai-workspace/research/multivector/data/lotte-full-integrity.json');a.write_text(json.dumps(receipt,indent=2)+'\n')
p['files']['documents.npy'].update(sha256=source_sha,checksum_status='verified_full_local_file_against_pinned_LFS_SHA256',verified_at_unix=time.time());tmp=src/'download-provenance.json.partial';tmp.write_text(json.dumps(p,indent=2)+'\n');os.replace(tmp,src/'download-provenance.json')
m['artifacts']={}
for name in ['embeddings.bin','doclens.bin','queries.bin']:
 f=dst/name;sha=raw_sha if name=='embeddings.bin' else hashlib.sha256(f.read_bytes()).hexdigest();m['artifacts'][name]={'bytes':f.stat().st_size,'sha256':sha}
m['full_expansion_integrity']=receipt;tmp=dst/'manifest.json.partial';tmp.write_text(json.dumps(m,indent=2)+'\n');os.replace(tmp,dst/'manifest.json')
print(json.dumps(receipt),flush=True)
