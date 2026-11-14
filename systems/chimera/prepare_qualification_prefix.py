"""Exact small raw-half prefix; explicit --run, CPU-only, no full-file hash."""
import os,argparse,json,hashlib
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args()
D=Path('/home/lxq/graduate/data/multivector-colbertv2');src=D/'lotte-pooled-dev/documents.npy';dst=D/'chimera-inputs/lotte-prefix-10000/documents-prefix-qualification.npy';n=1087049
if not a.run:print(json.dumps(dict(source=str(src),destination=str(dst),tokens=n,executed=False)));raise SystemExit
if dst.exists():raise FileExistsError(dst)
with open(src,'rb') as f:
 version=np.lib.format.read_magic(f);reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
 shape,fortran,dtype=reader(f);offset=f.tell();f.seek(0);header=f.read(offset)
assert shape[0]>=n and shape[1]==128 and not fortran and dtype==np.dtype('<u2')
raw=np.memmap(src,mode='r',dtype=dtype,offset=offset,shape=shape);tmp=dst.with_suffix('.npy.partial')
if tmp.exists():raise FileExistsError(tmp)
out=np.lib.format.open_memmap(tmp,mode='w+',dtype=dtype,shape=(n,128));h=hashlib.sha256()
for start in range(0,n,32768):
 block=raw[start:min(start+32768,n)];out[start:start+len(block)]=block;h.update(memoryview(block).cast('B'))
out.flush();del out
with open(tmp,'rb') as f:payload=hashlib.file_digest(f,'sha256').hexdigest()
os.link(tmp,dst);tmp.unlink()
receipt=dict(source=str(src),source_revision='e3745bd22a3be48d167c438fc8a7a839b86860b1',source_header_sha256=hashlib.sha256(header).hexdigest(),source_data_prefix_sha256=h.hexdigest(),rows=n,dimension=128,dtype='<u2',output=str(dst),output_sha256=payload,bit_patterns_preserved=True)
dst.with_suffix('.provenance.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
