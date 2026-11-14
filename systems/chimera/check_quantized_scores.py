"""Independent NumPy FP64 MaxSim scoring of on-disk 4-bit codes, no GPU."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from pathlib import Path
import struct,csv,json,sys
D=128
index=Path(sys.argv[1]);queries=Path(sys.argv[2]);tsv=Path(sys.argv[3])
with open(index/'doc_1bit.bin','rb') as f:
 n,d,c,ex,pad=struct.unpack('<5Q',f.read(40));assert (d,ex,pad)==(128,3,128)
 flips=np.frombuffer(f.read(4*D//8),dtype=np.uint8).copy().reshape(4,D//8)
factors=np.memmap(index/'doc_1bit.bin',mode='r',dtype='<f4',offset=40+4*D//8+n*(D//8)+n*4,shape=(n,))
codes=np.memmap(index/'doc_full.bin',mode='r',dtype=np.uint8,shape=(n,D//2))
lens=np.fromfile(index/'doclens.bin',dtype='<i4')[1:];offsets=np.r_[0,np.cumsum(lens)]
with open(queries,'rb') as f: nq,ql,qd=struct.unpack('<3i',f.read(12))
assert (ql,qd)==(32,D)
q=np.memmap(queries,mode='r',dtype='<f4',offset=12,shape=(nq,32,D))
def rotate(x):
 x=x.copy()
 for flip in flips:
  sign=1-2*np.unpackbits(flip,bitorder='little').astype(np.int8)
  x*=sign
  step=1
  while step<D:
   v=x.reshape(32,-1,2*step);a=v[:,:,:step].copy();b=v[:,:,step:].copy();v[:,:,:step]=a+b;v[:,:,step:]=a-b;step*=2
  x*=np.float32(1/np.sqrt(np.float32(D)))
 return x
selected=[]
with open(tsv) as f:
 for r in csv.DictReader(f,delimiter='\t'):
  if int(r['query_id']) in (0,37,2930) and int(r['rank']) in (1,10,100):selected.append(r)
assert len(selected)==9
out=[]
for r in selected:
 qi=int(r['query_id']);doc=int(r['document_id']);qr=rotate(q[qi]).astype(np.float64)
 v=codes[offsets[doc]:offsets[doc+1]].reshape(-1,D//16,8)
 dec=np.concatenate((v&15,v>>4),axis=2).reshape(-1,D).astype(np.float64)
 scores=(qr@dec.T-qr.sum(axis=1)[:,None]*7.5)*factors[offsets[doc]:offsets[doc+1]].astype(np.float64)[None,:]
 ref=float(scores.max(axis=1).sum());err=abs(ref-float(r['quantized_score']))
 out.append(dict(query=qi,rank=int(r['rank']),doc=doc,reference_score=ref,returned_score=float(r['quantized_score']),abs_error=err))
assert max(v['abs_error'] for v in out)<3e-4,out
print(json.dumps(dict(passed=True,checks=len(out),max_abs_error=max(v['abs_error'] for v in out),results=out),indent=2))
