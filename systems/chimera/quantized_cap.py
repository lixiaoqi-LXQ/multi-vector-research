"""Exhaustive 4-bit MaxSim quality diagnostic on all prefix documents."""
import os,sys,json,struct,fcntl,time,hashlib
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import torch
sys.path.insert(0,'/home/lxq/graduate/ai-workspace/research/multivector/scripts')
from segmented_maxsim import maxsim
ROOT=Path(__file__).resolve().parent
DATA=Path('/home/lxq/graduate/data/multivector-colbertv2')
INDEX=DATA/'indexes/chimera/lotte-prefix-10000-c8192-b4'
QUERY=DATA/'splits/lotte-calibration-seed42/queries.bin'
GT=DATA/'groundtruth/lotte-prefix-10000/calibration/ids.npy'
OUT=ROOT/'runs/quantized-cap4-cal100-prefix10000'
def rotate(x,flips):
 x=x.copy()
 for flip in flips:
  x*=1-2*np.unpackbits(flip,bitorder='little').astype(np.int8)
  step=1
  while step<128:
   v=x.reshape(-1,128//(2*step),2*step);a=v[:,:,:step].copy();b=v[:,:,step:].copy();v[:,:,:step]=a+b;v[:,:,step:]=a-b;step*=2
  x*=np.float32(1/np.sqrt(np.float32(128)))
 return x
assert sys.argv[1:]==['--run']
OUT.mkdir(parents=True,exist_ok=True)
lock=open('/tmp/uma-paper-device0.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 for _ in range(10):
  util=int(__import__('subprocess').check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip())
  if util:raise RuntimeError('GPU nonidle')
  time.sleep(1)
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 with open(INDEX/'doc_1bit.bin','rb') as f:
  n,d,c,ex,pad=struct.unpack('<5Q',f.read(40));assert(d,ex,pad)==(128,3,128);flips=np.frombuffer(f.read(64),dtype=np.uint8).copy().reshape(4,16)
 lens=np.fromfile(INDEX/'doclens.bin',dtype='<i4')[1:];offsets=np.r_[0,np.cumsum(lens)]
 factors=np.memmap(INDEX/'doc_1bit.bin',mode='r',dtype='<f4',offset=104+n*20,shape=(n,))
 codes=np.memmap(INDEX/'doc_full.bin',mode='r',dtype=np.uint8,shape=(n,64))
 with open(QUERY,'rb') as f:nq,ql,qd=struct.unpack('<3i',f.read(12));assert(ql,qd)==(32,128)
 q=np.memmap(QUERY,mode='r',dtype='<f4',offset=12,shape=(nq,32,128))
 rot=rotate(q,flips);qt=torch.from_numpy(rot).cuda();cb=torch.from_numpy(rot.sum(axis=2,dtype=np.float32)*np.float32(7.5)).cuda()
 all_scores=torch.full((nq,len(lens)),-torch.inf,device='cuda',dtype=torch.float32)
 start=time.monotonic()
 for ds in range(0,len(lens),512):
  de=min(ds+512,len(lens));ts,te=offsets[ds],offsets[de];v=codes[ts:te].reshape(-1,8,8);dec=np.concatenate((v&15,v>>4),axis=2).reshape(-1,128).astype(np.float32)
  dt=torch.from_numpy(dec).cuda();fac=torch.from_numpy(np.array(factors[ts:te])).cuda();off=torch.from_numpy(offsets[ds:de+1]-ts).cuda()
  for qs in range(0,nq,8):
   qe=min(qs+8,nq);sim=qt[qs:qe].reshape(-1,128)@dt.T;sim-=cb[qs:qe].reshape(-1,1);sim*=fac[None,:]
   all_scores[qs:qe,ds:de]=maxsim(sim,off,qe-qs,int(lens[ds:de].max()))
  del dt,fac,off
 torch.cuda.synchronize();values,ids=torch.topk(all_scores,100,dim=1);values=values.cpu().numpy();ids=ids.cpu().numpy()
 gt=np.load(GT);recall=float(np.mean([len(set(a)&set(b))/100 for a,b in zip(ids,gt)]))
 np.save(OUT/'ids.npy',ids);np.save(OUT/'scores.npy',values)
 result=dict(queries=nq,docs=len(lens),tokens=n,recall100_vs_unquantized_FP32_GT=recall,elapsed_s=time.monotonic()-start,scope='exhaustive all-prefix-document quantized MaxSim quality diagnostic, no QPS claim',arithmetic='GPU FP32 GEMM TF32 disabled; then subtract7.5*sum(rotated_q), multiply per-token full-bit factor; segmented max and FP32 sum',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),complete=True)
 (OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
