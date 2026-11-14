"""Bounded GPU validation/100k timing; canonical GT and existing runners are read-only."""
import os,sys,argparse,json,time,struct,hashlib,fcntl,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for name,sub in [('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
 p=ROOT/'cache'/sub;p.mkdir(parents=True,exist_ok=True);os.environ[name]=str(p)
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import torch
from fused_ieee import score_block,resident_topk,merge_topk
sys.path.insert(0,'/home/lxq/graduate/ai-workspace/research/multivector/scripts')
from exact_gt import load_prefix
from segmented_maxsim import maxsim
DATA=Path('/home/lxq/graduate/data/multivector-colbertv2/lotte-pooled-dev')
GT=Path('/home/lxq/graduate/data/multivector-colbertv2/groundtruth/lotte-prefix-10000')

def need(x,msg):
 if not x:raise RuntimeError(msg)
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(out,name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
def load_resident(data,docs):
 lens,raw=load_prefix(data,docs)
 try:x=np.array(raw[:len(raw)],dtype=np.float32)
 finally:raw.close()
 offsets=np.r_[0,np.cumsum(lens,dtype=np.int64)]
 need(len(x)==offsets[-1] and np.all(lens>0),'Invalid packed document boundaries')
 return lens,x,torch.from_numpy(x).cuda(),torch.from_numpy(offsets).cuda()
def event_time(fn):
 a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
 torch.cuda.synchronize();start=time.monotonic();a.record();result=fn();b.record();b.synchronize()
 return result,dict(cuda_event_s=a.elapsed_time(b)/1000,host_wall_s=time.monotonic()-start)
def canonical_resident(q,d,off_cpu,k=100,query_batch=32,token_block=131072):
 nq=len(q);nd=len(off_cpu)-1;v=torch.full((nq,k),-torch.inf,device='cuda',dtype=torch.float32);ids=torch.full((nq,k),-1,device='cuda',dtype=torch.long);ds=0
 while ds<nd:
  target=off_cpu[ds]+token_block;de=min(nd,max(ds+1,int(np.searchsorted(off_cpu,target,side='right')-1)));start,end=int(off_cpu[ds]),int(off_cpu[de])
  doc=d[start:end];offsets=torch.from_numpy(off_cpu[ds:de+1]-start).cuda()
  maxlen=int(np.diff(off_cpu[ds:de+1]).max())
  for qs in range(0,nq,query_batch):
   qe=min(qs+query_batch,nq);sim=q[qs:qe].reshape(-1,128)@doc.T
   scores=maxsim(sim,offsets,qe-qs,maxlen)
   nv,ni=merge_topk(v[qs:qe],ids[qs:qe],scores,ds,k);v[qs:qe]=nv;ids[qs:qe]=ni
  ds=de
 return v,ids

def compare(ids,scores,ref_ids,ref_scores):
 need(ids.shape==ref_ids.shape==scores.shape==ref_scores.shape,'Top-k shape mismatch')
 need(np.all(np.isfinite(scores)) and np.all(np.diff(scores,axis=1)<=0),'Invalid fused scores')
 r100=np.asarray([len(set(a)&set(b))/100 for a,b in zip(ids,ref_ids)]);r10=np.asarray([len(set(a[:10])&set(b[:10]))/10 for a,b in zip(ids,ref_ids)])
 return dict(queries=len(ids),top100_mean_set_overlap=float(r100.mean()),top10_mean_set_overlap=float(r10.mean()),top100_queries_equal=int(np.sum(r100==1)),top10_queries_equal=int(np.sum(r10==1)),max_sorted_score_abs_difference=float(np.max(np.abs(scores-ref_scores))),strict_top100_all_equal=bool(np.all(r100==1)),strict_top10_all_equal=bool(np.all(r10==1)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--out',required=True);p.add_argument('--budget-s',type=float,default=110);p.add_argument('--benchmark-queries',type=int,default=128);p.add_argument('--data',default=str(DATA));a=p.parse_args();need(a.run,'Explicit --run and parent GPU authorization required')
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);data=Path(a.data);begun=time.monotonic();lock=open('/tmp/uma-paper-device0.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);finished=False
 identity=dict(scope='Independent FP32 IEEE GT-tool prototype, not retrieval-system QPS',kernel_sha256=sha(ROOT/'fused_ieee.py'),validator_sha256=sha(__file__),baseline_reducer_sha256=sha(Path('/home/lxq/graduate/ai-workspace/research/multivector/scripts/segmented_maxsim.py')),torch=torch.__version__,cuda=torch.version.cuda,query_source_sha256=sha(data/'queries.npy'),canonical_manifest=json.loads((GT/'manifest.json').read_text()),budget_s=a.budget_s,arithmetic='FP32 inputs; tl.dot(input_precision=ieee,out_dtype=float32); FP32 max/sum; TF32 and FP16 arithmetic prohibited')
 dump(out,'identity.json',identity)
 def time_left(minimum=10):return a.budget_s-(time.monotonic()-begun)>minimum
 try:
  # Idle gate is inside the shared lock; this is a correctness/speed-tool window.
  import subprocess
  for _ in range(10):
   v=int(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip());need(v==0,'GPU nonidle');time.sleep(1)
  torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
  q_np=np.load(data/'queries.npy',allow_pickle=False);need(q_np.shape==(2931,32,128) and q_np.dtype==np.float32,'Expected original LoTTE FP32 queries')
  need(hashlib.sha256(q_np.tobytes()).hexdigest()==identity['canonical_manifest']['query_sha256'],'Canonical query identity mismatch')
  qt=torch.from_numpy(q_np.copy()).cuda()
  # Same 7 queries / 128 documents used by the existing independent FP64 check.
  lens,x,d,off=load_resident(data,128);offsets=np.r_[0,np.cumsum(lens)]
  ref=np.asarray([[np.max(qq.astype(np.float64)@x[offsets[i]:offsets[i+1]].astype(np.float64).T,axis=1).sum() for i in range(128)] for qq in q_np[:7]],dtype=np.float64)
  for m,warps in [(64,8),(32,4)]:
   got,kernel=score_block(qt[:7],d,off,m=m,warps=warps,return_kernel=True);torch.cuda.synchronize();gn=got.cpu().numpy();ptx=kernel.asm['ptx'];(out/f'ieee-m{m}.ptx').write_text(ptx)
   forbidden=bool(re.search(r'(?i)(?:mma|wgmma|tcgen|cvt)[^\n]*(?:tf32|f16)',ptx));need(not forbidden,'Non-IEEE arithmetic found in compiled PTX')
   error=float(np.max(np.abs(gn-ref)));equal=all(set(np.argsort(-gn[i])[:10])==set(np.argsort(-ref[i])[:10]) for i in range(7));need(error<1e-4 and equal,'FP64 scorer validation failed')
   dump(out,f'fp64-m{m}.json',dict(max_abs_error=error,top10_sets_equal=equal,queries=7,documents=128,ptx_forbidden_precision_opcodes=forbidden,ptx_has_fp32_fma='fma.rn.f32' in ptx))
  del d,x,off;torch.cuda.empty_cache()
  if not time_left(25):dump(out,'stopped.json',dict(reason='budget before complete10k validation'));return
  lens,x,d,off=load_resident(data,10000);expected_ids=np.load(GT/'ids.npy');expected_scores=np.load(GT/'scores.npy');offsets=np.r_[0,np.cumsum(lens)]
  comparisons=[]
  for m,warps in [(64,8),(32,4)]:
   if not time_left(20):break
   (values,ids),elapsed=event_time(lambda:resident_topk(qt,d,off,k=100,query_batch=64,doc_chunk=2048,m=m,warps=warps));vn=values.cpu().numpy();ix=ids.cpu().numpy();check=compare(ix,vn,expected_ids,expected_scores);check.update(m=m,warps=warps,**elapsed);comparisons.append(check);np.save(out/f'10k-m{m}-ids.npy',ix);np.save(out/f'10k-m{m}-scores.npy',vn);dump(out,f'10k-m{m}-validation.json',check);print(json.dumps(dict(phase='10k_complete',**check)),flush=True)
  passing=[x for x in comparisons if x['strict_top100_all_equal'] and x['strict_top10_all_equal'] and x['max_sorted_score_abs_difference']<1e-4]
  if not passing:dump(out,'decision.json',dict(adopt=False,reason='strict full10k top-k/score equivalence not established',observations=comparisons));return
  chosen=min(passing,key=lambda x:x['cuda_event_s']);del d,x,off;torch.cuda.empty_cache()
  if not time_left(20):dump(out,'decision.json',dict(adopt=False,reason='budget before100k paired timing',chosen=chosen));return
  lens,x,d,off=load_resident(data,100000);offsets=np.r_[0,np.cumsum(lens)];nq=min(a.benchmark_queries,len(qt));qb=qt[:nq];m=chosen['m'];warps=chosen['warps']
  fused=lambda:resident_topk(qb,d,off,k=100,query_batch=32,doc_chunk=1024,m=m,warps=warps)
  canonical=lambda:canonical_resident(qb,d,offsets,k=100,query_batch=32,token_block=131072)
  # Compile/warm both paths before comparing the same resident FP32 inputs.
  fv,fi=fused();cv,ci=canonical();torch.cuda.synchronize();parity=compare(fi.cpu().numpy(),fv.cpu().numpy(),ci.cpu().numpy(),cv.cpu().numpy());dump(out,'100k-parity.json',parity)
  samples=[]
  for rnd in range(3):
   if not time_left(8):break
   order=[('canonical',canonical),('fused_ieee',fused)] if rnd%2==0 else [('fused_ieee',fused),('canonical',canonical)]
   for name,fn in order:
    _,elapsed=event_time(fn);samples.append(dict(round=rnd,path=name,**elapsed));dump(out,'100k-timing.json',dict(queries=nq,docs=100000,samples=samples,scope='resident compute comparison; transfer/input validation excluded; no system QPS claim'))
  c=[s['cuda_event_s'] for s in samples if s['path']=='canonical'];f=[s['cuda_event_s'] for s in samples if s['path']=='fused_ieee'];need(len(c)==len(f)>0,'No complete paired timing')
  ratio=float(np.mean(c)/np.mean(f));adopt=ratio>1.05 and parity['strict_top100_all_equal'] and parity['strict_top10_all_equal'] and len(c)==3
  dump(out,'decision.json',dict(adopt_candidate=adopt,canonical_over_fused_compute_ratio=ratio,paired_rounds=len(c),chosen=chosen,canonical_parity=parity,reason='Parent must decide any canonical integration. If slower, retain canonical GT and stop prototype optimization.',elapsed_s=time.monotonic()-begun));finished=True
  print(json.dumps(dict(phase='finished',adopt_candidate=adopt,canonical_over_fused_compute_ratio=ratio,paired_rounds=len(c))),flush=True)
 finally:
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close();dump(out,'release.json',dict(flock_released=True,finished_benchmark=finished,elapsed_s=time.monotonic()-begun,canonical_files_modified=False))
if __name__=='__main__':main()
