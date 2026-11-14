"""Run an explicitly supplied search-only Chimera grid, serially under GPU lock."""
import os,sys,json,time,hashlib,subprocess,fcntl,csv,argparse,struct,statistics,re
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
ROOT=Path(__file__).resolve().parent

def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def index_identity(index):
 index=Path(index)
 with open(index/'doc_1bit.bin','rb') as f:n,d,c,ex,pad=struct.unpack('<5Q',f.read(40))
 files={}
 for name in ('centroids.carga','cluster_1bit.bin','doc_1bit.bin','doc_full.bin','doclens.bin','ivf.bin'):
  p=index/name;st=p.stat();files[name]=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,device=st.st_dev,sha256=sha(p))
 return dict(N=n,D=d,paddedD=pad,n_clusters=c,ex_bits=ex,total_bits=ex+1,files=files)
def check_index_unchanged(index,identity):
 for name,v in identity['files'].items():
  st=(Path(index)/name).stat();assert (st.st_size,st.st_mtime_ns,st.st_ino,st.st_dev)==(v['size'],v['mtime_ns'],v['inode'],v['device']),'Index file changed: '+name

def check(path,gt_path,nq,nd,k=100,heldout_path=None):
 ids=np.full((nq,k),-1,dtype=np.int64);scores=np.full((nq,k),np.nan);lat=np.full(nq,np.nan);counts=np.zeros(nq,dtype=int)
 with open(path) as f:
  for r in csv.DictReader(f,delimiter='\t'):
   q=int(r['query_id']);j=int(r['rank'])-1;doc=int(r['document_id']);s=float(r['quantized_score']);t=float(r['latency_ms'])
   assert 0<=q<nq and 0<=j<k and 0<=doc<nd and ids[q,j]==-1
   assert np.isfinite(s) and np.isfinite(t) and t>0
   if counts[q]:assert lat[q]==t
   ids[q,j]=doc;scores[q,j]=s;lat[q]=t;counts[q]+=1
 assert np.all(counts==k) and all(len(set(a))==k for a in ids)
 assert np.all(np.diff(scores,axis=1)<=0)
 gt=np.load(gt_path);assert gt.shape[0]>=nq and gt.shape[1]>=k
 result=dict(queries=nq,returned=nq*k,recall100=float(np.mean([len(set(a)&set(b[:100]))/100 for a,b in zip(ids,gt[:nq])])),recall10=float(np.mean([len(set(a[:10])&set(b[:10]))/10 for a,b in zip(ids,gt[:nq])])),qps=float(1000/lat.mean()),mean_ms=float(lat.mean()),p50_ms=float(np.percentile(lat,50)),p99_ms=float(np.percentile(lat,99)),ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),output_sha256=sha(path))

 if heldout_path:
  rows=np.load(heldout_path);assert rows.ndim==1 and len(np.unique(rows))==len(rows) and np.all((rows>=0)&(rows<nq))
  result['heldout_queries']=int(len(rows));result['heldout_recall100']=float(np.mean([len(set(ids[i])&set(gt[i,:100]))/100 for i in rows]));result['heldout_recall10']=float(np.mean([len(set(ids[i,:10])&set(gt[i,:10]))/10 for i in rows]))
  cal=np.setdiff1d(np.arange(nq),rows);result['calibration_recall100']=float(np.mean([len(set(ids[i])&set(gt[i,:100]))/100 for i in cal]))
 return result

def main():
 a=argparse.ArgumentParser();a.add_argument('--spec',required=True);a.add_argument('--output',required=True);a.add_argument('--query');a.add_argument('--gt');a.add_argument('--gt-ids');a.add_argument('--run',action='store_true');args=a.parse_args()
 spec=json.loads(Path(args.spec).read_text())
 limit=int(re.search(r'kCandidateRefinementMaxNprobe\s*=\s*(\d+)',(ROOT/'arm-source/cpp/include/chimera/config.cuh').read_text()).group(1))
 if any(point['nprobe']>limit for point in spec['points']):raise ValueError(f'Chimera candidate refinement supports nprobe <= {limit}')
 for key,val in [('query',args.query),('gt',args.gt),('gt_ids',args.gt_ids)]:
  if val:spec[key]=val
 with open(spec['query'],'rb') as f:nq=struct.unpack('<i',f.read(4))[0]
 if spec.get('nq',-1)>0:nq=min(nq,spec['nq'])
 nd=struct.unpack('<i',(Path(spec['index'])/'doclens.bin').read_bytes()[:4])[0]
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 binary=Path(spec.get('binary',ROOT/'build/gpu_search'))
 index_meta=index_identity(spec['index']);(out/'index-identity.json').write_text(json.dumps(index_meta,indent=2))
 gtmanifest=Path(spec['gt_ids']).parent/'manifest.json'
 if not gtmanifest.exists():gtmanifest=gtmanifest.parent.parent/'manifest.json'
 gtmeta=json.loads(gtmanifest.read_text()) if gtmanifest.exists() else None
 splitmanifest=Path(spec['query']).parent/'manifest.json'
 splitmeta=json.loads(splitmanifest.read_text()) if splitmanifest.exists() else None
 base=['taskset','-c',str(spec.get('cpu_core',7)),str(binary),'--index',spec['index'],'--query',spec['query'],'--gt',spec['gt'],'--k','100','--nq',str(nq),'--warmup',str(spec.get('warmup',5))]
 commands=[]
 for pointidx,point in enumerate(spec['points']):
  for rnd in range(spec.get('rounds',1)):
   name=f'point{pointidx:02d}-round{rnd}'
   cmd=base+['--nprobe',str(point['nprobe']),'--k-refine',str(point['k_refine']),'--k-full-bit',str(point['k_full_bit']),'--cagra-itopk-size',str(point.get('cagra_itopk_size',point['nprobe']+32)),'--num-chunks',str(point['num_chunks']),'--output',str(out/(name+'.tsv'))]
   commands.append((name,point,cmd))
 for fname in ('arm-port.patch','arm-port-manifest.json'):
  if (ROOT/fname).exists():(out/fname).write_bytes((ROOT/fname).read_bytes())
 (out/'prepared.json').write_text(json.dumps(dict(spec=spec,queries=nq,docs=nd,index_identity=index_meta,GT_manifest=gtmeta,GT_manifest_path=str(gtmanifest),GT_manifest_sha256=sha(gtmanifest) if gtmanifest.exists() else None,query_rows_manifest=splitmeta,query_rows_manifest_sha256=sha(splitmanifest) if splitmanifest.exists() else None,heldout_rows_sha256=sha(spec['heldout_rows']) if spec.get('heldout_rows') else None,binary_sha256=sha(binary),spec_sha256=sha(args.spec),gt_ids_sha256=sha(spec['gt_ids']),query_sha256=sha(spec['query']),runner_sha256=sha(__file__),commands=[x[2] for x in commands],capture_scores=True,CPU_affinity=spec.get('cpu_core',7),boundary='sum of synchronous search calls; encoding/index loading/warmup/disk output excluded; loop_qps includes score snapshot'),indent=2))
 if not args.run:print('Prepared only: '+str(out));return
 env=dict(os.environ,LD_LIBRARY_PATH='/home/lxq/miniconda3/envs/cuvs_benchmarks/lib:/usr/local/cuda-13.0/lib64',OMP_NUM_THREADS=str(spec.get('threads',1)),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0')
 lock=open('/tmp/uma-paper-device0.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  for _ in range(10):
   util=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True)
   if int(util.strip())!=0:raise RuntimeError('GPU compute nonidle before run')
   time.sleep(1)
  results=[]
  for name,point,cmd in commands:
   if (out/(name+'.tsv')).exists():raise FileExistsError(name+' output already exists')
   start=time.monotonic()
   with open(out/(name+'.log'),'w') as f:
    p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    try:
     with open(out/(name+'-monitor.jsonl'),'w') as mon:
      while p.poll() is None:
       util=subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True)
       apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True)
       mon.write(json.dumps(dict(time=time.time(),pid=p.pid,gpu=util.stdout.strip(),apps=apps.stdout.strip()))+'\n');mon.flush()
       if time.monotonic()-start>spec.get('timeout_per_point_s',1800):raise TimeoutError(name)
       try:p.wait(timeout=1)
       except subprocess.TimeoutExpired:pass
    except BaseException:
     if p.poll() is None:
      p.terminate()
      try:p.wait(timeout=20)
      except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
     raise
   assert p.returncode==0,name+' subprocess failed'
   check_index_unchanged(spec['index'],index_meta)
   val=check(out/(name+'.tsv'),spec['gt_ids'],nq,nd,heldout_path=spec.get('heldout_rows'));val.update(point=point,round_name=name,process_wall_seconds=time.monotonic()-start)
   results.append(val);(out/(name+'-validation.json')).write_text(json.dumps(val,indent=2));(out/'summary.json').write_text(json.dumps(results,indent=2));print(json.dumps(val),flush=True)
 finally:
  exited=('p' not in locals()) or p.poll() is not None
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
  (out/'release.json').write_text(json.dumps(dict(subprocesses_exited=exited,flock_released=True),indent=2))
if __name__=='__main__':main()
