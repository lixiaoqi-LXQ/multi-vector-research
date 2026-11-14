import os,sys,json,time,hashlib,subprocess,fcntl,threading,csv,statistics,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
DATA=Path('/home/lxq/graduate/data/multivector-colbertv2')
INP=DATA/'chimera-inputs/lotte-prefix-10000'
GT=DATA/'groundtruth/lotte-prefix-10000/ids.npy'
INDEX=DATA/'indexes/chimera/lotte-prefix-10000-c8192-b4'
OUT=ROOT/'runs'/('lotte-prefix10000-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate(path):
 ids=np.full((2931,100),-1,dtype=np.int64);scores=np.full((2931,100),np.nan);lat=np.full(2931,np.nan);counts=np.zeros(2931,dtype=int)
 with open(path) as f:
  for row in csv.DictReader(f,delimiter='\t'):
   q=int(row['query_id']);r=int(row['rank'])-1;d=int(row['document_id']);v=float(row['quantized_score']);t=float(row['latency_ms'])
   assert 0<=q<2931 and 0<=r<100 and 0<=d<10000
   assert ids[q,r]==-1 and np.isfinite(v) and np.isfinite(t) and t>0
   if counts[q]:assert lat[q]==t
   ids[q,r]=d;scores[q,r]=v;lat[q]=t;counts[q]+=1
 assert np.all(counts==100) and all(len(set(x))==100 for x in ids)
 assert np.all(np.diff(scores,axis=1)<=0)
 gt=np.load(GT); assert gt.shape[0]==2931 and gt.shape[1]>=100
 r100=np.mean([len(set(a)&set(b[:100]))/100 for a,b in zip(ids,gt)])
 r10=np.mean([len(set(a[:10])&set(b[:10]))/10 for a,b in zip(ids,gt)])
 return dict(queries=2931,returned=293100,recall100=float(r100),recall10=float(r10),qps=float(1000/lat.mean()),mean_ms=float(lat.mean()),p50_ms=float(np.percentile(lat,50)),p99_ms=float(np.percentile(lat,99)),output_sha256=digest(path),ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
def run(cmd,env,name,timeout=1800):
 (OUT/(name+'-command.json')).write_text(json.dumps(cmd,indent=2));t=time.time()
 with open(OUT/(name+'.log'),'w') as f:
  p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 result=dict(returncode=p.returncode,wall_seconds=time.time()-t)
 (OUT/(name+'-process.json')).write_text(json.dumps(result,indent=2))
 if p.returncode:raise RuntimeError(name+' failed; '+str(OUT/(name+'.log')))
 return result
if __name__=='__main__':
 if sys.argv[1:] and sys.argv[1]=='--validate': print(json.dumps(validate(sys.argv[2]),indent=2));sys.exit()
 assert sys.argv[1:]==['--run'], 'explicit --run required'
 OUT.mkdir(parents=True)
 env=dict(os.environ,LD_LIBRARY_PATH='/home/lxq/miniconda3/envs/cuvs_benchmarks/lib:/usr/local/cuda-13.0/lib64',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0')
 identity=dict(build_sha256=digest(ROOT/'build/gpu_build'),search_sha256=digest(ROOT/'build/gpu_search'),inputs=json.loads((INP/'manifest.json').read_text()),omp_threads_search=1,omp_threads_build=8,config=dict(nclusters=8192,ex_bits=3,k=100,nprobe=128,k_refine=3000,k_full_bit=300,cagra_itopk_size=150,num_chunks=4),scope='prefix pipeline validation; not paper full corpus',capture='top-k score stores timed; score snapshot and TSV writing outside search timer')
 (OUT/'identity.json').write_text(json.dumps(identity,indent=2))
 lock=open('/tmp/uma-paper-device0.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 stop=threading.Event()
 try:
  # Require ten consecutive seconds of device compute idle before starting.
  t0=time.monotonic()
  while time.monotonic()-t0<10:
   p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
   apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True)
   if int(p.stdout.strip())!=0 or apps.stdout.strip():raise RuntimeError('GPU not idle: '+p.stdout+apps.stdout)
   time.sleep(1)
  def monitor():
   with open(OUT/'device-monitor.jsonl','w') as f:
    while not stop.is_set():
     util=subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,power.draw','--format=csv,noheader'],capture_output=True,text=True)
     apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True)
     f.write(json.dumps(dict(timestamp=time.time(),gpu=util.stdout.strip(),apps=apps.stdout.strip()))+'\n');f.flush();stop.wait(1)
  th=threading.Thread(target=monitor);th.start()
  INDEX.parent.mkdir(parents=True,exist_ok=True)
  if not (INDEX/'doclens.bin').exists():
   cmd=[str(ROOT/'build/gpu_build'),'--index',str(INDEX),'--doclens',str(INP/'doclens.bin'),'--data',str(INP/'embeddings.bin'),'--n-clusters','8192','--ex-bits','3']
   run(cmd,dict(env,OMP_NUM_THREADS='8'),'build')
  results=[]
  for i in range(3):
   file=OUT/f'round{i}.tsv'
   cmd=[str(ROOT/'build/gpu_search'),'--query',str(INP/'queries.bin'),'--gt',str(GT.parent/'groundtruth.tsv'),'--index',str(INDEX),'--k','100','--nprobe','128','--k-refine','3000','--k-full-bit','300','--cagra-itopk-size','150','--num-chunks','4','--warmup','5','--output',str(file)]
   run(cmd,env,f'round{i}');v=validate(file);results.append(v);(OUT/f'round{i}-validation.json').write_text(json.dumps(v,indent=2));print(json.dumps(dict(round=i,**v)),flush=True)
  (OUT/'summary.json').write_text(json.dumps(dict(identity=identity,rounds=results,qps_mean=statistics.mean(x['qps'] for x in results),qps_range=[min(x['qps'] for x in results),max(x['qps'] for x in results)]),indent=2))
 finally:
  stop.set()
  if 'th' in globals():th.join(timeout=10)
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 print(str(OUT),flush=True)
