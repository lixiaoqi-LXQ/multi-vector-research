"""Serial Chimera search grid with frozen index/GT identity and explicit CPU budget."""
import os,sys,json,time,hashlib,subprocess,fcntl,csv,argparse,struct,re,shutil
from pathlib import Path
sys.path.insert(0, "/home/lxq/graduate/ai-workspace/research/multivector/chimera")
from search_supervision_v2 import supervise_child
from search_modes_v2 import modes,selection_family,verify_mode_log,validate_points
from routing_audit import reject_quarantined,verify_routing_dataset
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
ROOT=Path("/home/lxq/graduate/ai-workspace/research/multivector/chimera")

def need(condition,message):
 if not condition:raise ValueError(message)
def pagefault_evidence(log,low_memory=False):
 content=Path(log).read_text()
 rows=re.findall(r'^\[PAGEFAULTS\] scope=query_loop (.+)$',content,re.M)
 need(len(rows)==1,'Missing/duplicate query-loop pagefault receipt')
 values={key:int(value) for key,value in re.findall(r'(\w+)=(\d+)',rows[0])}
 for kind in ('major','minor'):
  need(all(kind+'_'+suffix in values for suffix in ('before','after','delta')),'Incomplete pagefault counters')
  need(values[kind+'_after']-values[kind+'_before']==values[kind+'_delta'],'Inconsistent pagefault delta')
 prefault=re.findall(r'^\[PREFAULT\] bytes=(\d+) elapsed_ms=([0-9.eE+-]+)$',content,re.M)
 need(len(prefault)==int(low_memory),'Unexpected/missing prefault receipt')
 return dict(query_loop=values,prefault=[dict(bytes=int(b),elapsed_ms=float(t)) for b,t in prefault],major_fault_investigation_required=values['major_delta']>0)

def sha(path):
 with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def index_identity(index):
 index=Path(index)
 cache=index/'artifact-manifest.json'
 if cache.exists():
  cached=json.loads(cache.read_text())
  try:
   check_index_unchanged(index,cached)
   with open(index/'doc_1bit.bin','rb') as f:header=struct.unpack('<5Q',f.read(40))
   need(header==(cached['N'],cached['D'],cached['n_clusters'],cached['ex_bits'],cached['paddedD']),'Cached index header mismatch')
   return cached
  except (ValueError,FileNotFoundError,KeyError):pass
 with open(index/'doc_1bit.bin','rb') as f:n,d,c,ex,pad=struct.unpack('<5Q',f.read(40))
 files={}
 for name in ('centroids.carga','cluster_1bit.bin','doc_1bit.bin','doc_full.bin','doclens.bin','ivf.bin'):
  p=index/name;st=p.stat();files[name]=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,device=st.st_dev,sha256=sha(p))
 return dict(N=n,D=d,paddedD=pad,n_clusters=c,ex_bits=ex,total_bits=ex+1,files=files)
def check_index_unchanged(index,identity):
 for name,v in identity['files'].items():
  st=(Path(index)/name).stat();need((st.st_size,st.st_mtime_ns,st.st_ino,st.st_dev)==(v['size'],v['mtime_ns'],v['inode'],v['device']),'Index file changed: '+name)
def check_derived_build_provenance(spec, buildmeta, index_meta):
 operation='CPU requantize from fixed 4-bit index to 8-bit'
 derived=buildmeta.get('fresh_8bit_build') is False or buildmeta.get('operation')==operation
 if not derived:return
 need(spec.get('allow_derived_index') is True,'Derived index requires explicit allow_derived_index')
 need(buildmeta.get('operation')==operation and buildmeta.get('fresh_8bit_build') is False,'Invalid derived operation identity')
 need(buildmeta.get('requantize_tool_sha256')==spec.get('approved_requantize_tool_sha256') and isinstance(spec.get('approved_requantize_tool_sha256'),str),'Unapproved requantize tool identity')
 need(buildmeta.get('qualification_receipt_sha256')==spec.get('approved_requantize_qualification_sha256') and isinstance(spec.get('approved_requantize_qualification_sha256'),str),'Unapproved requantization qualification')
 need(buildmeta.get('index_identity')==index_meta,'Derived metadata/index identity mismatch')
 rp=Path(buildmeta['requantize_receipt']);need(sha(rp)==buildmeta['requantize_receipt_sha256'],'Derived receipt SHA mismatch')
 r=json.loads(rp.read_text());need(r['operation']==operation and r['fresh_8bit_build'] is False and r['validation']['all_rows_bitwise_source_verified'] is True and r['validation']['gpu_used'] is False,'Incomplete CPU requantization evidence')
 need(r['source_build_receipt_sha256']==buildmeta['source4_build_receipt_sha256'] and r['tool_sha256']==buildmeta['requantize_tool_sha256'],'Derived/source receipt binding mismatch')
 for name,detail in index_meta['files'].items():need(detail['sha256']==r['outputs'][name]['sha256'] and detail['size']==r['outputs'][name]['bytes'],'Derived output payload mismatch: '+name)
 qp=Path(buildmeta['qualification_receipt']);need(sha(qp)==buildmeta['qualification_receipt_sha256'],'Qualification receipt changed')
 q=json.loads(qp.read_text());need(q['status']=='PASS_CPU_BYTE_QUALIFICATION' and q['binary_sha256']==r['tool_sha256'],'Qualification/tool mismatch')

def cpu_set(text):
 cores=set()
 for p in str(text).split(','):
  if '-' in p:
   lo,hi=map(int,p.split('-'));cores.update(range(lo,hi+1))
  else:cores.add(int(p))
 return cores

FAST_FIRST=[7,5,6,8,9,15,16,17,18,19,0,1,2,3,4,10,11,12,13,14]
def omp_budget(spec,affinity,threads):
 cores=cpu_set(affinity);order=[i for i in FAST_FIRST if i in cores]+sorted(cores-set(FAST_FIRST))
 places=spec.get('omp_places',','.join('{'+str(i)+'}' for i in order))
 actual=[int(x) for x in re.findall(r'\{(\d+)\}',places)]
 need(len(actual)>=threads and len(set(actual))==len(actual) and set(actual)<=cores,'Invalid explicit OpenMP places')
 return dict(OMP_NUM_THREADS=str(threads),OMP_PLACES=places,OMP_PROC_BIND='true',OMP_DYNAMIC='FALSE',OMP_MAX_ACTIVE_LEVELS='1',OMP_DISPLAY_AFFINITY='TRUE',OMP_THREAD_LIMIT=str(threads))
def check_thread_info(log,environment):
 text=Path(log).read_text();found=re.findall(r'\[THREAD\] omp_thread=(\d+) linux_tid=(\d+) cpu=(\d+) allowed=([0-9,]+)',text)
 expected=int(environment['OMP_NUM_THREADS']);places=[int(x) for x in re.findall(r'\{(\d+)\}',environment['OMP_PLACES'])]
 need(len(found)==expected and {int(x[0]) for x in found}==set(range(expected)),'Missing OpenMP thread placement evidence')
 masks={};tids=set()
 for worker,tid,current,mask in found:
  w=int(worker);need(cpu_set(mask)=={places[w]} and int(current)==places[w],'OpenMP worker is not bound to declared place');need(int(tid) not in tids,'Repeated OpenMP Linux TID');tids.add(int(tid));masks[w]=int(current)
 return dict(verified=True,workers=expected,worker_to_cpu=masks,union=sorted(set(masks.values())),master_cpu=masks[0],note='Master singleton affinity is expected; worker group defines the budget')

def identity_checks(spec,index_meta,nq,nd):
 routing=verify_routing_dataset(spec,index_meta)
 k=int(spec.get('k',100));need(k in (10,100),'Use explicit independent k10 or k100 search')
 for key,value in spec.get('expected_index',{}).items():need(index_meta[key]==value,'Index header mismatch: '+key)
 with open(spec['query'],'rb') as f:qn,ql,qd=struct.unpack('<3i',f.read(12))
 need(ql==32 and qd==128 and nq<=qn,'Invalid query dimensions')
 need(Path(spec['query']).stat().st_size==12+qn*ql*qd*4,'Incomplete query binary')
 lens=np.fromfile(Path(spec['index'])/'doclens.bin',dtype='<i4');need(len(lens)==nd+1 and lens[0]==nd and np.all(lens[1:]>0),'Invalid index document lengths')
 need(int(lens[1:].sum(dtype=np.int64))==index_meta['N'],'Index token/document count mismatch')
 gtpath=Path(spec.get('gt_manifest',Path(spec['gt_ids']).parent/'manifest.json'))
 if not gtpath.exists():gtpath=gtpath.parent.parent/'manifest.json'
 gtmeta=json.loads(gtpath.read_text()) if gtpath.exists() else None
 splitpath=Path(spec.get('split_manifest',Path(spec['query']).parent/'manifest.json'))
 splitmeta=json.loads(splitpath.read_text()) if splitpath.exists() else None
 ids=np.load(spec['gt_ids']);need(ids.ndim==2 and ids.shape[0]==nq and ids.shape[1]>=k,'GT/query row count mismatch')
 need(np.all((ids>=0)&(ids<nd)),'GT has out-of-range IDs')
 payload=np.memmap(spec['query'],mode='r',dtype='<f4',offset=12,shape=(qn,32,128))[:nq]
 build_path=Path(spec['index'])/'build-provenance.json'
 buildmeta=json.loads(build_path.read_text()) if build_path.exists() else None
 if spec.get('require_build_identity'):
  need(buildmeta is not None and buildmeta.get('complete') is True,'Require completed index build provenance')
  source=buildmeta['source_identity']
  need(all(source[key]==gtmeta['corpus_source'][key] for key in ('repository','revision','documents_url')),'Index build source does not match GT source')
  need(buildmeta['docs']==nd and buildmeta['tokens']==index_meta['N'] and buildmeta['total_bits']==index_meta['total_bits'],'Build provenance dimensions/bitwidth mismatch')
 if buildmeta is not None:check_derived_build_provenance(spec,buildmeta,index_meta)
 if spec.get('require_identity'):
  need(gtmeta is not None and gtmeta.get('complete') is True,'Require completed GT manifest')
  need(int(gtmeta['docs'])==nd and int(gtmeta['num_tokens'])==index_meta['N'],'GT corpus does not match index')
  need(gtmeta['doclens_sha256']==array_sha(lens[1:].astype(np.int64)),'GT/index document order/length mismatch')
  full_gt=Path(spec.get('full_gt_ids',gtpath.parent/'ids.npy'))
  need(sha(full_gt)==gtmeta['artifact_sha256']['ids.npy'],'GT ids artifact SHA mismatch')
  if spec.get('query_rows'):
   rows=np.load(spec['query_rows']);need(rows.ndim==1 and len(rows)==nq and len(np.unique(rows))==nq,'Invalid calibration query rows')
   full_ids=np.load(full_gt);need(np.all((rows>=0)&(rows<len(full_ids))),'Calibration rows out of range')
   need(np.array_equal(ids,full_ids[rows]),'Calibration GT does not match frozen source rows')
   need(splitmeta is not None and array_sha(rows)==splitmeta['calibration_rows_sha256'],'Calibration split row identity mismatch')
   need(array_sha(payload)==splitmeta['calibration_query_sha256'],'Calibration query payload identity mismatch')
  else:need(array_sha(payload)==gtmeta['query_sha256'],'Full query payload/order mismatch')
 if spec.get('heldout_rows'):
  held=np.load(spec['heldout_rows']);expected_heldout=int(spec.get('expected_queries',2931))-int(spec.get('calibration_queries',100));need(len(held)==expected_heldout and len(np.unique(held))==expected_heldout and np.all((held>=0)&(held<nq)),'Heldout rows do not match declared dataset split')
  if spec.get('require_identity'):need(array_sha(held)==splitmeta['heldout_rows_sha256'],'Heldout row identity mismatch')
 return dict(k=k,routing_dataset_validation=routing,index_build_provenance=buildmeta,index_build_provenance_sha256=sha(build_path) if build_path.exists() else None,GT_manifest=gtmeta,GT_manifest_path=str(gtpath),GT_manifest_sha256=sha(gtpath) if gtpath.exists() else None,query_rows_manifest=splitmeta,query_rows_manifest_sha256=sha(splitpath) if splitpath.exists() else None,gt_ids_sha256=sha(spec['gt_ids']),query_sha256=sha(spec['query']),query_payload_sha256=array_sha(payload),heldout_rows_sha256=sha(spec['heldout_rows']) if spec.get('heldout_rows') else None)

def check(path,gt_path,nq,nd,k=100,heldout_path=None):
 ids=np.full((nq,k),-1,dtype=np.int64);scores=np.full((nq,k),np.nan);lat=np.full(nq,np.nan);count=0
 with open(path) as f:
  for r in csv.DictReader(f,delimiter='\t'):
   q=int(r['query_id']);j=int(r['rank'])-1;doc=int(r['document_id']);s=float(r['quantized_score']);t=float(r['latency_ms'])
   need((q,j)==divmod(count,k),'Missing, repeated, or reordered query/rank rows')
   need(0<=q<nq and 0<=j<k and 0<=doc<nd,'Invalid query/rank/document ID')
   need(np.isfinite(s) and np.isfinite(t) and t>0,'Invalid score/latency')
   if j:need(lat[q]==t,'Inconsistent per-query timing')
   ids[q,j]=doc;scores[q,j]=s;lat[q]=t;count+=1
 need(count==nq*k and all(len(set(a))==k for a in ids),'Incomplete or duplicate output IDs')
 need(np.all(np.diff(scores,axis=1)<=0),'Scores are not descending')
 gt=np.load(gt_path);need(gt.shape[0]==nq and gt.shape[1]>=k,'GT dimensions changed')
 recalls={cut:np.asarray([len(set(a[:cut])&set(b[:cut]))/cut for a,b in zip(ids,gt)],dtype=float) for cut in ((10,100) if k==100 else (10,))}
 result=dict(k=k,primary_metric='recall'+str(k),queries=nq,returned=count,qps=float(1000/lat.mean()),mean_ms=float(lat.mean()),p50_ms=float(np.percentile(lat,50)),p99_ms=float(np.percentile(lat,99)),ids_sha256=array_sha(ids),output_sha256=sha(path))
 for cut,v in recalls.items():result['recall'+str(cut)]=float(v.mean())
 if heldout_path:
  rows=np.load(heldout_path);need(rows.ndim==1 and len(np.unique(rows))==len(rows) and np.all((rows>=0)&(rows<nq)),'Invalid heldout selection')
  result['heldout_queries']=int(len(rows));cal=np.setdiff1d(np.arange(nq),rows)
  for cut,v in recalls.items():result['heldout_recall'+str(cut)]=float(v[rows].mean());result['calibration_recall'+str(cut)]=float(v[cal].mean())
 return result

def memory_sample(pid):
 mem={}
 for line in Path('/proc/meminfo').read_text().splitlines():
  key,value=line.split(':',1)
  if key in ('MemTotal','MemAvailable','MemFree','Cached','Buffers','SwapTotal','SwapFree'):mem[key+'_kB']=int(value.split()[0])
 process={}
 try:
  for line in Path(f'/proc/{pid}/status').read_text().splitlines():
   key,_,value=line.partition(':')
   if key in ('VmRSS','VmSize','RssAnon','RssFile','VmSwap'):process[key+'_kB']=int(value.split()[0])
   elif key=='Cpus_allowed_list':process[key]=value.strip()
 except FileNotFoundError:pass
 group={}
 try:
  for path in Path(f'/proc/{pid}/task').iterdir():
   try:
    allowed=next(x.split(':',1)[1].strip() for x in (path/'status').read_text().splitlines() if x.startswith('Cpus_allowed_list:'));group[path.name]=allowed
   except (FileNotFoundError,ProcessLookupError,StopIteration):pass
 except FileNotFoundError:pass
 return dict(system_memory=mem,process_memory=process,thread_group_allowed_cpus=group,memory_note='MemAvailable/MemFree and process RSS are separate views; do not sum them; cudaMemGetInfo is not physical memory accounting')
def query_gpu():
 util=subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=5)
 apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,timeout=5)
 return util,apps

def run_main(lock=None,initial_load_snapshot=None):
 a=argparse.ArgumentParser();a.add_argument('--spec',required=True);a.add_argument('--output',required=True);a.add_argument('--query');a.add_argument('--gt');a.add_argument('--gt-ids');a.add_argument('--run',action='store_true');args=a.parse_args()
 spec=json.loads(Path(args.spec).read_text());mode=modes(spec);reject_quarantined(spec['index']);limit=int(re.search(r'kCandidateRefinementMaxNprobe\s*=\s*(\d+)',(ROOT/'arm-source/cpp/include/chimera/config.cuh').read_text()).group(1))
 validate_points(spec,limit)
 for key,val in [('query',args.query),('gt',args.gt),('gt_ids',args.gt_ids)]:
  if val:spec[key]=val
 with open(spec['query'],'rb') as f:nq=struct.unpack('<i',f.read(4))[0]
 if spec.get('nq',-1)>0:nq=min(nq,spec['nq'])
 with open(Path(spec['index'])/'doclens.bin','rb') as f:nd=struct.unpack('<i',f.read(4))[0]
 point_semantics=validate_points(spec,limit,nd)
 out=Path(args.output);out.mkdir(parents=True,exist_ok=False);binary=Path(spec.get('binary',ROOT/'build/gpu_search'))
 affinity=str(spec.get('cpu_affinity',spec.get('cpu_core',7)));threads=int(spec.get('threads',1));need(0<threads<=len(cpu_set(affinity)),'OMP threads exceed selected CPU cores')
 need(cpu_set(affinity)<=cpu_set(Path('/sys/devices/system/cpu/online').read_text().strip()),'Requested offline CPU cores')
 omp=omp_budget(spec,affinity,threads)
 index_meta=index_identity(spec['index']);identity=identity_checks(spec,index_meta,nq,nd);k=identity['k'];(out/'index-identity.json').write_text(json.dumps(index_meta,indent=2))
 if spec.get('phase')=='formal':need(nq==int(spec.get('expected_queries',2931)) and spec.get('rounds')==3 and spec.get('selection_frozen_from'),'Formal run needs frozen calibration choice and three complete declared query sets')
 frozen_binary=out/'frozen-gpu-search'
 if frozen_binary.exists():need(sha(frozen_binary)==sha(binary),'Existing frozen executable differs; select another output directory')
 else:shutil.copy2(binary,frozen_binary);frozen_binary.chmod(0o755)
 need(sha(frozen_binary)==sha(binary),'Executable changed while freezing')
 frozen_guard=out/'frozen-exec-guard';shutil.copy2(ROOT/'search_exec_guard_v2',frozen_guard);frozen_guard.chmod(0o755)
 base=['taskset','-c',affinity,str(frozen_binary),'--index',spec['index'],'--query',spec['query'],'--gt',spec['gt'],'--k',str(k),'--nq',str(nq),'--warmup',str(spec.get('warmup',5))]
 if spec.get('low_memory',False):base+=['--low-memory']
 base+=['--thread-info','--full-bit-device',spec.get('full_bit_device','gpu')]
 if spec.get('verify_full_bit'):base+=['--verify-full-bit']
 if 'kernel_mode' in spec:base+=['--kernel-mode',mode['kernel_mode']]
 if 'algorithm_variant' in spec:base+=['--algorithm-variant',mode['algorithm_variant']]
 if mode['expected_coarse_clusters'] is not None:
  need(mode['expected_coarse_clusters']==index_meta['n_clusters'],'Declared coarse clusters differ from index')
  base+=['--ablation-coarse-clusters',str(mode['expected_coarse_clusters'])]
 commands=[]
 for pointidx,point in enumerate(spec['points']):
  for rnd in range(spec.get('rounds',1)):
   name=f'point{pointidx:02d}-round{rnd}';cmd=base+['--nprobe',str(point['nprobe']),'--k-refine',str(point['k_refine']),'--k-full-bit',str(point['k_full_bit']),'--cagra-itopk-size',str(point.get('cagra_itopk_size',min(point['nprobe']+32,512))),'--num-chunks',str(point['num_chunks']),'--output',str(out/(name+'.tsv'))]
   commands.append((name,point,cmd))
 for fname in ('arm-port.patch','arm-port-manifest.json'):
  if (ROOT/fname).exists():(out/fname).write_bytes((ROOT/fname).read_bytes())
 prepared=dict(GPU_environment={'CUDA_VISIBLE_DEVICES':'0','UMA_RUN_GPU_TESTS':'1'},spec=spec,queries=nq,docs=nd,index_identity=index_meta,**identity,binary_sha256=sha(frozen_binary),binary_snapshot=str(frozen_binary),binary_build_path=str(binary),spec_sha256=sha(args.spec),runner_sha256=sha(__file__),commands=[x[2] for x in commands],expected_searches=len(commands),capture_scores=True,load_mode='read_only_mmap_stream_upload' if spec.get('low_memory',False) else 'original_owning_arrays',prefault_full_codes=bool(spec.get('low_memory',False)),CPU_affinity=affinity,OpenMP_environment=omp,OMP_NUM_THREADS=threads,query_input_placement='host FP32, upload included',query_output_placement='host document IDs and captured quantized scores',boundary='sum of synchronous search calls; encoding/index loading/warmup/disk output excluded; loop_qps includes score snapshot')
 prepared['device_lock_policy']={'path':str(Path.home()/'device.lock'),'mode':'LOCK_EX|LOCK_NB','held_before_run_initialization':bool(lock is not None),'inherited_by_native':bool(args.run),'fixed_idle_wait_s':0,'initial_external_load':initial_load_snapshot}
 prepared['nprobe_capacity_limit']=limit;prepared['point_parameter_semantics']=point_semantics
 prepared['search_modes']=mode;prepared['selection_family']=selection_family(prepared)
 prepared['runner_dependencies_sha256']={name:sha(ROOT/name) for name in ('search_supervision_v2.py','search_modes_v2.py','routing_audit.py')};prepared['exec_guard_sha256']=sha(frozen_guard)
 sources=out/'frozen-runner-sources';sources.mkdir();prepared['runner_source_snapshot']={}
 for name in ('run_search_grid_device_lock.py','search_supervision_v2.py','search_modes_v2.py','routing_audit.py'):
  shutil.copy2(Path(__file__) if name=="run_search_grid_device_lock.py" else ROOT/name,sources/name);prepared['runner_source_snapshot'][name]={'path':str(sources/name),'sha256':sha(sources/name)}
 need(prepared['runner_source_snapshot']['run_search_grid_device_lock.py']['sha256']==prepared['runner_sha256'],'Runner changed during source snapshot')
 (out/'prepared.json').write_text(json.dumps(prepared,indent=2))
 if not args.run:print('Prepared only: '+str(out));return
 env=dict(os.environ,LD_LIBRARY_PATH='/home/lxq/miniconda3/envs/cuvs_benchmarks/lib:/usr/local/cuda-13.0/lib64',OMP_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',UMA_RUN_GPU_TESTS='1');env.update(omp);env.pop('GOMP_CPU_AFFINITY',None)
 need(lock is not None and not lock.closed,'Persistent device.lock must be held before run initialization');complete=False;results=[]
 try:
  for name,point,cmd in commands:
   need(not (out/(name+'.tsv')).exists(),'Refusing to overwrite existing output: '+name);start=time.monotonic()
   with open(out/(name+'.log'),'x') as f,open(out/(name+'-monitor.jsonl'),'x') as mon:
    last_poll=[float('-inf')]
    def monitor(child):
     if time.monotonic()-start>spec.get('timeout_per_point_s',7200):raise TimeoutError(name)
     if time.monotonic()-last_poll[0]<1:return
     last_poll[0]=time.monotonic();util,apps=query_gpu();foreign=[]
     for row in apps.stdout.strip().splitlines():
      try:other=int(row.split(',',1)[0])
      except ValueError:continue
      if other!=child.pid:foreign.append(row)
     mon.write(json.dumps(dict(time=time.time(),pid=child.pid,gpu=util.stdout.strip(),gpu_query_returncode=util.returncode,apps=apps.stdout.strip(),foreign_gpu_processes=foreign,foreign_note='Residency alone is not evidence of active compute; review actual interference',**memory_sample(child.pid)))+'\n');mon.flush()
    def started(child):
     nonlocal p
     p=child
    def write_receipt(path,value):
     with open(path,'x') as receipt:json.dump(value,receipt,indent=2);receipt.flush();os.fsync(receipt.fileno())
    p=None
    supervision=supervise_child(cmd,env=env,log=f,output_lock=lock,exec_guard=frozen_guard,receipt_path=out/(name+'-supervision.json'),write_receipt=write_receipt,on_started=started,on_poll=monitor)
    need(supervision['requested_signal'] is None,'Search runner interrupted')
   need(p.returncode==0,name+' subprocess failed');thread_check=check_thread_info(out/(name+'.log'),omp);check_index_unchanged(spec['index'],index_meta);verify_routing_dataset(spec,index_meta)
   val=check(out/(name+'.tsv'),spec['gt_ids'],nq,nd,k,heldout_path=spec.get('heldout_rows'));val.update(parameter_semantics=validate_points(dict(spec,points=[point]),limit,nd)[0],actual_search_modes=verify_mode_log(out/(name+'.log'),spec),point=point,round_name=name,thread_affinity=thread_check,process_wall_seconds=time.monotonic()-start,pagefaults=pagefault_evidence(out/(name+'.log'),spec.get('low_memory',False)))
   if spec.get('quality_target'):
    target=spec['quality_target'];key=('heldout_' if spec.get('heldout_rows') else '')+target['metric'];val['quality_validation_pass']=val[key]>=target['threshold']
   if spec.get('quality_targets'):
    val['quality_validation']={str(t['metric'])+'@'+str(t['threshold']):val[('heldout_' if spec.get('heldout_rows') else '')+t['metric']]>=t['threshold'] for t in spec['quality_targets']}
   results.append(val);(out/(name+'-validation.json')).write_text(json.dumps(val,indent=2));(out/'summary.json').write_text(json.dumps(results,indent=2));print(json.dumps(val),flush=True)
  need(len(results)==len(commands),'Incomplete configured grid');complete=True
 finally:
  exited=('p' not in locals()) or p is None or p.poll() is not None;need(exited,'Cannot release global lease with owned child alive');fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
  (out/'release.json').write_text(json.dumps(dict(subprocesses_exited=exited,flock_released=True,complete=complete,expected_searches=len(commands),completed_searches=len(results)),indent=2))
def main():
 lock=None;initial_load=None
 try:
  if '--run' in sys.argv:
   lock=open(Path.home()/'device.lock','a+')
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   util,apps=query_gpu();initial_load={'time':time.time(),'GPU_query_returncode':util.returncode,'GPU':util.stdout.strip(),'GPU_process_query_returncode':apps.returncode,'GPU_processes':apps.stdout.strip(),'system_loadavg':Path('/proc/loadavg').read_text().strip()}
  run_main(lock,initial_load)
 finally:
  if lock is not None and not lock.closed:
   fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
