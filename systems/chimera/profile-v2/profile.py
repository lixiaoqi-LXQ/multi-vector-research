"""Independent Full-only cal32/cal100 diagnostic preparation, QA and NSYS tree supervision."""
import argparse,fcntl,hashlib,json,os,shutil,struct,subprocess,sys,time,uuid
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parent;BASE=ROOT.parent;DATA=Path('/home/lxq/graduate/data/multivector-colbertv2')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(P/'controls'))
from run_search_grid_v2 import sha,need,identity_checks,check_index_unchanged,omp_budget,check_thread_info,pagefault_evidence,check
from search_modes_v2 import modes,validate_points,verify_mode_log
from child_supervision import SupervisedPopen
class ProfilePopen(SupervisedPopen):
 # Outer TERM allows the inner subreaper's10s cleanup to finish and write its receipt.
 def terminate_and_reap(self,grace=25):return super().terminate_and_reap(grace=grace)
MAIN=BASE/'chimera-alg-ablation/build/gpu_search';OBS=ROOT/'observation-v2/build/gpu_search'
MAIN_SHA='c54539b5143f63ad96ab0118f165dded7861d9c5af180767cd74c66d7bd4945d';OBS_SHA='65589e36d4772f31c1dc112468658cc2ffe64363521ceb1256477d5735331875'
SOURCE_FREEZE=ROOT/'observation-v2/source-freeze.json';SOURCE_SHA='1798b5ef1acdbb30593fab5c87c64e3156b7bd91dcfc6acc980cceb48ccf42f4'
RUNNER_FREEZE=BASE/'chimera-protocol/search-runner-v2/freeze.json';RUNNER_SHA='64957121b7edd97197802d461fd4f0785d101b69d7ddf382547afb70b1eef18b'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with open(p,'x')as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def cal_spec(s):
 need(modes(s)==dict(kernel_mode='full',algorithm_variant='full',expected_coarse_clusters=None),'Profile v2 supports main Full only')
 datasets={'lotte-pooled-dev-full':('lotte-calibration-seed42','lotte-pooled-dev-full','calibration',2428854,266205513),'msmarco-v1-passage-full':('msmarco-calibration-seed42','msmarco-v1-passage-full','calibration-seed42',8841823,597909919)}
 need(s['dataset']in datasets,'Explicit full LoTTE/MS dataset required');split,dataset,gtcal,docs,n=datasets[s['dataset']];sp=DATA/'splits'/split;gt=DATA/'groundtruth'/dataset
 s=dict(s);s.update(query=str(sp/'queries.bin'),query_rows=str(sp/'calibration-rows.npy'),split_manifest=str(sp/'manifest.json'),gt=str(gt/gtcal/'groundtruth.tsv'),gt_ids=str(gt/gtcal/'ids.npy'),gt_manifest=str(gt/'manifest.json'),full_gt_ids=str(gt/'ids.npy'),full_gt=str(gt/'groundtruth.tsv'),full_query=str(DATA/'chimera-inputs'/dataset/'queries.bin'),phase='calibration',nq=100,rounds=1,warmup=32,algorithm_variant='full',kernel_mode='full',require_identity=True,require_build_identity=True)
 s.pop('heldout_rows',None);need(s['expected_docs']==docs and s['expected_index']['N']==n,'Dataset dimensions mismatch');need(s.get('threads')==20 and s.get('cpu_affinity')=='0-19','Profile budget requires explicit OMP20 cores0-19');need(len(s['points'])==1,'Profile one frozen point only');validate_points(s,512,docs)
 return s
def accepted_identity(s):
 a=Path(s['root_index_acceptance']);need(sha(a)==s['root_index_acceptance_sha256'],'Root acceptance changed');r=read(a);need(r.get('root_accepted')is True and r['index']==s['index'],'Exact root-accepted index required');meta=s['finalized_index_identity'];need({n:v['sha256']for n,v in meta['files'].items()}==r['payload_sha256'],'Finalized/root payload hashes differ');check_index_unchanged(s['index'],meta)
 need(sha(Path(s['index'])/'build-provenance.json')==r['build_provenance_sha256'],'Accepted build provenance changed');need(sha(Path(s['index'])/'ROUTING_DATASET_AUDIT.json')==r['routing_sidecar_sha256'],'Accepted sidecar changed');need(s['routing_dataset_audit_receipt']==r['routing_dataset_audit_receipt']and s['routing_dataset_audit_sha256']==r['routing_dataset_audit_sha256'],'Spec/root routing audit differs')
 return meta,identity_checks(s,meta,100,s['expected_docs'])
def engine_command(s,binary,nq,out,profile=False):
 p=s['points'][0];cmd=['taskset','-c','0-19',str(binary),'--index',s['index'],'--query',s['query'],'--gt',s['gt'],'--k',str(s.get('k',100)),'--nq',str(nq),'--warmup','32','--thread-info','--algorithm-variant','full','--kernel-mode','full','--nprobe',str(p['nprobe']),'--k-refine',str(p['k_refine']),'--k-full-bit',str(p['k_full_bit']),'--cagra-itopk-size',str(p.get('cagra_itopk_size',min(p['nprobe']+32,512))),'--num-chunks',str(p['num_chunks']),'--output',str(out)]
 if s.get('low_memory'):cmd+=['--low-memory']
 if profile:cmd+=['--profile-range']
 return cmd
def prepare(a):
 need(sha(RUNNER_FREEZE)==RUNNER_SHA,'V2 runner freeze changed')
 for name,digest in read(RUNNER_FREEZE)['files'].items():need(sha(name)==digest,'V2 runner dependency changed: '+name)
 s=cal_spec(read(a.spec));meta,identity=accepted_identity(s)
 need(sha(MAIN)==MAIN_SHA and sha(OBS)==OBS_SHA and sha(SOURCE_FREEZE)==SOURCE_SHA,'Qualified binary/observation source identity changed');sf=read(SOURCE_FREEZE);need(sf['search_computation_text_identical_after_removing_NVTX_scaffolding']is True,'Missing computation equality evidence')
 for row in sf['source_files']:need(sha(SOURCE_FREEZE.parent/row['path'])==row['sha256'],'Observation source changed: '+row['path'])
 out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);(out/'tools').mkdir();files={}
 dependencies=[P/'profile.py',P/'tree_guard.py',P/'leaf_exec.py',P/'controls/child_supervision.py',P/'controls/child_entry.py',ROOT/'run_search_grid_v2.py',ROOT/'search_supervision_v2.py',ROOT/'search_modes_v2.py',ROOT/'routing_audit.py']
 for src in dependencies:
  relative=Path('controls')/src.name if src.parent.name=='controls'else Path(src.name);dst=out/'tools'/relative;dst.parent.mkdir(exist_ok=True);shutil.copy2(src,dst);files[str(dst)]=sha(dst);need(sha(src)==files[str(dst)],'Source snapshot changed');files[str(src)]=sha(src)
 for src,name in [(MAIN,'main-search'),(OBS,'observation-search'),(SOURCE_FREEZE,'observation-source-freeze.json'),(BASE/'chimera-alg-ablation/source-manifest.json','main-source-manifest.json'),(RUNNER_FREEZE,'runner-freeze.json')]:
  dst=out/name;shutil.copy2(src,dst);files[str(dst)]=sha(dst)
 nsys=Path(a.nsys).resolve(strict=True);need(nsys.is_file(),'NSYS executable missing');files[str(nsys)]=sha(nsys)
 rows=np.load(s['query_rows']);ids=np.load(s['gt_ids']);need(len(rows)==100 and ids.shape[0]==100,'Actual calibration set must contain100 rows')
 for n in (32,100):
  np.save(out/f'original-query-rows-{n}.npy',rows[:n]);np.save(out/f'gt-ids-{n}.npy',ids[:n]);files[str(out/f'original-query-rows-{n}.npy')]=sha(out/f'original-query-rows-{n}.npy');files[str(out/f'gt-ids-{n}.npy')]=sha(out/f'gt-ids-{n}.npy')
 omp=omp_budget(s,'0-19',20);env=dict(omp,OMP_NUM_THREADS='20',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',UMA_RUN_GPU_TESTS='1',LD_LIBRARY_PATH='/home/lxq/miniconda3/envs/cuvs_benchmarks/lib:/usr/local/cuda-13.0/lib64')
 libs=subprocess.check_output(['ldd',str(out/'observation-search')],text=True,env=dict(os.environ,**env));need('not found'not in libs,'Unresolved observation runtime libraries')
 for line in libs.splitlines():
  import re
  m=re.search(r'(?:=>\s+)?(/[^\s]+)\s+\(',line)
  if m:files[str(Path(m[1]).resolve())]=sha(Path(m[1]).resolve())
 write(out/'prepared.json',dict(status='PREPARED_PROFILE_DIAGNOSTIC_NO_GPU_RUN',spec=s,input_spec=str(Path(a.spec).resolve()),input_spec_sha256=sha(a.spec),index_identity=meta,identity=identity,queries=a.queries,actual_original_query_rows=rows[:a.queries].tolist(),all_calibration_original_rows=rows.tolist(),environment=env,nsys=str(nsys),files=files,timeout_s=1800,cleanup_s=10,performance_claim=False,profile_timing_eligible_for_QPS=False,expected_query_ranges=a.queries,source_freeze_sha256=SOURCE_SHA))
 print(json.dumps({'prepared':str(out),'prepared_sha256':sha(out/'prepared.json'),'GPU_used':False,'NSYS_executed':False}))
def validate_output(out,stage,s,nq,prepared):
 log=out/stage/'output.log';tsv=out/stage/'results.tsv';v=check(tsv,Path(prepared)/f'gt-ids-{nq}.npy',nq,s['expected_docs'],s.get('k',100));v={k:x for k,x in v.items()if k not in ['qps','mean_ms','p50_ms','p99_ms']};v.update(actual_modes=verify_mode_log(log,s),OMP=check_thread_info(log,read(Path(prepared)/'prepared.json')['environment']),pagefaults=pagefault_evidence(log,s.get('low_memory',False)),diagnostic_timing_only=True);need(v['pagefaults']['query_loop']['major_delta']==0,'Query major faults need investigation before qualification');write(out/stage/'output-validation.json',v)
 return v
def id_scores(path):
 import csv
 rows=list(csv.DictReader(open(path),delimiter='\t'));return np.array([int(x['document_id'])for x in rows],dtype=np.int64),np.array([float(x['quantized_score'])for x in rows],dtype=np.float32)
def run_stage(out,stage,binary,nq,s,config,lease,profile=False):
 d=out/stage;d.mkdir();nonce=uuid.uuid4().hex;cfgpath=d/'tree-config.json';engine=engine_command(s,binary,nq,d/'results.tsv',profile)
 cfg=dict(socket_name='chimera-profile-'+nonce,token=nonce,lock_path='/tmp/uma-paper-device0.lock',engine_command=engine,engine_binary=str(binary),engine_sha256=sha(binary),leaf_receipt=str(d/'leaf-lease.json'),tree_receipt=str(d/'tree-supervision.json'),timeout_s=config['timeout_s'],cleanup_s=config['cleanup_s'])
 leaf=[sys.executable,str(out/'tools/leaf_exec.py'),str(cfgpath)]
 cfg['launch_command']=([config['nsys'],'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-memory-usage=false','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','--output',str(d/'trace'),*leaf]if profile else leaf);write(cfgpath,cfg)
 env=dict(os.environ,**config['environment']);env.pop('GOMP_CPU_AFFINITY',None)
 with open(d/'output.log','x')as log:
  child=ProfilePopen([sys.executable,str(out/'tools/tree_guard.py'),str(cfgpath),str(lease.fileno())],env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
  write(d/'process.json',dict(supervisor_pid=child.pid,supervisor_start_ticks=int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]),command=cfg['launch_command'],GPU_used=True,performance_claim=False))
  try:rc=child.wait(timeout=config['timeout_s']+config['cleanup_s']+20)
  finally:
   if child.poll()is None:child.terminate_and_reap()
 need(rc==0,'Profiler/QA child tree failed');tree=read(d/'tree-supervision.json');leaf=read(d/'leaf-lease.json');need(tree['all_owned_descendants_gone']and len(tree['lease_transfers'])==1 and tree['lease_transfers'][0]['pid']==leaf['pid']and leaf['lease_inheritable'],'GPU leaf lease not proven');return validate_output(out,stage,s,nq,out)
def execute(a):
 out=Path(a.prepared).resolve();config=read(out/'prepared.json');s=config['spec']
 for path,h in config['files'].items():need(sha(path)==h,'Frozen profile dependency changed: '+path)
 need(sha(config['input_spec'])==config['input_spec_sha256'],'Input spec changed');meta,identity=accepted_identity(s);need(meta==config['index_identity']and identity==config['identity'],'Profile identity changed')
 need(os.environ.get('UMA_RUN_GPU_TESTS')=='1','Explicit UMA_RUN_GPU_TESTS=1 and root scheduling required')
 os.environ['PLAID_DEVICE_LOCK']='/tmp/uma-paper-device0.lock'
 if a.stage=='profile':
  qa=read(out/'qa-acceptance.json');need(qa['status']=='PASS_MAIN_OBSERVATION_32_ORDERED_IDS_F32_BITS'and qa['prepared_sha256']==sha(out/'prepared.json'),'Same prepared identity 32QA required before trace')
 stages=['main-qa','observation-qa']if a.stage=='qa'else['profile']
 for st in stages:need(not(out/st).exists(),'Fresh stage output required: '+st)
 with open('/tmp/uma-paper-device0.lock','a+')as lease:
  fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  idle=[]
  for _ in range(10):
   util=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5);need(int(util.strip())==0,'GPU nonidle before diagnostic');idle.append(dict(unix=time.time(),utilization=int(util.strip())));time.sleep(1)
  write(out/(a.stage+'-idle.json'),dict(global_exclusive_lease=True,samples=idle,prepared_sha256=sha(out/'prepared.json')))
  for st in stages:run_stage(out,st,out/('main-search'if st=='main-qa'else'observation-search'),32 if a.stage=='qa'else config['queries'],s,config,lease,profile=a.stage=='profile');check_index_unchanged(s['index'],meta)
  if a.stage=='qa':
   i,x=id_scores(out/'main-qa/results.tsv');j,y=id_scores(out/'observation-qa/results.tsv');need(np.array_equal(i,j)and np.array_equal(x.view(np.uint32),y.view(np.uint32)),'Observation changed ordered IDs/F32 score bits');write(out/'qa-acceptance.json',dict(status='PASS_MAIN_OBSERVATION_32_ORDERED_IDS_F32_BITS',count=len(i),ids_sha256=hashlib.sha256(i.tobytes()).hexdigest(),scores_f32_sha256=hashlib.sha256(x.tobytes()).hexdigest(),prepared_sha256=sha(out/'prepared.json'),performance_claim=False))
  else:
   i,x=id_scores(out/'main-qa/results.tsv');j,y=id_scores(out/'profile/results.tsv');need(np.array_equal(i,j[:len(i)])and np.array_equal(x.view(np.uint32),y[:len(x)].view(np.uint32)),'Profile first32 ordered IDs/F32bits differ from same main QA');write(out/'profile-execution.json',dict(status='TRACE_CREATED_AWAITING_CAPTURE_COUNT_AND_KHAT_REVIEW',prepared_sha256=sha(out/'prepared.json'),expected_query_ranges=config['queries'],profile_first32_equals_main_QA=True,performance_claim=False,trace_sha256=sha(out/'profile/trace.nsys-rep'),actual_original_query_rows=config['actual_original_query_rows']))
 print(json.dumps({'stage':a.stage,'out':str(out),'performance_claim':False}))
def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True);q=sub.add_parser('prepare');q.add_argument('--spec',required=True);q.add_argument('--out',required=True);q.add_argument('--queries',type=int,choices=[32,100],default=32);q.add_argument('--nsys',default='/usr/local/cuda/bin/nsys');r=sub.add_parser('run');r.add_argument('--prepared',required=True);r.add_argument('--stage',choices=['qa','profile'],required=True);a=p.parse_args();prepare(a)if a.action=='prepare'else execute(a)
if __name__=='__main__':main()
