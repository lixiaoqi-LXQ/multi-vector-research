"""Frozen bounded kernel correctness cohort; no profiling/performance experiment."""
from pathlib import Path
import os,sys,json,signal,time,subprocess,hashlib,fcntl,argparse,csv
F=Path(__file__).resolve().parent;R=F.parent.parent;sys.path.insert(0,str(F))
from supervision import supervise_child
from checker import need,check,check_thread_info,omp_budget,pagefault_evidence
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args();freeze=json.loads((F/'freeze.json').read_text())
def sha(p):
 with open(p,'rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
for path,digest in freeze['files_sha256'].items():need(sha(path)==digest,'Freeze changed: '+path)
if not a.run:print(json.dumps({'prepared_only':True,'GPU_used':False,'freeze':str(F/'freeze.json')}));raise SystemExit
D=Path('/home/lxq/graduate/data/multivector-colbertv2');out=F/'run';out.mkdir(exist_ok=False);env=os.environ.copy();env.update(freeze['environment']);env.pop('GOMP_CPU_AFFINITY',None);index=D/'indexes/chimera/lotte-prefix-10000-c8192-b4-owned-device-qualification';audit=json.loads((index/'ROUTING_DATASET_AUDIT.json').read_text());expected=json.loads(Path(freeze['saved_dataset_receipt']).read_text());need(expected['float32_bit_mismatches']==0 and expected['final_cagra_file_sha256']==sha(index/'centroids.carga'),'Saved dataset audit changed');gt=D/'groundtruth/lotte-prefix-10000';query=D/'chimera-inputs/lotte-prefix-10000/queries.bin';np.save(out/'gt32.npy',np.load(gt/'ids.npy')[:32]);np.save(out/'gt8.npy',np.load(gt/'ids.npy')[:8]);metrics={};outputs={}
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2))
def live_expected(pid,start):
 try:
  raw=Path(f'/proc/{pid}/stat').read_text();v=raw[raw.rfind(')')+2:].split();return v[0]!='Z' and int(v[19])==start
 except FileNotFoundError:return False
def lease():
 text=subprocess.check_output(['lslocks','-n','-o','PID,MODE,PATH'],text=True);holders=[int(x.split()[0]) for x in text.splitlines() if x.split()[-1]=='/tmp/uma-paper-device0.lock' and '*' not in x.split()[1]]
 need((holders==[3228697] and live_expected(3233295,245395633)) or (holders==[3255085] and live_expected(3255093,245446209)),'Authorized offline holder/long job no longer live')
def deadline(sig,frame):raise TimeoutError('170-second owned helper limit')
def execute(name,command):
 lease();folder=out/name;folder.mkdir();write(folder/'command.json',{'command':command,'environment':freeze['environment'],'performance_claim':False,'timeout_s':170,'grace_s':10});start=time.monotonic()
 with open(folder/'output.lock','a+') as lock,open(folder/'stdout.log','w') as log:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);old=signal.signal(signal.SIGALRM,deadline);signal.alarm(170)
  def started(child):write(folder/'launch.json',{'pid':child.pid,'command':command});print(name,child.pid,flush=True)
  try:supervision=supervise_child(command,env=env,log=log,output_lock=lock,exec_guard=F/'exec_guard',receipt_path=folder/'supervision.json',write_receipt=write,on_started=started,terminate_grace_s=10)
  finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)
 need(supervision['returncode']==0,name+' failed: stop subsequent helpers');return folder
try:
 p1=execute('P1-kernels',['taskset','-c','15-18',str(F/'kernel_ablation_qa'),'--run']);lines=(p1/'stdout.log').read_text().splitlines();result=json.loads(next(x for x in reversed(lines) if x.startswith('{')));need(result['passed'] is True,'Kernel numerical oracle failed');metrics['P1']=result
 cases=[('P2-reference',None,32,100,10000,400,1)]+[(f'P2-{mode}',mode,32,100,10000,400,1) for mode in ('full','no-lut','no-aggregation','neither')]+[(f'P3-{mode}',mode,8,10,11,10,10) for mode in ('full','no-lut','no-aggregation','neither')]
 for name,mode,nq,k,refine,full,chunks in cases:
  binary=F/('original_gpu_search' if mode is None else 'gpu_search');file=out/(name+'.tsv');cmd=['taskset','-c','15-18',str(binary),'--index',str(index),'--query',str(query),'--gt',str(gt/'groundtruth.tsv'),'--nq',str(nq),'--k',str(k),'--warmup','5','--nprobe','128','--k-refine',str(refine),'--k-full-bit',str(full),'--num-chunks',str(chunks),'--cagra-itopk-size','160','--thread-info','--output',str(file)]
  if mode is not None:cmd+=['--algorithm-variant','full','--kernel-mode',mode]
  folder=execute(name,cmd);metric=check(file,out/f'gt{nq}.npy',nq,10000,k);metric['threads']=check_thread_info(folder/'stdout.log',env);metric['pagefaults']=pagefault_evidence(folder/'stdout.log',False);need(not metric['pagefaults']['major_fault_investigation_required'],'Major fault needs investigation');metrics[name]=metric
  with open(file) as f:outputs[name]={(int(x['query_id']),int(x['rank'])):(int(x['document_id']),float(x['quantized_score'])) for x in csv.DictReader(f,delimiter='\t')}
  if name in ('P2-full','P2-no-aggregation'):
   baseline=outputs['P2-reference'] if name=='P2-full' else outputs['P2-full'];need(outputs[name]==baseline,'Full/reference or LUT aggregation final IDs/scores not bitwise equal')
  for baseline_name in ('P2-full','P3-full'):
   if name.startswith(baseline_name[:2]) and baseline_name in outputs and name!=baseline_name:
    first={(q,v[0]):v[1] for (q,r),v in outputs[baseline_name].items()};current={(q,v[0]):v[1] for (q,r),v in outputs[name].items()};common=first.keys()&current.keys();need(all(first[x]==current[x] for x in common),'Unchanged CPU fullbit scores differ');metric['versus_full']={'common_query_doc_pairs':len(common),'same_ordered_rows':sum(outputs[name][x]==outputs[baseline_name][x] for x in outputs[name]),'same_topK_sets':sum({v[0] for x,v in outputs[name].items() if x[0]==q}=={v[0] for x,v in outputs[baseline_name].items() if x[0]==q} for q in range(nq)),'common_CPU_score_max_error':0}
  write(folder/'validation.json',metric);write(out/'progress.json',metrics)
 write(out/'result.json',{'passed':True,'metrics':metrics,'performance_claim':False,'freeze_sha256':sha(F/'freeze.json')});print('PASS all bounded kernel/prefix helpers',flush=True)
except BaseException as e:
 write(out/'failure.json',{'type':type(e).__name__,'message':str(e),'completed':list(metrics),'all_later_helpers_stopped':True,'performance_claim':False});raise
