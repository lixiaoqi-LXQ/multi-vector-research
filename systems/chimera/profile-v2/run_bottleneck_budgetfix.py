"""Thin NSYS-software/targeted-NCU adapter over existing profile-v2 exact tree supervision."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time,uuid
from pathlib import Path
import bottleneck_plan as bp
v2=bp.v2;P=bp.P
def read(p):return json.loads(Path(p).read_text())
def write(p,v):v2.write(p,v)
def values(directory,plan):
 if plan.get('engine_kind')!='plaid':return v2.id_scores(directory/'rows.tsv')
 rows=[json.loads(x)for x in(directory/'endpoint/per-query.jsonl').read_text().splitlines()];return v2.np.asarray([i for x in rows for i in x['ids']],dtype=v2.np.int64),v2.np.asarray([s for x in rows for s in x['scores']],dtype=v2.np.float32)
def output_validation(out,plan):
 if plan.get('engine_kind')!='plaid':
  s=plan['spec'];v=v2.check(out/'rows.tsv',Path(plan['_directory'])/'gt32.npy',32,s['expected_docs'],10);v={k:x for k,x in v.items()if k not in ['qps','mean_ms','p50_ms','p99_ms']};v.update(actual_modes=v2.verify_mode_log(out/'output.log',s),OMP=v2.check_thread_info(out/'output.log',plan['environment']),pagefaults=v2.pagefault_evidence(out/'output.log',s.get('low_memory',False)),performance_claim=False);return v
 rows=[json.loads(x)for x in(out/'endpoint/per-query.jsonl').read_text().splitlines()];v2.need(len(rows)==32 and[x['query_row']for x in rows]==plan['original_query_rows']and all(x['round']==0 and len(x['ids'])==len(x['scores'])==10 for x in rows),'PLAID32 Top10 output shape changed')
 for x in rows:v2.need(len(set(x['ids']))==10 and all(0<=i<2428854 for i in x['ids'])and all(v2.np.isfinite(s)for s in x['scores'])and all(a>=b for a,b in zip(x['scores'],x['scores'][1:])),'PLAID output contract failed')
 identity=read(out/'endpoint/run-identity.json');v2.need(identity['query_placement']=='host'and identity['index_placement']==plan['index_placement']and identity['torch_threads']==20 and identity['cpu_affinity']==[7],'PLAID actual placement/budget changed')
 affinity=read(out/'endpoint/thread-affinity-after-warmup.json');bound={int(x['allowed_cpus'])for x in affinity['threads']if x['allowed_cpus'].isdigit()};v2.need(bound==set(range(20))and affinity['launcher_cpu_budget']=='0-19'and affinity['OMP_NUM_THREADS']=='20'and affinity['torch_threads']==20 and affinity['OMP_PLACES']==plan['environment']['OMP_PLACES'],'PLAID20core group budget/places changed')
 ref=read(Path(plan['_directory'])/'reference32.json');i,x=values(out,plan);ri=v2.np.asarray([i for q in ref for i in q['ids']],dtype=v2.np.int64);rs=v2.np.asarray([s for q in ref for s in q['scores']],dtype=v2.np.float32);v2.need(v2.np.array_equal(i,ri)and v2.np.array_equal(x.view(v2.np.uint32),rs.view(v2.np.uint32)),'PLAID32 differs from frozen calibration reference');return dict(status='PASS_PLAID32_TOP10_IDS_F32BITS_TO_FROZEN_CAL',count=320,performance_claim=False,run_identity_sha256=bp.sha(out/'endpoint/run-identity.json'),pagefaults_not_exposed_by_PLAID_endpoint=True)
def execute_one(plan,out,phase,kernel,count):
 plaid=plan.get('engine_kind')=='plaid';out.mkdir(exist_ok=False);nonce=uuid.uuid4().hex;cfgpath=out/'tree-config.json';command=list(plan['engine_command']);command[command.index('--output')+1]=str(out/('endpoint'if plaid else'rows.tsv'))
 if phase in ['main-qa','observation-qa']:
  command.remove('--profile-range')
  if plaid:
   if phase=='main-qa':command.remove('--nvtx-stages')
  else:command[3]=str(Path(plan['_directory'])/('main-search'if phase=='main-qa'else'observation-search'))
 cfg=dict(socket_name='chimera-bottleneck-'+nonce,token=nonce,lock_path='/tmp/uma-paper-device0.lock',engine_command=command,engine_binary=command[3],engine_sha256=bp.sha(command[3]),leaf_receipt=str(out/'leaf-lease.json'),tree_receipt=str(out/'tree-supervision.json'),timeout_s=900,cleanup_s=10)
 if phase.endswith('-qa'):launch=[sys.executable,str(P/'leaf_exec.py'),str(cfgpath)]
 else:
  template=plan['NSYS']if phase=='nsys'else plan['NCU'][phase];mapping={'{phase_dir}':str(out),'{python}':sys.executable,'{tree_config}':str(cfgpath),'{exact_kernel_or_regex}':kernel or'','{1_to_3}':str(count)};launch=list(template)
  for i,value in enumerate(launch):
   for key,replacement in mapping.items():value=value.replace(key,replacement)
   launch[i]=value
 cfg['launch_command']=launch;write(cfgpath,cfg);env=dict(os.environ,**plan['environment']);env.pop('GOMP_CPU_AFFINITY',None)
 with open(out/'output.log','x')as log:
  child=v2.ProfilePopen([sys.executable,str(P/'tree_guard.py'),str(cfgpath),str(plan['_lease'].fileno())],env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(plan['_lease'].fileno(),));write(out/'process.json',dict(pid=child.pid,command=launch,timeout_s=900,cleanup_s=10,performance_claim=False))
  try:rc=child.wait(timeout=935)
  finally:
   if child.poll()is None:child.terminate_and_reap()
 tree=read(out/'tree-supervision.json');v2.need(rc==0 and tree['all_owned_descendants_gone']and not tree['timed_out'],'Profiling tree failed; retain stage and stop');v=output_validation(out,plan)
 if phase.endswith('-qa')and not plaid:v2.need(v['pagefaults']['query_loop']['major_delta']==0,'Unprofiled QA majorfault gate failed')
 write(out/'validation.json',v)
 if not phase.endswith('-qa'):
  i,x=values(Path(plan['_directory'])/'observation-qa',plan);j,y=values(out,plan);v2.need(v2.np.array_equal(i,j)and v2.np.array_equal(x.view(v2.np.uint32),y.view(v2.np.uint32)),'Profiled output IDs/F32bits changed');write(out/'counter-or-trace-generated.json',dict(status='GENERATED_PENDING_TRACE_OR_COUNTER_REVIEW',query_major_faults=v['pagefaults']['query_loop']['major_delta']if'pagefaults'in v else None,ordered_ID_F32bits_equal_QA=True,report_sha256=bp.sha(out/('trace.nsys-rep'if phase=='nsys'else'counters.ncu-rep')),performance_claim=False))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--phase',choices=['qa','nsys','sol','memory','compute','latency'],required=True);p.add_argument('--kernel');p.add_argument('--launch-count',type=int,choices=[1,2,3],default=1);p.add_argument('--label');p.add_argument('--run',action='store_true');p.add_argument('--resume-qa',action='store_true');a=p.parse_args();file=Path(a.plan).resolve();plan=read(file);directory=file.parent
 if a.phase not in ['qa','nsys']:v2.need(bool(a.kernel),'Select actual NSYS kernel before NCU')
 if not a.run:print(json.dumps(dict(prepared_only=True,GPU_launched=False,phase=a.phase,kernel=a.kernel,launch_count=a.launch_count,plan_sha256=bp.sha(file))));return
 v2.need(os.environ.get('UMA_RUN_GPU_TESTS')=='1','Root scheduling and explicit UMA_RUN_GPU_TESTS=1 required')
 for name,h in plan['files'].items():v2.need(bp.sha(name)==h,'Frozen diagnostic dependency changed: '+name)
 if plan.get('engine_kind')=='plaid':
  import plaid_bottleneck_plan
  plaid_bottleneck_plan.guard(plan)
 else:v2.accepted_identity(plan['spec'])
 os.environ['PLAID_DEVICE_LOCK']='/tmp/uma-paper-device0.lock';plan['_directory']=str(directory)
 if a.phase!='qa':v2.need(read(directory/'QA-acceptance.json')['plan_sha256']==bp.sha(file),'Same-plan32 QA required')
 if a.phase not in ['qa','nsys']:
  selection=read(directory/'NSYS-kernel-selection.json');v2.need(selection['trace_sha256']==bp.sha(directory/'nsys/trace.nsys-rep')and selection['kernel_tables_present']is True and selection['captured_queries']==32 and selection['original_query_rows']==plan['original_query_rows'],'Require actual complete query/kernel trace review');v2.need(1<=len(selection['counter_kernel_filters'])<=3 and a.kernel in selection['counter_kernel_filters'],'Select at most3 actual major kernel families from reviewed NSYS data')
 with open('/tmp/uma-paper-device0.lock','a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);plan['_lease']=lock;idle=[]
  for _ in range(10):
   u=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5);v2.need(int(u.strip())==0,'GPU nonidle');idle.append(dict(unix=time.time(),utilization=0));time.sleep(1)
  label=a.label or(a.phase if a.phase in ['qa','nsys']else a.phase+'-'+hashlib.sha256(a.kernel.encode()).hexdigest()[:12]);v2.need('/'not in label and label not in ['.','..'],'Simple fresh label required');write(directory/(label+'-idle.json'),dict(real_global_exclusive_lease=True,samples=idle,plan_sha256=bp.sha(file)))
  if a.phase=='qa':
   for st in ['main-qa','observation-qa']:
    if a.resume_qa and st=='main-qa':
     tree=read(directory/st/'tree-supervision.json');v2.need(tree['returncode']==0 and tree['all_owned_descendants_gone'],'Existing mainQA not reaped successful');v=output_validation(directory/st,plan);write(directory/st/'validation.json',v);write(directory/'main-QA-CPU-adjudication.json',dict(status='PASS_EXISTING_MAIN_QA_GROUP_UNION_NOT_MASTER_MASK',reused_GPU_run=True,extra_GPU_run=False,plan_sha256=bp.sha(file)));continue
    execute_one(plan,directory/st,st,None,1)
   i,x=values(directory/'main-qa',plan);j,y=values(directory/'observation-qa',plan);v2.need(v2.np.array_equal(i,j)and v2.np.array_equal(x.view(v2.np.uint32),y.view(v2.np.uint32)),'Main/observation32 IDs or F32bits changed');write(directory/'QA-acceptance.json',dict(status='PASS_MAIN_OBSERVATION_K10_32',count=320,plan_sha256=bp.sha(file),performance_claim=False))
  else:execute_one(plan,directory/label,a.phase,a.kernel,a.launch_count)
 print(json.dumps(dict(phase=a.phase,plan=str(file),GPU_subtrees_reaped=True,performance_claim=False)))
if __name__=='__main__':main()
