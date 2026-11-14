"""Small command-plan adapter for frozen LoTTE Top10 bottleneck diagnostics; never launches GPU."""
import argparse,hashlib,importlib.util,json,os,re,shutil,sys
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parent
spec=importlib.util.spec_from_file_location('profile_v2_frozen',P/'profile.py');v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)
from run_search_grid_v2 import cpu_set
SECTIONS=Path('/home/lxq/Documents/NVIDIA Nsight Compute/2026.1.0/Sections')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def make(a):
 s=json.loads(Path(a.spec).read_text());v2.need(s['dataset']=='lotte-pooled-dev-full'and s['k']==10 and s['expected_index']['total_bits']==4,'First delivery is fullLoTTE4bit Top10 only');v2.need(len(s['points'])==1 and s.get('selection_frozen_from'),'A single actually frozen fastest calibration point is required')
 # Reuse only canonical cal paths from the old OMP20 entry, then restore the actual selected budget.
 cal=v2.cal_spec(dict(s,threads=20,cpu_affinity='0-19'));cal.update(threads=s['threads'],cpu_affinity=s['cpu_affinity']);omp=v2.omp_budget(cal,cal['cpu_affinity'],cal['threads']);v2.need(0<cal['threads']<=len(cpu_set(cal['cpu_affinity']))and cpu_set(cal['cpu_affinity'])<=set(range(20)),'Selected CPU budget exceeds affinity');meta,identity=v2.accepted_identity(cal)
 out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);old_freeze=json.loads((P/'freeze.json').read_text());v2.need(sha(P/'freeze.json')=='1885cd3f522c130ba1f55fa90f465347385516fbbc768336af2dc5bc8adc7455','Original profile freeze changed')
 for p,h in old_freeze['files'].items():v2.need(sha(p)==h,'Frozen profiling dependency changed: '+p)
 v2.need(sha(v2.OBS)==v2.OBS_SHA and sha(v2.MAIN)==v2.MAIN_SHA,'Frozen search binary changed');shutil.copy2(v2.OBS,out/'observation-search');shutil.copy2(v2.MAIN,out/'main-search')
 np=v2.np;rows=np.load(cal['query_rows'])[:32];np.save(out/'original-query-rows.npy',rows);np.save(out/'gt32.npy',np.load(cal['gt_ids'])[:32])
 # Commands are passed structurally to the existing tree_guard/leaf_exec, not via shell.
 engine=v2.engine_command(cal,out/'observation-search',32,out/'rows.tsv',True);engine[engine.index('--nq')+1]='32';engine[2]=cal['cpu_affinity']
 for key,val in [('--cagra-itopk-size',cal['points'][0].get('cagra_itopk_size',min(cal['points'][0]['nprobe']+32,512))),('--warmup',32)]:v2.need(engine[engine.index(key)+1]==str(val),'Point changed during preparation')
 ns=['/usr/local/cuda/bin/nsys','profile','--trace=cuda-sw,nvtx','--sample=none','--cpuctxsw=none','--cuda-memory-usage=false','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','--output','{phase_dir}/trace','{python}',str(P/'leaf_exec.py'),'{tree_config}']
 sections={'sol':['SpeedOfLight','LaunchStats','Occupancy'],'memory':['SpeedOfLight','MemoryWorkloadAnalysis','LaunchStats','Occupancy'],'compute':['SpeedOfLight','ComputeWorkloadAnalysis','LaunchStats','Occupancy'],'latency':['SpeedOfLight','LaunchStats','Occupancy','InstructionStats']}
 ncu=['/usr/local/cuda/bin/ncu','--config-file','off','--target-processes','all','--profile-from-start','off','--replay-mode','kernel','--cache-control','none','--clock-control','none','--apply-rules','off','--import-sass','off','--rename-kernels','off','--kernel-name-base','demangled','--kernel-name','{exact_kernel_or_regex}','--launch-skip','0','--launch-count','{1_to_3}','--export','{phase_dir}/counters','--section-folder',str(SECTIONS)]
 templates={}
 ncu+=['--metrics','smsp__inst_executed.sum,smsp__inst_issued.sum,lts__t_sectors_op_read.sum.per_second,lts__t_sectors_op_write.sum.per_second']
 for mode,values in sections.items():
  cmd=list(ncu)
  for section in values:cmd+=['--section',section]
  templates[mode]=cmd+['{python}',str(P/'leaf_exec.py'),'{tree_config}']
 env=dict(omp,OMP_NUM_THREADS=str(cal['threads']),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',UMA_RUN_GPU_TESTS='1',LD_LIBRARY_PATH='/home/lxq/miniconda3/envs/cuvs_benchmarks/lib:/usr/local/cuda-13.0/lib64')
 files=[P/'bottleneck_plan.py',P/'run_bottleneck.py',P/'profile.py',ROOT/'run_search_grid_v2.py',ROOT/'search_modes_v2.py',ROOT/'routing_audit.py',ROOT/'search_supervision_v2.py',P/'tree_guard.py',P/'leaf_exec.py',P/'controls/child_supervision.py',P/'controls/child_entry.py',P/'freeze.json',Path(a.spec),out/'observation-search',out/'main-search',out/'original-query-rows.npy',out/'gt32.npy',Path('/usr/local/cuda/bin/ncu').resolve(),Path('/usr/local/cuda/bin/nsys').resolve()]
 for filename in ['SpeedOfLight.section','MemoryWorkloadAnalysis.section','MemoryWorkloadAnalysis_Tables.section','ComputeWorkloadAnalysis.section','LaunchStatistics.section','Occupancy.section','InstructionStatistics.section']:files.append(SECTIONS/filename)
 result=dict(status='CPU_PREPARED_COMMAND_PLAN_NO_GPU',scope='fullLoTTE4bit/B1/K10; frozen best Recall10 point and selected placement',input_spec=str(Path(a.spec).resolve()),input_spec_sha256=sha(a.spec),spec=cal,index_identity=meta,identity=identity,original_query_rows=rows.tolist(),environment=env,engine_command=engine,NSYS=ns,NCU=templates,max_kernel_families=3,max_launches_per_family_per_pass=3,execution_adapter=str(P/'run_bottleneck.py'),supervision={'existing_tree_guard':str(P/'tree_guard.py'),'leaf_exec':str(P/'leaf_exec.py'),'lease':'real global exclusive /tmp/uma-paper-device0.lock held and transferred SCM_RIGHTS to leaf','idle_seconds':10,'timeout_seconds_per_phase':900,'cleanup_seconds':10,'outer_cleanup_seconds':25},NCU_conditions={'replay':'kernel','cache':'none','clock_control':'none','kernel_counter_runtime_is_not_formal_QPS':True},CC121_metric_gap='Static sections filter DRAM Throughput/Frequency/bytes-per-second out of CC121; retain actual missingness, do not equate aggregate MemoryThroughput to LPDDR bandwidth',files={str(x):sha(x)for x in files},GPU_launched=False,performance_claim=False)
 with open(out/'plan.json','x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'plan':str(out/'plan.json'),'sha256':sha(out/'plan.json'),'GPU_launched':False}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',required=True);p.add_argument('--out',required=True);make(p.parse_args())
