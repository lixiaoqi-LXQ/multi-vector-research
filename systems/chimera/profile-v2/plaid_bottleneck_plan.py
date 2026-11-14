"""CPU-only PLAID selection-to-profile command adapter; same endpoint and environment as core10_run."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import bottleneck_plan as bp
P=bp.P;R=P.parent.parent/'plaid'
def read(p):return json.loads(Path(p).read_text())
def guard(plan):
 ix=Path(plan['index']);arts=read(ix/'index-artifact-identity.json');g=read(ix/'index-stat-guard.json');names=set(arts)|{'index-artifact-identity.json','input-identity.json'};actual={}
 for name in names:
  s=(ix/name).stat();actual[name]=dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns)
 assert actual==g['stats']==plan['index_stats'],'PLAID index changed after accepted endpoint/calibration'
 assert bp.sha(ix/'index-artifact-identity.json')==plan['index_artifact_manifest_sha256'];assert bp.sha(Path(plan['groundtruth'])/'manifest.json')==plan['GT_manifest_sha256']
def make(a):
 selection=Path(a.selection).resolve();s=read(selection);assert s['heldout_used_for_selection']is False and s['actual_topk']==10 and s['index_placement']in ['native','gpu-all-payload'];winner=s['selection'][a.target];assert winner is not None and winner['topk']==10 and winner['bitwise_cal_agreement'];ref=Path(winner['path']);cfg=read(ref/'config.json');ident=read(ref/'run-identity.json');contract=read(ref/'contract.json');assert cfg['topk']==10 and cfg['threads']==20 and cfg['index_placement']==s['index_placement']and cfg['query_placement']=='host'
 assert tuple(winner['point'])==(cfg['ncells'],cfg['ndocs'],cfg['threshold']);assert contract['topk']==10 and contract['query_placement']=='host';index=Path(cfg['index']);metadata=read(index/'metadata.json');assert metadata['num_documents']==2428854 and metadata['num_embeddings']==266205513 and metadata['config']['nbits']==4
 rows=np.load(cfg['query_rows']);assert rows.shape==(100,);original=rows[:32];reference=[json.loads(x)for x in(ref/'per-query.jsonl').read_text().splitlines()][:32];assert[x['query_row']for x in reference]==original.tolist()and all(len(x['ids'])==len(x['scores'])==10 for x in reference)
 for path,h in {**ident['source_sha256'],**ident['extension_sha256']}.items():assert bp.sha(path)==h,'Calibrated endpoint source/extension changed'
 out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);np.save(out/'original-query-rows.npy',original)
 with open(out/'reference32.json','x')as f:json.dump(reference,f)
 python=R/'.venv/bin/python';cmd=['taskset','-c','0-19',str(python),str(R/'endpoint_search_runner.py'),'search','--data',cfg['data'],'--index',str(index),'--groundtruth',cfg['groundtruth'],'--device','cuda','--threads','20','--index-placement',cfg['index_placement'],'--query-placement','host','--warmup','32','--query-rows',cfg['query_rows'],'--queries','32','--rounds','1','--topk','10','--ncells',str(cfg['ncells']),'--ndocs',str(cfg['ndocs']),'--threshold',str(cfg['threshold']),'--output',str(out/'endpoint'),'--profile-range','--nvtx-stages']
 env=dict(UMA_RUN_GPU_TESTS='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='20',OMP_THREAD_LIMIT='20',OMP_DYNAMIC='FALSE',OMP_PROC_BIND='true',OMP_PLACES='{7},{5},{6},{8},{9},{15},{16},{17},{18},{19},{0},{1},{2},{3},{4},{10},{11},{12},{13},{14}',OPENBLAS_NUM_THREADS='20',MKL_NUM_THREADS='20',UMA_SEARCH_CPU_BUDGET='0-19',PATH=str(R/'.venv/bin')+os.pathsep+os.environ.get('PATH',''),PYTHONPATH=str(R/'faiss-gpu-env/lib/python3.13/site-packages'),LD_LIBRARY_PATH=str(R/'faiss-gpu-env/lib'),CUDA_HOME='/usr/local/cuda-13.0',TORCH_CUDA_ARCH_LIST='12.1',TORCH_EXTENSIONS_DIR=str(R/'torch-extensions-cuda'))
 nsys=['/usr/local/cuda/bin/nsys','profile','--trace=cuda-sw,nvtx','--sample=none','--cpuctxsw=none','--cuda-memory-usage=false','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','--output','{phase_dir}/trace','{python}',str(P/'leaf_exec.py'),'{tree_config}'];base=['/usr/local/cuda/bin/ncu','--config-file','off','--target-processes','all','--profile-from-start','off','--replay-mode','kernel','--cache-control','none','--clock-control','none','--apply-rules','off','--import-sass','off','--rename-kernels','off','--kernel-name-base','demangled','--kernel-name','{exact_kernel_or_regex}','--launch-skip','0','--launch-count','{1_to_3}','--export','{phase_dir}/counters','--section-folder',str(bp.SECTIONS)];templates={}
 base+=['--metrics','smsp__inst_executed.sum,smsp__inst_issued.sum,lts__t_sectors_op_read.sum.per_second,lts__t_sectors_op_write.sum.per_second']
 for kind,extra in [('sol',[]),('memory',['MemoryWorkloadAnalysis']),('compute',['ComputeWorkloadAnalysis']),('latency',['InstructionStats'])]:
  c=list(base)
  for section in ['SpeedOfLight','LaunchStats','Occupancy',*extra]:c+=['--section',section]
  templates[kind]=c+['{python}',str(P/'leaf_exec.py'),'{tree_config}']
 guards=read(index/'index-stat-guard.json');files=[P/'plaid_bottleneck_plan.py',P/'bottleneck_plan.py',P/'run_bottleneck.py',P/'tree_guard.py',P/'leaf_exec.py',P/'controls/child_supervision.py',P/'controls/child_entry.py',selection,ref/'run-identity.json',ref/'contract.json',ref/'per-query.jsonl',ref/'config.json',out/'reference32.json',out/'original-query-rows.npy',python.resolve(),index/'index-artifact-identity.json',index/'index-stat-guard.json',Path(cfg['groundtruth'])/'manifest.json',Path(cfg['query_rows']),Path('/usr/local/cuda/bin/nsys').resolve(),Path('/usr/local/cuda/bin/ncu').resolve()]
 files += [bp.SECTIONS/(n+'.section')for n in ['SpeedOfLight','LaunchStatistics','Occupancy','MemoryWorkloadAnalysis','MemoryWorkloadAnalysis_Tables','ComputeWorkloadAnalysis','InstructionStatistics']]
 files += [P/'profile.py',bp.ROOT/'run_search_grid_v2.py',bp.ROOT/'search_modes_v2.py',bp.ROOT/'routing_audit.py',bp.ROOT/'search_supervision_v2.py',Path(cfg['data'])/'queries.npy',Path(cfg['data'])/'queries_ids.npy',Path(cfg['data'])/'doclens.npy']
 frozen={str(f):bp.sha(f)for f in files};frozen.update(ident['source_sha256']);frozen.update(ident['extension_sha256'])
 plan=dict(status='CPU_PREPARED_PLAID_K10_PROFILE_NO_GPU',engine_kind='plaid',input_selection=str(selection),input_selection_sha256=bp.sha(selection),target=a.target,frozen_point=winner['point'],index=str(index),groundtruth=cfg['groundtruth'],index_stats=guards['stats'],index_artifact_manifest_sha256=contract['index_artifact_manifest_sha256'],GT_manifest_sha256=contract['gt_manifest_sha256'],query_placement='host',index_placement=cfg['index_placement'],original_query_rows=original.tolist(),environment=env,engine_command=cmd,NSYS=nsys,NCU=templates,files=frozen,GPU_launched=False,performance_claim=False)
 guard(plan)
 with open(out/'plan.json','x')as f:json.dump(plan,f,indent=2);f.write('\n')
 print(json.dumps({'plan':str(out/'plan.json'),'sha256':bp.sha(out/'plan.json'),'GPU_launched':False}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',required=True);p.add_argument('--target',choices=['R10>=0.9','R10>=0.95'],required=True);p.add_argument('--out',required=True);make(p.parse_args())
