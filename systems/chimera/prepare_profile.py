"""Prepare a locked NSYS command; never starts GPU or executes the command."""
import argparse,json,shlex,os
from pathlib import Path
import numpy as np
from run_search_grid import ROOT,sha,index_identity,identity_checks,need,omp_budget

def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True);p.add_argument('--queries',type=int,choices=(32,100),default=32);a=p.parse_args();s=json.loads(Path(a.spec).read_text());need(len(s['points'])==1,'Profile one frozen point')
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);data=Path('/home/lxq/graduate/data/multivector-colbertv2');cal=dict(s)
 cal.update(query=s.get('calibration_query',str(data/'splits/lotte-calibration-seed42/queries.bin')),gt=s.get('calibration_gt',str(Path(s.get('full_gt_ids',s['gt_ids'])).parent/'calibration/groundtruth.tsv')),gt_ids=s.get('calibration_gt_ids',str(Path(s.get('full_gt_ids',s['gt_ids'])).parent/'calibration/ids.npy')),query_rows=s.get('calibration_rows',str(data/'splits/lotte-calibration-seed42/calibration-rows.npy')))
 cal.pop('heldout_rows',None);meta=index_identity(s['index']);with_docs=int(np.fromfile(Path(s['index'])/'doclens.bin',dtype='<i4',count=1)[0]);identity=identity_checks(cal,meta,100,with_docs)
 rows=np.load(cal['query_rows'])[:a.queries];np.save(out/'original-query-rows.npy',rows);np.save(out/'reference-gt-ids.npy',np.load(cal['gt_ids'])[:a.queries]);point=s['points'][0];binary=Path(s.get('binary',ROOT/'build/gpu_search'));affinity=str(s.get('cpu_affinity',s.get('cpu_core',7)));threads=s.get('threads',1)
 cmd=['/usr/local/cuda/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-memory-usage=false','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','--output',str(out/'trace'),'taskset','-c',affinity,str(binary),'--index',s['index'],'--query',cal['query'],'--gt',cal['gt'],'--k',str(s.get('k',100)),'--nq',str(a.queries),'--warmup',str(s.get('warmup',5)),'--nprobe',str(point['nprobe']),'--k-refine',str(point['k_refine']),'--k-full-bit',str(point['k_full_bit']),'--cagra-itopk-size',str(point.get('cagra_itopk_size',min(point['nprobe']+32,544))),'--num-chunks',str(point['num_chunks']),'--profile-range','--output',str(out/'results.tsv')]
 cmd+=['--thread-info']
 if s.get('low_memory',False):cmd+=['--low-memory']
 idle=out/'idle.py';idle.write_text("import subprocess,time\nfor _ in range(10):\n v=int(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip())\n if v:raise RuntimeError('GPU nonidle before profile')\n time.sleep(1)\n")
 env=['env']+[key+'='+value for key,value in omp_budget(s,affinity,threads).items()]+['LD_LIBRARY_PATH=/home/lxq/miniconda3/envs/cuvs_benchmarks/lib:/usr/local/cuda-13.0/lib64',f'OMP_NUM_THREADS={threads}','OPENBLAS_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=0','UMA_RUN_GPU_TESTS=1']
 body=out/'capture.sh';body.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+shlex.join([str(ROOT/'build-env/bin/python'),str(idle)])+'\nexec '+shlex.join(env+cmd)+'\n');body.chmod(0o755)
 launch=out/'run-profile.sh';launch.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.join(['flock','-n','/tmp/uma-paper-device0.lock','bash',str(body)])+'\n');launch.chmod(0o755)
 reports='cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_kern_exec_sum,nvtx_sum'
 stats=['/usr/local/cuda/bin/nsys','stats','--report',reports,'--format','csv','--output',str(out/'stats'),str(out/'trace.nsys-rep')]
 (out/'identity.json').write_text(json.dumps(dict(prepared_only=True,phase='profile_diagnostic_not_QPS',formal_spec=s,index_identity=meta,GT_query_identity=identity,original_query_rows=rows.tolist(),binary_sha256=sha(binary),spec_sha256=sha(a.spec),profile_command=cmd,stats_command=stats,interpretation='NVTX host ranges are nested submission/wait/CPU spans; actual GPU intervals come from CUDA trace. Do not sum overlapping spans. Byte counts are logical memcpy traffic, not physical DRAM traffic.',expected_query_ranges=a.queries,source_qps_evidence=s.get('selection_frozen_from')),indent=2))
 (out/'export-stats.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.join(stats)+'\n')
 print(launch)
if __name__=='__main__':main()
