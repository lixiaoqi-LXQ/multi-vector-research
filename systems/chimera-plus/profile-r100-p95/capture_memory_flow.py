"""Small, same-binary R100 p95 diagnostics under the shared device lease."""
from pathlib import Path
import json,os,sys,time,hashlib,fcntl,subprocess,uuid
R=Path('/home/lxq/graduate/ai-workspace/research/multivector');P=R/'chimera-plus/profile-r100-p95';G=R/'chimera/profile-v2';phase=sys.argv[1];assert phase in ['memflow']
B=Path('/home/lxq/graduate/ai-workspace/builds/chimera-plus/gpu_search');expected=json.loads((R/'chimera-plus/implementation-identity.json').read_text())['binary_sha256'];assert hashlib.sha256(B.read_bytes()).hexdigest()==expected
out=P/phase;out.mkdir(exist_ok=False);spec=json.loads((R/'chimera-plus/r100-p95.json').read_text());env=dict(os.environ,LD_LIBRARY_PATH='/home/lxq/miniconda3/envs/cuvs_benchmarks/lib:/usr/local/cuda-13.0/lib64',CUDA_VISIBLE_DEVICES='0',UMA_RUN_GPU_TESTS='1',OMP_NUM_THREADS='20',OMP_PLACES=spec['omp_places'],OMP_PROC_BIND='true',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');env.pop('GOMP_CPU_AFFINITY',None)
cmd=['taskset','-c','0-19',str(B),'--index',spec['index'],'--query',spec['query'],'--gt',spec['gt'],'--k','100','--nq','32','--warmup','32','--thread-info','--kernel-mode','full','--algorithm-variant','full','--nprobe','256','--k-refine','3000','--k-full-bit','300','--cagra-itopk-size','288','--num-chunks','1','--full-bit-device','gpu','--output',str(out/'rows.tsv'),'--profile-range']
nonce=uuid.uuid4().hex;cfgpath=out/'tree-config.json';cfg=dict(socket_name='chimera-plus-'+nonce,token=nonce,lock_path=str(Path.home()/'device.lock'),engine_command=cmd,engine_binary=str(B),engine_sha256=expected,leaf_receipt=str(out/'leaf-lease.json'),tree_receipt=str(out/'tree-supervision.json'),timeout_s=240,cleanup_s=10)
leaf=[sys.executable,str(G/'leaf_exec.py'),str(cfgpath)]
if phase=='nsys':launch=['/usr/local/cuda/bin/nsys','profile','--trace=cuda-sw,nvtx','--sample=none','--cpuctxsw=none','--cuda-memory-usage=false','--gpu-metrics-devices=0','--gpu-metrics-frequency=10000','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','--output',str(out/'trace')]+leaf
else:
 metrics=['gpu__time_duration.sum','lts__t_bytes.sum','lts__t_sectors_op_read_lookup_miss.sum','lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum','lts__t_sectors_op_write.sum']
 launch=['/usr/local/cuda/bin/ncu','--target-processes','all','--profile-from-start','off','--launch-count','24','--cache-control','none','--disable-extra-suffixes','--metrics',','.join(metrics),'--force-overwrite','--export',str(out/'counters')]+leaf
cfg['launch_command']=launch;cfgpath.write_text(json.dumps(cfg,indent=2)+'\n')
with open(Path.home()/'device.lock','a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with open(out/'output.log','w') as log,open(out/'monitor.jsonl','w') as mon:
  child=subprocess.Popen([sys.executable,str(G/'tree_guard.py'),str(cfgpath),str(lock.fileno())],pass_fds=(lock.fileno(),),env=env,stdout=log,stderr=subprocess.STDOUT)
  (out/'process.json').write_text(json.dumps({'driver_pid':os.getpid(),'guard_pid':child.pid,'phase':phase,'queries':32,'NCU_launch_count':24,'scope':'one complete query; every GPU kernel; cross-check sequence against existing NSYS query0','performance_claim':False},indent=2)+'\n')
  try:
   while child.poll() is None:
    q=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,power.draw,clocks.sm','--format=csv,noheader'],capture_output=True,text=True);apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True)
    mon.write(json.dumps({'time':time.time(),'gpu':q.stdout.strip(),'gpu_query_rc':q.returncode,'apps':apps.stdout.strip(),'meminfo':{l.split(':')[0]:l.split(':')[1].strip() for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemAvailable:','MemFree:'))}})+'\n');mon.flush();time.sleep(1)
  finally:
   if child.poll() is None:child.terminate();child.wait(timeout=20)
  rc=child.returncode
 tree=json.loads((out/'tree-supervision.json').read_text());assert tree['all_owned_descendants_gone'] and not tree['timed_out']
 fcntl.flock(lock,fcntl.LOCK_UN)
(out/'capture-completion.json').write_text(json.dumps({'returncode':rc,'all_descendants_gone':True,'lease_released':True,'no_formal_QPS':True},indent=2)+'\n');assert rc==0,f'capture failed, see {out}/output.log'
print(phase,'completed, children reaped and lease released')
