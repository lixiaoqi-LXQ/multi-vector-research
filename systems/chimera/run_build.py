"""Prepare or execute one immutable Chimera mmap build. No GPU use without --run."""
import os,sys,json,time,struct,subprocess,fcntl,argparse,hashlib,shutil
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from run_search_grid import ROOT,need,sha,cpu_set,index_identity,check_index_unchanged,memory_sample,query_gpu,omp_budget

def readiness(spec):
 root=Path(spec['input']);manifest=root/'manifest.json';problems=[];meta=None;mode=spec.get('input_mode','fp32_bin')
 need(mode in ('fp32_bin','fp16_npy'),'Unknown build input mode')
 if not manifest.exists():problems.append('missing complete input preparation manifest')
 else:
  meta=json.loads(manifest.read_text())
  if meta.get('complete') is not True:problems.append('input preparation is incomplete')
 vector=Path(spec.get('fp16_npy','')) if mode=='fp16_npy' else root/'embeddings.bin'
 for path in (vector,root/'doclens.bin',root/'queries.bin'):
  if not path.is_file():problems.append('missing '+str(path))
 if mode=='fp16_npy' and str(vector).endswith('.partial') and not spec.get('allow_partial',False):problems.append('partial FP16 source requires explicit prefix authorization')
 if problems:return dict(ready=False,input_mode=mode,problems=problems,partial_bytes=(root/'embeddings.bin.partial').stat().st_size if (root/'embeddings.bin.partial').exists() else None)
 n=int(spec['tokens']);d=128
 if mode=='fp32_bin':
  with open(vector,'rb') as f:hn,hd=struct.unpack('<2i',f.read(8))
  need((hn,hd)==(n,d) and vector.stat().st_size==8+n*d*4,'Invalid/incomplete FP32 embedding payload')
 else:
  with open(vector,'rb') as f:
   version=np.lib.format.read_magic(f);need(version in ((1,0),(2,0)),'Unsupported NPY format')
   reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
   shape,fortran,dtype=reader(f);offset=f.tell()
  need(not fortran and len(shape)==2 and shape[0]>=n and shape[1]==d and dtype in (np.dtype('<u2'),np.dtype('<f2')),'Expected C-order raw FP16 bits or FP16 NPY')
  need(offset+n*d*2<=vector.stat().st_size<=offset+shape[0]*d*2,'Required FP16 prefix is incomplete or file oversized')
 with open(root/'queries.bin','rb') as f:nq,ql,qd=struct.unpack('<3i',f.read(12))
 expected_nq=int(spec.get('queries',meta['queries']));need((nq,ql,qd)==(expected_nq,32,128),'Query header mismatch')
 need((root/'queries.bin').stat().st_size==12+nq*ql*qd*4,'Incomplete FP32 query payload')
 lens=np.fromfile(root/'doclens.bin',dtype='<i4')
 need(lens[0]==spec['docs'] and len(lens)==spec['docs']+1 and np.all(lens[1:]>0) and int(lens[1:].sum(dtype=np.int64))==n,'Input document order/length mismatch')
 need(meta['docs']==spec['docs'] and meta['tokens']==n and meta['dimension']==d and meta['queries']==nq,'Input manifest dimensions mismatch')
 source_root=Path(meta['data']);prov=json.loads((source_root/'download-provenance.json').read_text())
 source=dict(repository=prov['repository'],revision=prov['revision'],documents_url=f"https://huggingface.co/datasets/{prov['repository']}/resolve/{prov['revision']}/documents.npy")
 for key,value in spec['source_identity'].items():need(source[key]==value,'Input source mismatch: '+key)
 files={}
 for label,path in [('vector_input',vector),('doclens.bin',root/'doclens.bin'),('queries.bin',root/'queries.bin'),('manifest.json',manifest)]:
  st=path.stat();files[label]=dict(path=str(path.resolve()),size=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,device=st.st_dev)
 return dict(ready=True,input_mode=mode,vector_input=str(vector),conversion_manifest=meta,conversion_manifest_sha256=sha(manifest),source_identity=source,source_provenance=prov,source_provenance_sha256=sha(source_root/'download-provenance.json'),input_files=files)

def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True);p.add_argument('--run',action='store_true');p.add_argument('--reuse',action='store_true');a=p.parse_args();spec=json.loads(Path(a.spec).read_text());ready=readiness(spec)
 print(json.dumps(dict(spec=spec,readiness=ready),indent=2),flush=True)
 if not a.run:return
 need(ready['ready'],'Input conversion not finalized; do not build from .partial')
 need(0<spec['n_clusters']<=spec['tokens'] and spec['ex_bits'] in (3,7),'Invalid build configuration')
 affinity=spec.get('cpu_affinity','0-19');threads=int(spec.get('threads',20));need(threads<=len(cpu_set(affinity)),'Build threads exceed CPU affinity')
 index=Path(spec['index']);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);binary=Path(spec.get('binary',ROOT/'build/gpu_build'))
 if a.reuse and (index/'build-provenance.json').exists():
  old=json.loads((index/'build-provenance.json').read_text());need(old.get('complete') is True,'Existing build incomplete')
  need(all(old[k]==spec[k] for k in ('docs','tokens','n_clusters','ex_bits')) and old['source_identity']==ready['source_identity'],'Existing build identity mismatch')
  identity=index_identity(index);need(identity['files']==old['index_identity']['files'],'Existing index payload mismatch');print('Reused verified '+str(index));return
 need(not index.exists() or not any(index.iterdir()),'Refusing to overwrite existing index artifacts; select another index path')
 frozen=out/'frozen-gpu-build'
 if frozen.exists():need(sha(frozen)==sha(binary),'Existing frozen build executable differs')
 else:shutil.copy2(binary,frozen);frozen.chmod(0o755)
 cmd=['taskset','-c',affinity,str(frozen),'--index',str(index),'--fp16-npy' if ready['input_mode']=='fp16_npy' else '--data',ready['vector_input'],'--doclens',str(Path(spec['input'])/'doclens.bin'),'--n-clusters',str(spec['n_clusters']),'--ex-bits',str(spec['ex_bits'])]
 spill=spec.get('spill_full_codes')
 if spill:
  spill_path=Path(spill).resolve();workspace=Path('/home/lxq/graduate/ai-workspace').resolve()
  need(spill_path.is_relative_to(workspace),'Spill temporary file must be under ai-workspace')
  need(not spill_path.exists(),'Refusing to overwrite existing spill temporary file')
  need(spill_path.parent.is_dir(),'Spill parent directory must already exist')
  cmd+=['--spill-full-codes',str(spill_path)]
 meta=dict(GPU_environment={'CUDA_VISIBLE_DEVICES':'0','UMA_RUN_GPU_TESTS':'1'},spec=spec,docs=spec['docs'],tokens=spec['tokens'],n_clusters=spec['n_clusters'],ex_bits=spec['ex_bits'],total_bits=1+spec['ex_bits'],full_code_build_storage='mmap_shared_spill' if spill else 'owning_vector',spill_temporary_path=str(spill_path) if spill else None,**ready,binary_sha256=sha(frozen),binary_snapshot=str(frozen),runner_sha256=sha(__file__),command=cmd,OpenMP_environment=omp_budget(spec,affinity,threads),complete=False)
 (out/'prepared.json').write_text(json.dumps(meta,indent=2));env=dict(os.environ,LD_LIBRARY_PATH='/home/lxq/miniconda3/envs/cuvs_benchmarks/lib:/usr/local/cuda-13.0/lib64',OMP_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',UMA_RUN_GPU_TESTS='1');env.update(omp_budget(spec,affinity,threads));env.pop('GOMP_CPU_AFFINITY',None)
 lock=open('/tmp/uma-paper-device0.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic()
 try:
  for _ in range(10):
   util=int(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip());need(util==0,'GPU compute nonidle before build');time.sleep(1)
  with open(out/'build.log','w') as f,open(out/'monitor.jsonl','w') as mon:
   proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
   try:
    while proc.poll() is None:
     util,apps=query_gpu();mon.write(json.dumps(dict(time=time.time(),pid=proc.pid,gpu=util.stdout.strip(),apps=apps.stdout.strip(),**memory_sample(proc.pid)))+'\n');mon.flush()
     if time.monotonic()-started>spec.get('timeout_s',21600):raise TimeoutError('Build timed out')
     try:proc.wait(timeout=1)
     except subprocess.TimeoutExpired:pass
   except BaseException:
    if proc.poll() is None:
     proc.terminate()
     try:proc.wait(timeout=20)
     except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
    raise
  need(proc.returncode==0,'Build subprocess failed; preserve log and partial artifacts')
 finally:
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close();(out/'resource-release.json').write_text(json.dumps(dict(flock_released=True,subprocess_exited=('proc' not in locals() or proc.poll() is not None)),indent=2))
 # SHA scanning is outside the GPU lease, so another authorized correctness task can use GPU.
 for name,v in ready['input_files'].items():
  st=Path(v['path']).stat();need((st.st_size,st.st_mtime_ns,st.st_ino,st.st_dev)==(v['size'],v['mtime_ns'],v['inode'],v['device']),'Input changed during mmap build')
 identity=index_identity(index);need((identity['N'],identity['D'],identity['n_clusters'],identity['ex_bits'])==(spec['tokens'],128,spec['n_clusters'],spec['ex_bits']),'Built index header mismatch')
 (index/'artifact-manifest.json').write_text(json.dumps(identity,indent=2));meta.update(complete=True,index_identity=identity,build_and_identity_wall_seconds=time.monotonic()-started)
 (index/'build-provenance.json').write_text(json.dumps(meta,indent=2));(out/'summary.json').write_text(json.dumps(meta,indent=2));print('Completed '+str(index),flush=True)
if __name__=='__main__':main()
