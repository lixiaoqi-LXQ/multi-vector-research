#!/usr/bin/env python3
"""Optional checkpoint builder identity supervisor; --plan never hashes full data or runs GPU."""
import argparse, fcntl, hashlib, json, os, re, struct, subprocess, time
import sys
from supervision import supervise_child

_ACTIVE_RECEIPT = None
from pathlib import Path

PAYLOADS = ('centroids.carga','cluster_1bit.bin','doc_1bit.bin','doc_full.bin','doclens.bin','ivf.bin')
def need(value, message):
    if not value: raise ValueError(message)
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def sha(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def stat(path):
    s=Path(path).stat(); return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns)
def identity(path):
    path=Path(path).resolve(strict=True); before=stat(path); result=dict(path=str(path),**before,sha256=sha(path))
    need(stat(path)==before,'File changed while hashing: '+str(path)); return result
def exclusive_json(path,obj):
    with open(path,'x') as f: json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
def construction_input_provenance(input_mode, input_id, doclens_id):
    need(input_mode in ('fp16-npy','fp32-bin'),'Unknown input mode')
    return {'input_mode':'fp16_npy' if input_mode=='fp16-npy' else 'fp32_bin',
            'vector_input':input_id['path'],
            'input_files':{'vector_input':dict(input_id),'doclens.bin':dict(doclens_id)}}

def declared_derivation_identity(provenance):
    if 'input_derivation_receipt' not in provenance and 'input_derivation_receipt_sha256' not in provenance:
        return None
    need('input_derivation_receipt' in provenance and 'input_derivation_receipt_sha256' in provenance,'Incomplete declared input derivation receipt')
    result=identity(provenance['input_derivation_receipt'])
    need(result['sha256']==provenance['input_derivation_receipt_sha256'],'Declared input derivation receipt SHA mismatch')
    return result

def check_stats(files):
    for item in files: need(stat(item['path'])=={k:item[k] for k in ('device','inode','size','mtime_ns')},'Immutable input changed: '+item['path'])
def source_manifest_identity(path):
    result=identity(path);data=json.loads(Path(path).read_text());root=Path(data['source_root'])
    for name,digest in data['files'].items():need(sha(root/name)==digest,'Source manifest mismatch: '+name)
    return result

def main():
    global _ACTIVE_RECEIPT
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','input','doclens','index','cache','provenance-json','source-manifest','receipt-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--input-mode',choices=['fp16-npy','fp32-bin'],default='fp16-npy');p.add_argument('--clusters',type=int,required=True);p.add_argument('--ex-bits',type=int,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after-batches',type=int,default=0);p.add_argument('--spill-full-codes');p.add_argument('--library-dir',action='append',default=[]);p.add_argument('--extra-identity-file',action='append',default=[])
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--plan',action='store_true');mode.add_argument('--run',action='store_true');a=p.parse_args()
    need(a.clusters>0 and 1<=a.ex_bits<=7 and a.stop_after_batches>=0,'Invalid construction parameters')
    need(not Path(a.index).exists(),'Refusing an existing index output; retain failed outputs and use a fresh index pathname')
    need(Path(a.index).absolute().parent.is_dir(),'Index parent directory must already exist')
    need(not a.spill_full_codes or not Path(a.spill_full_codes).exists(),'Refusing an existing spill file')
    cache=Path(a.cache).absolute();binding_file=Path(str(cache)+'.identity.json')
    need((cache.exists() and binding_file.exists()) if a.resume else (not cache.exists() and not binding_file.exists()),'Cache/binding existence disagrees with explicit resume')
    if a.plan:
        print(json.dumps({'prepared_only':True,'full_data_hashed':False,'GPU_used':False,'args':vars(a)},indent=2));return
    need(os.environ.get('UMA_RUN_GPU_TESTS')=='1','Explicit UMA_RUN_GPU_TESTS=1 required')
    env=os.environ.copy()
    if a.library_dir:env['LD_LIBRARY_PATH']=':'.join(a.library_dir+([env['LD_LIBRARY_PATH']] if env.get('LD_LIBRARY_PATH') else []))
    linked=subprocess.run(['ldd',str(Path(a.binary).resolve(strict=True))],env=env,text=True,capture_output=True,check=True).stdout
    need('not found' not in linked,'Unresolved shared library; pass the original --library-dir paths')
    libs=[]
    for line in linked.splitlines():
        match=re.search(r'(?:=>\s+)?(/[^\s]+)\s+\(',line)
        if match:libs.append(str(Path(match.group(1)).resolve(strict=True)))
    need(any('libcuvs' in name for name in libs),'libcuvs missing from resolved library identity')
    input_id,doc_id,binary_id=identity(a.input),identity(a.doclens),identity(a.binary)
    manifest_id=source_manifest_identity(a.source_manifest);provenance_id=identity(a.provenance_json);prov=json.loads(Path(a.provenance_json).read_text())
    source=prov['source_identity'];need(re.fullmatch('[0-9a-f]{40}',source['revision']) is not None,'Pinned HF revision required')
    need(source['documents_url']==f"https://huggingface.co/datasets/{source['repository']}/resolve/{source['revision']}/documents.npy",'Pinned source URL mismatch')
    need(prov['input_sha256']==input_id['sha256'] and prov['doclens_sha256']==doc_id['sha256'],'Expected input/doclens SHA mismatch')
    need(prov['source_manifest_sha256']==manifest_id['sha256'],'Expected compiler source manifest mismatch')
    derivation_id=declared_derivation_identity(prov)
    library_ids=[identity(name) for name in sorted(set(libs+a.extra_identity_file))]
    guard_id=identity(Path(__file__).with_name('exec_guard'))
    guard_source_id=identity(Path(__file__).with_name('exec_guard.c'))
    supervision_id=identity(Path(__file__).with_name('supervision.py'))
    bindings={'schema':1,'input_mode':a.input_mode,'operation':'original Chimera construction with checkpointed assignment','input':input_id,'doclens':doc_id,'binary':binary_id,'source_manifest':manifest_id,'provenance':provenance_id,'source_identity':source,'libraries':library_ids,'wrapper_sha256':sha(__file__),'environment':{key:env.get(key) for key in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OMP_PLACES','OMP_PROC_BIND','OMP_DYNAMIC','OMP_MAX_ACTIVE_LEVELS','CUDA_DEVICE_MAX_CONNECTIONS','LD_LIBRARY_PATH')},'launcher_cpu_affinity':sorted(os.sched_getaffinity(0)),'parameters':dict(clusters=a.clusters,ex_bits=a.ex_bits,assignment_batch=65536,assignment_k=1,assignment_itopk=150,sample_seed=42,rotator_seed=42,cagra_default_index_params=True)}
    bindings['supervision']={'exec_guard':guard_id,'exec_guard_source':guard_source_id,'module':supervision_id}
    bindings['input_derivation_receipt']=derivation_id
    digest=hashlib.sha256(canonical(bindings)).hexdigest()
    if a.resume:
        existing=json.loads(binding_file.read_text());need(existing['identity_sha256']==digest and existing['bindings']==bindings,'Resume input/binary/library/environment identity changed')
    else:exclusive_json(binding_file,dict(identity_sha256=digest,bindings=bindings))
    receipt=Path(a.receipt_dir);receipt.mkdir(parents=True,exist_ok=False);_ACTIVE_RECEIPT=receipt
    cmd=[str(Path(a.binary).resolve()),'--index',str(Path(a.index).absolute()),'--doclens',doc_id['path'],'--fp16-npy' if a.input_mode=='fp16-npy' else '--data',input_id['path'],'--n-clusters',str(a.clusters),'--ex-bits',str(a.ex_bits),'--build-checkpoint-dir',str(cache),'--build-checkpoint-identity-sha256',digest]
    if a.resume:cmd+=['--resume-build-checkpoint']
    if a.stop_after_batches:cmd+=['--stop-after-assignment-batches',str(a.stop_after_batches)]
    if a.spill_full_codes:cmd+=['--spill-full-codes',str(Path(a.spill_full_codes).absolute())]
    tracked=[input_id,doc_id,binary_id,manifest_id,provenance_id,*library_ids,guard_id,guard_source_id,supervision_id]
    if derivation_id:tracked.append(derivation_id)
    exclusive_json(receipt/'prepared.json',dict(bindings=bindings,identity_sha256=digest,command=cmd,construction_only=True,run_start=time.time()))
    # The coordinator owns the device lease; this wrapper never starts a hidden performance test.
    output_lock=open(str(Path(a.index).absolute())+'.checkpoint-output.lock','a');fcntl.flock(output_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    need(not Path(a.index).exists(),'Index output appeared before launch')
    with open(receipt/'build.log','x') as log:
        supervision=supervise_child(cmd,env=env,log=log,output_lock=output_lock,
            exec_guard=guard_id['path'],receipt_path=receipt/'supervision.json',
            write_receipt=exclusive_json,
            on_started=lambda child:exclusive_json(receipt/'process.json',dict(pid=child.pid,command=cmd,exec_guard=guard_id['path'],output_lock_inherited=True)))
    rc=supervision['returncode']
    if supervision['requested_signal'] is not None:
        exclusive_json(receipt/'result.json',dict(status='interrupted',complete=False,identity_sha256=digest,supervision=supervision))
        raise SystemExit(128+supervision['requested_signal'])
    check_stats(tracked)
    result=dict(returncode=rc,identity_sha256=digest,construction_resumed=a.resume,checkpoint_dir=str(cache),intentional_stop=(rc==3),complete=False)
    if rc==3:
        need(not Path(a.index).exists(),'Intentional stop unexpectedly published final index')
        need((cache/'state.txt').is_file(),'Intentional stop missing committed state')
        result['checkpoint_state_sha256']=sha(cache/'state.txt')
    elif rc==0:
        index=Path(a.index)
        with open(index/'doc_1bit.bin','rb') as header_file:header=struct.unpack('<5Q',header_file.read(40))
        n,d,c,b,pad=header
        with open(a.doclens,'rb') as f:docs=struct.unpack('<i',f.read(4))[0]
        need(c==a.clusters and b==a.ex_bits and d==pad==128,'Final index header mismatch')
        files={}
        for name in PAYLOADS:
            item=identity(index/name);files[name]={k:item[k] for k in ('size','mtime_ns','inode','device','sha256')}
        ix=dict(N=n,D=d,paddedD=pad,n_clusters=c,ex_bits=b,total_bits=b+1,files=files)
        result.update(complete=True,index_identity=ix,source_identity=source,docs=docs,tokens=n,n_clusters=c,ex_bits=b,total_bits=b+1,operation=bindings['operation'],fresh_independent_build_time_measured=False)
        result.update(construction_input_provenance(a.input_mode,input_id,doc_id))
        exclusive_json(index/'build-provenance.json',dict(result,input_identity_sha256=digest,binary_sha256=binary_id['sha256'],source_manifest_sha256=manifest_id['sha256'],assignment_checkpoint_binding=str(binding_file),run_receipt=str(receipt.resolve())))
    exclusive_json(receipt/'result.json',result);print(json.dumps(result,indent=2))
    need(rc in (0,3),'Builder failed; retain cache/logs and inspect before explicit resume')
    if rc==3:raise SystemExit(3)
if __name__=='__main__':
    try:main()
    except SystemExit:raise
    except BaseException as exc:
        if _ACTIVE_RECEIPT is not None:
            try:
                supervision_path=_ACTIVE_RECEIPT/'supervision.json'
                exclusive_json(_ACTIVE_RECEIPT/'wrapper-failed.json',dict(status='failed',complete=False,error_type=type(exc).__name__,error=str(exc),supervision_receipt=str(supervision_path) if supervision_path.exists() else None))
            except Exception as receipt_error:
                print('Could not persist wrapper failure receipt: '+str(receipt_error),file=sys.stderr)
        raise
