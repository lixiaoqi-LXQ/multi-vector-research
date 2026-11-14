#!/usr/bin/env python3
"""CPU-only metadata preparation; direct FP16 corpus input, never expand corpus."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['CUDA_VISIBLE_DEVICES']='-1'
import hashlib,json,struct,tempfile
from pathlib import Path
import numpy as np
DATA=Path('/home/lxq/graduate/data/multivector-colbertv2');SRC=DATA/'msmarco-v1-passage';OUT=DATA/'chimera-inputs/msmarco-v1-passage-full';PLAN=Path(__file__).parent/'msmarco-preparation'
def need(v,msg):
 if not v:raise ValueError(msg)
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def st(p):
 s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def exclusive_json(p,j):
 with open(p,'x') as f:json.dump(j,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.sched_setaffinity(0,{2})
prov=json.loads((SRC/'download-provenance.json').read_text());rev='f8540899613ccc13049ac87db1cc4041a1dc4e31';expected='3d83999077ca98e3fb24f0bf2b7aa41eeb607f30ecda9d20decf6aa9ed3da690';raw=SRC/'documents.npy';doc=prov['files']['documents.npy'];need(prov['repository']=='tuskanny/ms_marco_colbertv2' and prov['revision']==rev,'Pinned source mismatch');need(doc['sha256']==doc['lfs_sha256']==expected,'Verified full SHA missing')
receipt=Path(doc['verification_evidence']);verified=json.loads(receipt.read_text());need(verified['status']=='VERIFIED_COMPLETE' and verified['actual_sha256']==expected and verified['actual_bytes']==153064939392,'Invalid source verification receipt');need(st(raw)==verified['file_after'],'Raw file stat differs from full verification')
with open(raw,'rb') as f:
 version=np.lib.format.read_magic(f);reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0;shape,order,dtype=reader(f);offset=f.tell()
need(shape==(597909919,128) and not order and dtype==np.dtype('<u2') and raw.stat().st_size==offset+597909919*128*2,'Raw header/size mismatch')
source_hashes={}
for n in ['doclens.npy','queries.npy','queries_ids.npy']:
 actual=sha(SRC/n);need(actual==prov['files'][n]['sha256'],'Source metadata SHA mismatch: '+n);source_hashes[n]=actual
lens=np.load(SRC/'doclens.npy',mmap_mode='r',allow_pickle=False);q=np.load(SRC/'queries.npy',mmap_mode='r',allow_pickle=False);qids=np.load(SRC/'queries_ids.npy',allow_pickle=False)
need(lens.shape==(8841823,) and lens.dtype==np.dtype('<i4') and int(lens.sum(dtype=np.int64))==597909919 and int(lens.min())==4 and int(lens.max())==300,'Doclens mismatch');need(q.shape==(6980,32,128) and q.dtype==np.float32 and np.isfinite(q).all(),'Query shape/dtype/finiteness mismatch');need(qids.shape==(6980,) and len(np.unique(qids))==6980,'External qids mismatch')
source_identity={'repository':prov['repository'],'revision':rev,'documents_url':doc['url']}
base={'schema_version':1,'data':str(SRC),'out':str(OUT),'docs':8841823,'tokens':597909919,'dimension':128,'queries':6980,'query_length':32,'doclen_min':4,'doclen_max':300,'complete':True,'completion_scope':'Chimera metadata inputs only','input_mode':'fp16_npy','fp16_npy':str(raw),'FP32_corpus_expanded':False,'source_precision':'Released FP16 bit patterns passed directly through read-only mmap','source_identity':source_identity,'source_full_sha256':expected,'source_full_verification_receipt':str(receipt),'source_full_verification_receipt_sha256':sha(receipt),'source_array_stat':st(raw),'source_metadata_sha256':source_hashes,'external_query_ids':{'source':str(SRC/'queries_ids.npy'),'source_sha256':source_hashes['queries_ids.npy'],'local_copy':'query_external_ids.npy','ANN_query_identifier':'original row0..6979','external_qids_are_rows':False},'document_ids':{'source':str(SRC/'doc_ids.npy'),'source_sha256':prov['files']['doc_ids.npy']['sha256'],'ANN_document_identifier':'original row0..8841822','verified_source_mapping':'external document IDs equal source rows; metadata-audit.json'},'source_sha256':sha(Path(__file__))}
names=['doclens.bin','queries.bin','query_external_ids.npy']
if OUT.exists():
 existing=json.loads((OUT/'manifest.json').read_text());need(all(existing.get(k)==v for k,v in base.items()),'Existing metadata manifest differs; refusing overwrite')
 for n in names:need(sha(OUT/n)==existing['artifacts'][n]['sha256'],'Existing artifact mismatch: '+n)
 status='VERIFIED_REUSED'
else:
 stage=Path(tempfile.mkdtemp(prefix='msmarco-v1-passage-full.staging-',dir=OUT.parent))
 with open(stage/'doclens.bin','xb') as f:f.write(struct.pack('<i',8841823));lens.tofile(f);f.flush();os.fsync(f.fileno())
 with open(stage/'queries.bin','xb') as f:f.write(struct.pack('<3i',6980,32,128));q.tofile(f);f.flush();os.fsync(f.fileno())
 with open(stage/'query_external_ids.npy','xb') as f:np.save(f,qids,allow_pickle=False);f.flush();os.fsync(f.fileno())
 base['artifacts']={n:{'bytes':(stage/n).stat().st_size,'sha256':sha(stage/n)} for n in names};exclusive_json(stage/'manifest.json',base)
 # Linux exclusive atomic directory publication; no existing dataset material replaced.
 import ctypes
 libc=ctypes.CDLL(None,use_errno=True);fn=libc.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
 if fn(-100,os.fsencode(stage),-100,os.fsencode(OUT),1)!=0:raise OSError(ctypes.get_errno(),'Exclusive metadata publication failed')
 fd=os.open(OUT.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);status='METADATA_ONLY_COMPLETE'
# Read back complete output payloads against original source arrays, without corpus reads.
need((OUT/'doclens.bin').stat().st_size==4+8841823*4 and (OUT/'queries.bin').stat().st_size==12+6980*32*128*4,'Output size mismatch')
readlens=np.memmap(OUT/'doclens.bin',mode='r',dtype='<i4',offset=4,shape=lens.shape);readq=np.memmap(OUT/'queries.bin',mode='r',dtype='<f4',offset=12,shape=q.shape)
need(np.array_equal(readlens,lens) and np.array_equal(readq,q) and np.array_equal(np.load(OUT/'query_external_ids.npy',allow_pickle=False),qids),'Output row/order mismatch');need(not (OUT/'embeddings.bin').exists() and not (OUT/'embeddings.bin.partial').exists(),'Unexpected FP32 corpus output')
PLAN.mkdir(exist_ok=True)
plan={'dataset':'msmarco-v1-passage-full','docs':8841823,'tokens':597909919,'queries':6980,'input':str(OUT),'input_mode':'fp16_npy','fp16_npy':str(raw),'index':str(DATA/'indexes/chimera/msmarco-v1-passage-full-c4194304-b4'),'n_clusters':4194304,'ex_bits':3,'threads':20,'cpu_affinity':'0-19','omp_places':','.join('{'+str(i)+'}' for i in [7,5,6,8,9,15,16,17,18,19,0,1,2,3,4,10,11,12,13,14]),'omp_proc_bind':'true','source_identity':source_identity,'timeout_s':21600,'spill_full_codes':'/home/lxq/graduate/ai-workspace/research/multivector/chimera/tmp/msmarco-full-b4-fullcodes.spill','execution_status':'PREPARED_ONLY_NO_GPU','preparation_manifest_sha256':sha(OUT/'manifest.json'),'source_verification_required':'verified complete original file; linked full-SHA receipt and stat checked','cluster_choice':'2^22 primary near N/143; author README N/200..N/100, exact paper count unavailable'}
derived={'operation':'CPU requantize from fixed 4-bit index to 8-bit','fresh_8bit_build':False,'execution_status':'PLAN_ONLY_NO_REQUANTIZATION','source4_index':plan['index'],'derived8_index':str(DATA/'indexes/chimera/msmarco-v1-passage-full-c4194304-b8-derived'),'raw_fp16_npy':str(raw),'docs':8841823,'tokens':597909919,'dimension':128,'n_clusters':4194304,'ex_bits':7,'total_bits':8,'source_input_mode':'direct FP16 read-only mmap','build_memory':'4bit source build and derived8 use safe MAP_SHARED fullcode spill; no full FP32 corpus expansion','search_load':'--low-memory for both full4/full8; file-backed CPU fullcodes, GPUcluster28N; no zero-copy GPU algorithm change','required_before_derivation':['accepted source4 build receipt + six SHA','qualified frozen requant binary + binding to raw fullSHA','exclusive offline CPU/I/O window','metadata-only adapter publishes truthful derived provenance after revalidation'],'calibration_split':str(DATA/'splits/msmarco-calibration-seed42/manifest.json'),'calibration_queries':100,'heldout_queries':6880,'seed':42,'query_order':'all6980 original rows for formal three-round QPS; external qids preserved separately','exact_GT':{'path':str(DATA/'groundtruth/msmarco-v1-passage-full'),'status':'PENDING_CURRENT_CANONICAL_STREAMING_GT; not assumed complete','launch_spec':str(PLAN/'exact-gt-full-spec.json'),'representation':'released FP16 exactly expanded to FP32 for FP32 MaxSim GT'},'future_derived_spec_fields':json.loads((Path(__file__).parent/'derived-index-metadata/future-derived-spec-fields.json').read_text())}
for name,value in [('build-fp16-b4.json',plan),('derived8-plan.json',derived)]:
 p=PLAN/name
 if p.exists():need(json.loads(p.read_text())==value,'Existing plan differs: '+name)
 else:exclusive_json(p,value)
result={'status':status,'output':str(OUT),'outputs_bytes':sum((OUT/n).stat().st_size for n in names),'full_array_rehashed':False,'FP32_corpus_created':False,'GPU_used':False,'output_rows_equal_source':True,'manifest_sha256':sha(OUT/'manifest.json'),'plan_paths':[str(PLAN/n) for n in ['build-fp16-b4.json','derived8-plan.json']]};print(json.dumps(result,indent=2))
