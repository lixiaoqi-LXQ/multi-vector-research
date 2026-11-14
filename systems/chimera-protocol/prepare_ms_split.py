"""CPU-only MS split and exact-GT launch specification; never launches GT/download."""
import hashlib,json,struct
from pathlib import Path
import numpy as np
ROOT=Path('/home/lxq/graduate/ai-workspace/research/multivector')
DATA=Path('/home/lxq/graduate/data/multivector-colbertv2')
src=DATA/'msmarco-v1-passage';out=DATA/'splits/msmarco-calibration-seed42';plans=ROOT/'chimera-protocol/msmarco-preparation'
def file_sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
prov=json.loads((src/'download-provenance.json').read_text());assert prov['repository']=='tuskanny/ms_marco_colbertv2' and prov['revision']=='f8540899613ccc13049ac87db1cc4041a1dc4e31'
verified={}
for n in ['queries.npy','queries_ids.npy','doclens.npy','doc_ids.npy']:
 detail=prov['files'][n];assert detail['url']==f"https://huggingface.co/datasets/{prov['repository']}/resolve/{prov['revision']}/{n}"
 measured=file_sha(src/n);assert measured==detail['sha256'];verified[n]=measured
q=np.load(src/'queries.npy',mmap_mode='r',allow_pickle=False);ids=np.load(src/'queries_ids.npy',allow_pickle=False);lens=np.load(src/'doclens.npy',mmap_mode='r',allow_pickle=False)
assert q.shape==(6980,32,128) and q.dtype==np.float32 and np.isfinite(q).all();assert ids.shape==(6980,) and np.unique(ids).size==6980
assert lens.shape==(8841823,) and int(lens.sum(dtype=np.int64))==597909919 and int(lens.min())>0
cal=np.sort(np.random.default_rng(42).choice(6980,100,replace=False)).astype(np.int64);held=np.setdiff1d(np.arange(6980,dtype=np.int64),cal,assume_unique=True)
assert len(held)==6880 and len(np.intersect1d(cal,held))==0 and np.array_equal(np.sort(np.r_[cal,held]),np.arange(6980))
arrays={'calibration-rows.npy':cal,'heldout-rows.npy':held,'all-query-rows.npy':np.arange(6980,dtype=np.int64),'calibration-external-ids.npy':ids[cal],'heldout-external-ids.npy':ids[held]}
out.mkdir(parents=True,exist_ok=True);reused=[]
for n,a in arrays.items():
 p=out/n
 if p.exists():assert np.array_equal(np.load(p,allow_pickle=False),a);reused.append(n)
 else:np.save(p,a,allow_pickle=False)
binbytes=struct.pack('<iii',100,32,128)+np.asarray(q[cal],dtype='<f4').tobytes()
p=out/'queries.bin'
if p.exists():assert p.read_bytes()==binbytes;reused.append(p.name)
else:p.write_bytes(binbytes)
manifest={'schema_version':2,'dataset':'MS MARCO v1 passage ColBERTv2','seed':42,'generator':'NumPy default_rng PCG64; sorted choice(nq,100,replace=False)','source_repository':prov['repository'],'source_revision':prov['revision'],'source_files_sha256':verified,'query_shape':list(q.shape),'query_sha256':array_sha(q),'query_ids_sha256':array_sha(ids),'calibration_query_sha256':array_sha(q[cal]),'calibration_rows_sha256':array_sha(cal),'heldout_rows_sha256':array_sha(held),'calibration_rows':cal.tolist(),'heldout_count':6880,'calibration_count':100,'queries_external_ids_unique':True,'external_qids_are_rows':bool(np.array_equal(ids,np.arange(6980))),'qps_query_rows':'all 6980 original rows','query_hash_definition':'All source queries in original row order; calibration hash covers only selected rows','selection_policy':'Tune only calibration100. Freeze before heldout6880 evaluation. Main QPS uses all6980 queries in original order; three rounds.','artifacts':{n:{'sha256':file_sha(out/n),'bytes':(out/n).stat().st_size} for n in [*arrays,'queries.bin']}}
p=out/'manifest.json'
if p.exists():assert json.loads(p.read_text())==manifest;reused.append(p.name)
else:p.write_text(json.dumps(manifest,indent=2)+'\n')
plans.mkdir(exist_ok=True)
script=ROOT/'scripts/exact_gt.py';gt=DATA/'groundtruth/msmarco-v1-passage-full'
argv=['/home/lxq/graduate/.venv/bin/python',str(script),'--data',str(src),'--docs','8841823','--queries','0','--out',str(gt),'--query-batch','32','--token-block','131072','--k','100','--threads','4','--reducer','triton','--wait-download']
spec={'status':'PREPARED_NOT_LAUNCHED','command_argv':argv,'launch_policy':'Root schedules exclusive GPU window after PLAID formal; do not execute during another measurement. Download remains paused until separately scheduled.','data':str(src),'output':str(gt),'docs':8841823,'tokens':597909919,'queries':6980,'k':100,'split_manifest':str(out/'manifest.json'),'split_manifest_sha256':file_sha(out/'manifest.json'),'representation':'Released HF uint16 bit-pattern view as FP16, exactly expanded to FP32; FP32 GEMM TF32 disabled then segmented max-sum; not original unpublished FP32 embeddings','query_id_policy':'ANN/GT rows use original query-row0..6979; queries_ids.npy external unique qids retained in query_external_ids.npy. Do not substitute qrels labels for exact MaxSim ANN GT.','source_identity':{'repository':prov['repository'],'revision':prov['revision'],'documents_url':prov['files']['documents.npy']['url'],'documents_expected_bytes':153064939392,'documents_expected_sha256':prov['files']['documents.npy']['lfs_sha256']},'script_sha256':{n:file_sha(ROOT/'scripts'/n) for n in ['exact_gt.py','gt_state.py','segmented_maxsim.py']},'checkpoint_policy':'Canonical schema2 atomic checkpoint.npz. On interruption use --resume same_output_directory; never pass a LoTTE checkpoint. Completed output refuses overwrite. Larger MS prefixes may reuse a proven same-corpus/same-query/same-k completed prefix.','postprocess':'After full GT complete and identities validated, slice ids/scores by calibration-rows.npy for cal100 and by heldout-rows.npy for heldout6880. Preserve external IDs in original source order.','source_metadata_verified_sha256':verified,'existing_split_artifacts_reused':reused,'validation':'CPU-only metadata checks and exact_gt.py --help; no GT execution, no download resumed'}
(plans/'exact-gt-full-spec.json').write_text(json.dumps(spec,indent=2)+'\n')
print(json.dumps({'split':str(out),'calibration':len(cal),'heldout':len(held),'calibration_first_rows':cal[:8].tolist(),'first_external_qids':ids[cal[:8]].tolist(),'spec':str(plans/'exact-gt-full-spec.json'),'reused':reused},indent=2))
