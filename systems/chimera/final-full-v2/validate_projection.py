from pathlib import Path
import numpy as np,json,hashlib,struct
D=Path('/home/lxq/graduate/data/multivector-colbertv2');out=[]
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def arr(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
for tag,dataset,cal,nq in [('lotte','lotte-pooled-dev-full','calibration',2931),('msmarco','msmarco-v1-passage-full','calibration-seed42',6980)]:
 gt=D/'groundtruth'/dataset;s=D/'splits'/f'{tag}-calibration-seed42';m=json.loads((s/'manifest.json').read_text());rows=np.load(s/'calibration-rows.npy');ids=np.load(gt/cal/'ids.npy');full=np.load(gt/'ids.npy');assert len(rows)==100 and len(np.unique(rows))==100 and full.shape==(nq,100);assert np.array_equal(ids,full[rows]);assert arr(rows)==m['calibration_rows_sha256']
 q=s/'queries.bin';header=struct.unpack('<3i',q.read_bytes()[:12]);assert header==(100,32,128);payload=np.fromfile(q,dtype='<f4',offset=12).reshape(header);assert arr(payload)==m['calibration_query_sha256'];fq=D/'chimera-inputs'/dataset/'queries.bin';fh=struct.unpack('<3i',fq.read_bytes()[:12]);f=np.memmap(fq,dtype='<f4',offset=12,shape=fh,mode='r');assert np.array_equal(payload,f[rows]);expected=np.sort(np.random.default_rng(42).choice(nq,100,replace=False));assert np.array_equal(rows,expected)
 lines=(gt/cal/'groundtruth.tsv').read_text().splitlines();assert len(lines)==10000
 for i,line in enumerate(lines):
  qid,doc,rank=map(int,line.split());assert qid==i//100 and rank==i%100+1 and doc==ids[qid,rank-1]
 if tag=='msmarco':assert sha(gt/'manifest.json')=='7015ecb4c8d10b882a280c13ef2379414e3cd8e5576486b031bde9dbc4f46ac1';assert np.array_equal(np.load(gt/cal/'query_rows.npy'),rows)
 out.append({'dataset':dataset,'passed':True,'GT_manifest_sha256':sha(gt/'manifest.json'),'cal_ids_sha256':sha(gt/cal/'ids.npy'),'cal_query_sha256':sha(q),'cal_rows_sha256':sha(s/'calibration-rows.npy'),'local_TSV_queries':100,'local_TSV_rows':len(lines),'seed42_rows_exact':True,'query_projection_exact':True,'GT_projection_exact':True})
Path(__file__).with_name('projection-validation.json').write_text(json.dumps({'passed':True,'GPU_used':False,'datasets':out},indent=2));print('Both full calibration projections PASS')
