#!/usr/bin/env python3
"""CPU-only dataset audit and strict B1 result aggregation; needs numpy only."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def summary(a):
    v=np.asarray(a,dtype=np.float32).reshape(-1,a.shape[-1])
    norms=np.linalg.norm(v,axis=1)
    return dict(rows=len(v),finite=bool(np.isfinite(v).all()),norm_min=float(norms.min()),norm_max=float(norms.max()),norm_mean=float(norms.mean()),zero_rows=int(np.count_nonzero(norms==0)),max_abs_norm_error=float(abs(norms-1).max()))

def audit(args):
    root=Path(args.dataset);out={'dataset':str(root.resolve()),'files':{}}
    arrays={}
    for name in ('doclens','doc_ids','queries','queries_ids'):
        f=root/(name+'.npy');a=np.load(f,mmap_mode='r',allow_pickle=False);arrays[name]=a
        out['files'][name]={'shape':list(a.shape),'dtype':str(a.dtype),'sha256':sha(f)}
    lens=arrays['doclens'];ids=arrays['doc_ids'];qids=arrays['queries_ids'];queries=arrays['queries']
    assert lens.ndim==1 and np.issubdtype(lens.dtype,np.integer) and np.all(lens>0)
    assert ids.shape==lens.shape and queries.shape==(len(qids),32,128)
    assert len(set(map(str,qids)))==len(qids)
    # Validate the source IDs in chunks without allocating a 2.4M-string Python set.
    identity=True
    for start in range(0,len(ids),100000):
        identity &= bool(np.array_equal(ids[start:start+100000].astype(np.int64),np.arange(start,min(start+100000,len(ids)))))
    assert identity,'document IDs do not equal corpus row positions'
    out['documents']={'count':len(lens),'tokens':int(lens.sum(dtype=np.int64)),'min_length':int(lens.min()),'max_length':int(lens.max()),'mean_length':float(lens.mean()),'ids_equal_row_positions':identity}
    out['queries']=summary(queries)
    out['queries']['ids_equal_row_positions']=bool(np.array_equal(qids.astype(np.int64),np.arange(len(qids))))
    out['qrels']={};pairs=[]
    for line in (root/'lotte_pooled_qas.search.tsv').read_text().splitlines():
        q,_,d,rel=line.split();pairs.append((q,int(d),int(rel)))
    valid_q=set(map(str,qids));assert all(q in valid_q and 0<=d<len(lens) for q,d,r in pairs)
    out['qrels']={'pairs':len(pairs),'queries':len({q for q,d,r in pairs}),'all_ids_valid':True}
    # Read only complete rows from a growing .partial. np.load would reject its short payload.
    f=root/'documents.npy';partial=False
    if not f.exists():f=root/'documents.npy.partial';partial=True
    with f.open('rb') as handle:
        version=np.lib.format.read_magic(handle)
        shape,fortran,dtype=(np.lib.format.read_array_header_1_0(handle) if version==(1,0) else np.lib.format.read_array_header_2_0(handle))
        offset=handle.tell()
    assert shape==(int(lens.sum(dtype=np.int64)),128) and not fortran
    assert dtype==np.dtype('<u2'),'expected raw uint16 IEEE FP16 bit patterns'
    available=min(shape[0],max(0,(f.stat().st_size-offset)//(128*dtype.itemsize)))
    assert available>0
    a=np.memmap(f,dtype=dtype,mode='r',offset=offset,shape=(available,128)).view(np.float16)
    count=min(args.sample_rows,available);selected=np.unique(np.linspace(0,available-1,count,dtype=np.int64))
    out['document_vectors']={'path':str(f),'partial':partial,'header_shape':list(shape),'header_dtype':str(dtype),'complete_rows_at_snapshot':available,'sample_selection':'evenly spaced over available complete rows','sample':summary(a[selected]),'full_scan':False}
    out['prefixes']={}
    for n in (10000,100000,len(lens)):
        tokens=int(lens[:n].sum(dtype=np.int64));out['prefixes'][str(n)]={'tokens':tokens,'fp16_bytes':tokens*256,'fp32_bytes':tokens*512,'complete_rows_available':tokens<=available,'qrels_in_prefix':sum(d<n for q,d,r in pairs),'queries_with_qrel_in_prefix':len({q for q,d,r in pairs if d<n})}
    assert out['queries']['finite'] and out['document_vectors']['sample']['finite']
    return out

def validate_ids(ids,n,name):
    assert ids.ndim==2 and np.issubdtype(ids.dtype,np.integer),name+' must be integer [Q,K]'
    assert np.all((ids>=0)&(ids<n)),name+' contains invalid IDs'
    assert all(len(np.unique(row))==len(row) for row in ids),name+' contains duplicate IDs'

def aggregate(args):
    gtpath=Path(args.gt);gt=np.load(gtpath,allow_pickle=False);gids=gt['ids'];qids=gt['query_ids'];n=args.documents
    validate_ids(gids,n,'ground truth');k=args.k
    assert len(qids)>0 and len(np.unique(qids))==len(qids) and gids.shape[0]==len(qids),'invalid ground truth query IDs'
    assert 0<k<=gids.shape[1]
    runs=[];reference=None
    required=('dataset_id','config_id','source_sha','binary_sha','timing_boundary','corpus_documents','warmup_queries','batch_size')
    for name in args.runs:
        path=Path(name);r=np.load(path,allow_pickle=False);ids=r['ids'];qs=r['query_ids'];lat=r['latency_ms']
        meta=json.loads(path.with_suffix('.json').read_text())
        assert all(key in meta for key in required),'missing run identity in '+name
        assert meta['corpus_documents']==n and meta['batch_size']==1
        assert meta['ground_truth_sha256']==sha(gtpath),'ground truth identity mismatch'
        identity={key:meta[key] for key in required}
        if reference is None:reference=identity
        assert identity==reference,'rounds use different configuration or timing identity'
        assert np.array_equal(qs,qids),'query IDs or order mismatch'
        assert ids.shape[0]==len(qids) and ids.shape[1]>=k
        validate_ids(ids,n,name)
        assert lat.shape==(len(qids),) and np.isfinite(lat).all() and np.all(lat>0),'invalid or incomplete latencies'
        recalls=np.array([len(set(a[:k]).intersection(b[:k]))/k for a,b in zip(ids,gids)])
        mean=float(lat.mean());runs.append({'file':str(path),'queries':len(qids),'mean_ms':mean,'p50_ms':float(np.quantile(lat,.5)),'p95_ms':float(np.quantile(lat,.95)),'p99_ms':float(np.quantile(lat,.99)),'qps':1000/mean,'recall':float(recalls.mean()),'recall_min':float(recalls.min())})
    assert len(runs)==3,'paper protocol requires exactly three complete rounds'
    return {'identity':reference,'k':k,'ground_truth_sha256':sha(gtpath),'rounds':runs,'mean_round_qps':float(np.mean([r['qps'] for r in runs])),'mean_round_recall':float(np.mean([r['recall'] for r in runs])),'mean_round_mean_ms':float(np.mean([r['mean_ms'] for r in runs])),'mean_round_p99_ms':float(np.mean([r['p99_ms'] for r in runs])),'qps_range':[min(r['qps'] for r in runs),max(r['qps'] for r in runs)],'statistics_note':'mean_round_p99_ms is mean of three within-round p99 values, not pooled p99; range is not a confidence interval'}

p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='command',required=True)
a=sp.add_parser('audit');a.add_argument('--dataset',required=True);a.add_argument('--sample-rows',type=int,default=10000)
b=sp.add_parser('aggregate');b.add_argument('--gt',required=True);b.add_argument('--documents',required=True,type=int);b.add_argument('--k',required=True,type=int);b.add_argument('runs',nargs='+')
p.add_argument('--output');args=p.parse_args();result=audit(args) if args.command=='audit' else aggregate(args)
text=json.dumps(result,ensure_ascii=False,indent=2)
if args.output:Path(args.output).write_text(text+'\n')
else:print(text)
