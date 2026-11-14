"""Exhaustive FP32 MaxSim ground truth over complete document blocks.
Corpus prefixes are distinct datasets; no padded/artificial vectors.
"""
import argparse, json, os, time, hashlib
from pathlib import Path
import numpy as np


class StreamingCorpus:
    def __init__(self, root, offset, n):
        self.root,self.offset,self.shape=root,offset,(n,128)
    def __getitem__(self, key):
        assert isinstance(key,slice) and key.step in (None,1)
        start=key.start or 0; stop=key.stop or self.shape[0]
        while True:
            f=self.root/'documents.npy'
            if not f.exists(): f=self.root/'documents.npy.partial'
            required=self.offset+stop*128*2
            if f.stat().st_size>=required: break
            print(json.dumps({'waiting_download_bytes':required,'have':f.stat().st_size}),flush=True)
            time.sleep(10)
        return np.memmap(f,mode='r',offset=self.offset+start*128*2,dtype='<f2',shape=(stop-start,128))


def load_prefix(path, docs, wait_download=False):
    root=Path(path)
    lens=np.load(root/'doclens.npy',allow_pickle=False)[:docs].astype(np.int64)
    n=int(lens.sum())
    f=root/'documents.npy'
    if not f.exists():f=root/'documents.npy.partial'
    with open(f,'rb') as inp:
        v=np.lib.format.read_magic(inp)
        reader = np.lib.format.read_array_header_1_0 if v == (1,0) else np.lib.format.read_array_header_2_0
        shape,fortran,dtype=reader(inp)
        offset=inp.tell()
    assert not fortran and shape[1]==128 and dtype in (np.dtype('<u2'),np.dtype('<f2'))
    required=offset+n*128*2
    if f.stat().st_size<required:
        if wait_download:return lens,StreamingCorpus(root,offset,n)
        raise RuntimeError(f'Need {required} downloaded bytes; have {f.stat().st_size}')
    x=np.memmap(f,mode='r',offset=offset,dtype='<f2',shape=(n,128))
    return lens,x


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--docs',type=int,required=True)
    p.add_argument('--queries',type=int,default=0);p.add_argument('--out',required=True)
    p.add_argument('--query-batch',type=int,default=16);p.add_argument('--token-block',type=int,default=131072)
    p.add_argument('--resume');p.add_argument('--wait-download',action='store_true');p.add_argument('--reducer',choices=['torch','triton'],default='torch')
    p.add_argument('--k',type=int,default=100);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    lens,corpus=load_prefix(a.data,a.docs,a.wait_download)
    q=np.load(Path(a.data)/'queries.npy',allow_pickle=False)
    if a.queries:q=q[:a.queries]
    assert np.isfinite(q).all()
    nq,ql,d=q.shape
    assert ql==32 and d==128
    qt=torch.from_numpy(q.copy()).cuda().float()
    top_scores=torch.full((nq,a.k),-torch.inf,device='cuda')
    top_ids=torch.full((nq,a.k),-1,device='cuda',dtype=torch.long)
    offsets=np.r_[0,np.cumsum(lens)]
    started=time.time();ds=0;blocks=0
    if a.resume:
        prior=Path(a.resume)
        manifest=json.loads((prior/'manifest.json').read_text())
        assert manifest['nq']==nq and manifest['query_sha256']==hashlib.sha256(q.tobytes()).hexdigest()
        ds=manifest['docs']
        assert ds<a.docs
        top_scores.copy_(torch.from_numpy(np.load(prior/'scores.npy')).cuda())
        top_ids.copy_(torch.from_numpy(np.load(prior/'ids.npy')).cuda())
    if a.reducer=='triton':
        from segmented_maxsim import maxsim
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    checkpoint_doc=ds
    while ds<len(lens):
        de=max(ds+1,int(np.searchsorted(offsets,offsets[ds]+a.token_block,side='right')-1))
        de=min(de,len(lens))
        x=torch.from_numpy(np.array(corpus[offsets[ds]:offsets[de]],dtype=np.float32)).cuda()
        lt=torch.from_numpy(lens[ds:de].copy()).cuda()
        local_offsets=torch.tensor(offsets[ds:de+1]-offsets[ds],device='cuda')
        for qs in range(0,nq,a.query_batch):
            qe=min(qs+a.query_batch,nq)
            sim=qt[qs:qe].reshape(-1,d)@x.T
            # Each segment is one document. Max is per query token.
            if a.reducer=='triton':
                scores=maxsim(sim,local_offsets,qe-qs,int(lens[ds:de].max()))
            else:
                seg=torch.segment_reduce(sim.T.contiguous(),reduce='max',lengths=lt,axis=0)
                scores=seg.reshape(de-ds,qe-qs,ql).sum(-1).T.contiguous()
            take=min(a.k,de-ds)
            v,ix=torch.topk(scores,take,dim=1)
            ids=ix+ds
            allv=torch.cat([top_scores[qs:qe],v],dim=1)
            alli=torch.cat([top_ids[qs:qe],ids],dim=1)
            nv,ni=torch.topk(allv,a.k,dim=1)
            top_scores[qs:qe]=nv;top_ids[qs:qe]=alli.gather(1,ni)
        torch.cuda.synchronize()
        blocks+=1;ds=de
        if ds-checkpoint_doc>=10000:
            np.save(out/'checkpoint-ids.npy',top_ids.cpu().numpy())
            np.save(out/'checkpoint-scores.npy',top_scores.cpu().numpy())
            (out/'checkpoint.json').write_text(json.dumps({'docs_done':ds,'blocks':blocks,'elapsed_s':time.time()-started})+'\n')
            checkpoint_doc=ds
        print(json.dumps({'docs_done':ds,'docs_total':len(lens),'blocks':blocks,'elapsed_s':time.time()-started}),flush=True)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ids=top_ids.cpu().numpy();scores=top_scores.cpu().numpy()
    assert (ids>=0).all() and (ids<len(lens)).all() and np.isfinite(scores).all()
    np.save(out/'ids.npy',ids);np.save(out/'scores.npy',scores)
    with open(out/'groundtruth.tsv','w') as f:
        for qi,row in enumerate(ids):
            for rank,di in enumerate(row,1):f.write(f'{qi}\t{di}\t{rank}\n')
    meta=vars(a)|{'complete':True,'nq':nq,'num_tokens':int(lens.sum()),'elapsed_s':time.time()-started,
        'arithmetic':'FP32 GEMM (TF32 disabled), segmented max then FP32 sum',
        'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),
        'query_sha256':hashlib.sha256(q.tobytes()).hexdigest(),
        'doclens_sha256':hashlib.sha256(lens.tobytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':main()
