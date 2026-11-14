"""Meaningful CPU-only GT persistence, streaming, resume and arithmetic checks."""
import sys, json, os, tempfile, hashlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/home/lxq/graduate/ai-workspace/research/multivector/scripts')
from exact_gt import load_prefix, make_identity
from gt_state import (array_sha, atomic_write, save_checkpoint, snapshot_meta, read_resume)
root=Path(__file__).parent/'gt-hardening-fixture'
root.mkdir(exist_ok=True)
checks=[]
def reject(fn, message):
    try: fn()
    except (ValueError,FileNotFoundError,RuntimeError) as e:
        assert message in str(e),(message,str(e));return
    raise AssertionError('Expected rejection: '+message)

with tempfile.TemporaryDirectory(dir=root) as td:
 p=Path(td);lens=np.array([1,2,1],np.int32);q=np.arange(2*32*128,dtype=np.float32).reshape(2,32,128)/10000
 np.save(p/'doclens.npy',lens);np.save(p/'doc_ids.npy',np.array(['0','1','2']));np.save(p/'queries_ids.npy',np.array(['0','1']));np.save(p/'queries.npy',q)
 provenance={'repository':'test/source','revision':'a'*40,'files':{}}
 (p/'download-provenance.json').write_text(json.dumps(provenance))
 values=np.linspace(-1,1,4*128,dtype=np.float16).reshape(4,128)
 np.save(p/'documents.npy',values.view(np.uint16))
 reject(lambda:load_prefix(p,0),'positive');reject(lambda:load_prefix(p,4),'exceed')
 selected,corpus=load_prefix(p,3)
 np.save(p/'doc_ids.npy',np.array(['doc-C','doc-A','doc-Z']));np.save(p/'queries_ids.npy',np.array(['qid-19','qid-5']))
 external=make_identity(p,selected,q,2,corpus);assert external['query_ids_sha256']==array_sha(np.array(['qid-19','qid-5']))
 np.save(p/'queries_ids.npy',np.array(['qid-19','qid-19']))
 reject(lambda:make_identity(p,selected,q,2,corpus),'unique')
 np.save(p/'doc_ids.npy',np.array(['0','1','2']));np.save(p/'queries_ids.npy',np.array(['0','1']))
 checks.append('unique external query/document IDs accepted; duplicate external query IDs rejected')
 identity=make_identity(p,selected,q,2,corpus)
 ids=np.array([[0,1],[1,0]],dtype=np.int64);scores=np.array([[2,1],[3,2]],dtype=np.float32)
 meta=snapshot_meta(identity|{'doclens_sha256':array_sha(selected[:2])},ids,scores,2,corpus.hash_prefix(3).hexdigest())
 cp=p/'checkpoint.npz';save_checkpoint(cp,ids,scores,meta)
 ds,i,s,h,m=read_resume(cp,identity,selected,identity['query_sha256'],corpus,2)
 assert ds==2 and np.array_equal(ids,i) and h.hexdigest()==meta['scanned_prefix_sha256'];checks.append('checkpoint resume verifies scanned prefix + IDs/scores')
 old=cp.read_bytes()
 def broken(f):f.write(b'partial');raise RuntimeError('injected crash')
 reject(lambda:atomic_write(cp,broken),'injected crash');assert cp.read_bytes()==old;checks.append('interrupted atomic write preserves previous snapshot')
 reject(lambda:read_resume(cp,identity|{'k':1},selected,identity['query_sha256'],corpus,1),'k');checks.append('resume rejects k identity mismatch')
 altered=selected.copy();altered[0]+=1
 reject(lambda:read_resume(cp,identity,altered,identity['query_sha256'],corpus,2),'length');checks.append('resume rejects changed doclens prefix')
 with open(p/'documents.npy','r+b') as f:f.seek(corpus.offset);b=f.read(2);f.seek(corpus.offset);f.write(b'\x00\x00')
 reject(lambda:read_resume(cp,identity,selected,identity['query_sha256'],corpus,2),'content changed');checks.append('resume rejects modified corpus payload')
 with open(p/'documents.npy','r+b') as f:f.seek(corpus.offset);f.write(b)
 corpus.close()
 # Appended partial input plus same-inode rename remain readable.
 payload=(p/'documents.npy').read_bytes();os.replace(p/'documents.npy',p/'documents.npy.partial')
 (p/'documents.npy.partial').write_bytes(payload[:128+256])
 reject(lambda:load_prefix(p,3),'not fully downloaded')
 selected,stream=load_prefix(p,3,True);assert np.array_equal(stream[:1],values[:1])
 with open(p/'documents.npy.partial','ab') as f:f.write(payload[128+256:])
 os.replace(p/'documents.npy.partial',p/'documents.npy')
 assert np.array_equal(stream[1:4],values[1:]);checks.append('partial append and atomic rename preserve pinned-fd reads')
 provenance['revision']='b'*40;(p/'download-provenance.json').write_text(json.dumps(provenance))
 reject(lambda:stream[:1],'provenance source SHA');checks.append('stream rejects changed source SHA')
 stream.close()

# Match the real tensor reduction layout on CPU and merge nonuniform corpus blocks.
torch.manual_seed(12)
lens=torch.tensor([1,3,7,2,5,1,4,6,2,8,3,1,5,7,2],dtype=torch.int64)
off=torch.cat([torch.zeros(1,dtype=torch.int64),lens.cumsum(0)])
q=torch.randn(3,32,128);x=torch.randn(int(off[-1]),128);k=5
vs=torch.full((3,k),-torch.inf);ids=torch.full((3,k),-1,dtype=torch.int64)
for ds,de in [(0,4),(4,6),(6,13),(13,15)]:
 sim=q.reshape(-1,128)@x[off[ds]:off[de]].T
 seg=torch.segment_reduce(sim.T.contiguous(),reduce='max',lengths=lens[ds:de],axis=0)
 scores=seg.reshape(de-ds,3,32).sum(-1).T.contiguous()
 v,i=scores.topk(min(k,de-ds),dim=1);i+=ds
 allv=torch.cat([vs,v],1);alli=torch.cat([ids,i],1);vs,ni=allv.topk(k,dim=1);ids=alli.gather(1,ni)
full=torch.stack([(q@x[off[d]:off[d+1]].T).amax(-1).sum(-1) for d in range(len(lens))],dim=1)
v,i=full.topk(k,dim=1);assert torch.equal(ids,i);error=float((vs-v).abs().max());assert error<1e-4
checks.append('irregular lengths segmented max/sum and disjoint block topk match direct CPU MaxSim')
report={'device':'CPU','checks':checks,'max_cpu_score_difference':error,'triton_gpu_tested':False}
(Path(__file__).parent/'gt-hardening-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
