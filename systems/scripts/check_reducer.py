import sys,time,json,pathlib
import numpy as np, torch
from exact_gt import load_prefix
from segmented_maxsim import maxsim

torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
data='/home/lxq/graduate/data/multivector-colbertv2/lotte-pooled-dev'
lens,raw=load_prefix(data,128)
try:
    x=torch.from_numpy(np.array(raw[:len(raw)],dtype=np.float32)).cuda()
    q=np.load(pathlib.Path(data)/'queries.npy')[:7]
    sim=torch.from_numpy(q.copy()).cuda().reshape(-1,128)@x.T
    offsets=np.r_[0,np.cumsum(lens)]
    o=torch.tensor(offsets,device='cuda');lt=torch.tensor(lens,device='cuda')
    a=maxsim(sim,o,len(q),int(lens.max()))
    b=torch.segment_reduce(sim.T.contiguous(),'max',lengths=lt,axis=0).reshape(128,7,32).sum(-1).T
    ref=np.array([[np.max(qq.astype('float64')@np.array(raw[offsets[i]:offsets[i+1]],dtype='float64').T,axis=1).sum() for i in range(128)] for qq in q])
    an=a.cpu().numpy();bn=b.cpu().numpy()
    result={'q':7,'docs':128,'max_error_vs_fp64':float(np.max(np.abs(an-ref))),'max_difference_vs_torch_segment_reduce':float(np.max(np.abs(an-bn))),
      'top10_match_fp64':all(set(np.argsort(-an[i])[:10])==set(np.argsort(-ref[i])[:10]) for i in range(7))}
    assert result['max_error_vs_fp64']<1e-4 and result['top10_match_fp64']
    # Longest/shortest ragged rows and negative scores are covered by the data;
    # independent randomized tensor tests cover non-power-of-two boundaries.
    g=torch.Generator(device='cuda').manual_seed(11)
    for nd in [1,3,19]:
        lens2=np.random.default_rng(nd).integers(1,181,size=nd,dtype=np.int64)
        lens2[0]=180
        nt=int(lens2.sum());nq=5
        s=torch.randn((nq*32,nt),generator=g,device='cuda')
        off=torch.tensor(np.r_[0,np.cumsum(lens2)],device='cuda')
        got=maxsim(s,off,nq,int(lens2.max()))
        r=torch.stack([s[:,int(off[i]):int(off[i+1])].max(1).values.reshape(nq,32).sum(-1) for i in range(nd)],dim=1)
        assert torch.allclose(got,r,atol=1e-4,rtol=1e-6)
    result['ragged_randomized_cases']=3
    p=pathlib.Path('/home/lxq/graduate/ai-workspace/research/multivector/scripts/reducer-validation.json')
    p.write_text(json.dumps(result,indent=2)+'\n');print(result)
finally:raw.close()
