"""FP32 IEEE fused MaxSim correctness/speed prototype, independent of canonical GT.

Never casts queries/documents to half, bfloat16 or TF32. Each CTA shares one
packed document across complete 32-token queries, computes IEEE FP32 dot/max/sum,
and writes one score per query/document. No global token-similarity matrix.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _packed_ieee(Q,D,O,S,NQ:tl.constexpr,ND:tl.constexpr,DIM:tl.constexpr,
                 M:tl.constexpr,BD:tl.constexpr,K:tl.constexpr):
    qblocks=tl.cdiv(NQ*32,M)
    pid=tl.program_id(0)
    di=pid//qblocks
    qb=pid%qblocks
    begin=tl.load(O+di).to(tl.int64)
    end=tl.load(O+di+1).to(tl.int64)
    qr=qb*M+tl.arange(0,M)
    kk=tl.arange(0,K)
    q=tl.load(Q+qr[:,None].to(tl.int64)*DIM+kk[None,:],
              mask=(qr[:,None]<NQ*32)&(kk[None,:]<DIM),other=0)
    maxima=tl.full((M,),float('-inf'),tl.float32)
    for tile in range(tl.cdiv(end-begin,BD)):
        dr=begin+tile*BD+tl.arange(0,BD)
        d=tl.load(D+dr[None,:]*DIM+kk[:,None],
                  mask=(dr[None,:]<end)&(kk[:,None]<DIM),other=0)
        sim=tl.dot(q,d,input_precision='ieee',out_dtype=tl.float32)
        # A zero-padded document tail must not win over a negative real score.
        sim=tl.where(dr[None,:]<end,sim,float('-inf'))
        maxima=tl.maximum(maxima,tl.max(sim,axis=1))
    scores=tl.sum(tl.reshape(maxima,(M//32,32)),axis=1)
    qi=qb*(M//32)+tl.arange(0,M//32)
    tl.store(S+qi.to(tl.int64)*ND+di,scores,mask=qi<NQ)


def score_block(q,documents,offsets,m=64,bd=32,warps=8,return_kernel=False):
    if (q.dtype!=torch.float32 or documents.dtype!=torch.float32
            or not q.is_cuda or not documents.is_cuda or offsets.device!=q.device
            or documents.device!=q.device or not q.is_contiguous()
            or not documents.is_contiguous() or not offsets.is_contiguous()):
        raise ValueError('Require contiguous CUDA FP32 Q/D and same-device offsets')
    if (q.ndim!=3 or q.shape[1:]!=(32,128) or documents.ndim!=2
            or documents.shape[1]!=128 or offsets.ndim!=1 or offsets.numel()<2
            or offsets.dtype!=torch.int64 or m not in (32,64) or bd!=32
            or (m,warps) not in ((32,4),(64,8))):
        raise ValueError('Prototype is limited to QL32/D128 and two explicit FP32 tiles')
    nq=q.shape[0];nd=offsets.numel()-1
    scores=torch.empty((nq,nd),device=q.device,dtype=torch.float32)
    kernel=_packed_ieee[(nd*triton.cdiv(nq*32,m),)](
        q,documents,offsets,scores,nq,nd,128,m,bd,128,
        num_warps=warps,num_stages=1)
    return (scores,kernel) if return_kernel else scores


def merge_topk(old_scores,old_ids,scores,doc_start,k):
    take=min(k,scores.shape[1])
    values,ids=scores.topk(take,dim=1)
    values=torch.cat((old_scores,values),dim=1)
    ids=torch.cat((old_ids,ids+doc_start),dim=1)
    values,positions=values.topk(min(k,values.shape[1]),dim=1)
    return values,ids.gather(1,positions)


def resident_topk(q,documents,offsets,k=100,query_batch=64,doc_chunk=2048,
                  m=64,warps=8):
    """Bounded query/document blocks; only document-score blocks are materialized.

    offsets must have positive document lengths, start at zero and end at the
    document tensor length. CPU caller verifies those invariants before upload.
    """
    nq=q.shape[0];nd=offsets.numel()-1
    if not 0<k<=nd or query_batch<=0 or doc_chunk<=0:
        raise ValueError('Invalid top-k/query/document block size')
    all_values=torch.full((nq,k),-torch.inf,device=q.device,dtype=torch.float32)
    all_ids=torch.full((nq,k),-1,device=q.device,dtype=torch.int64)
    for ds in range(0,nd,doc_chunk):
        de=min(ds+doc_chunk,nd)
        # Absolute offsets address the same resident packed FP32 document tensor.
        block_offsets=offsets[ds:de+1].contiguous()
        for qs in range(0,nq,query_batch):
            qe=min(qs+query_batch,nq)
            scores=score_block(q[qs:qe].contiguous(),documents,block_offsets,m=m,warps=warps)
            v,i=merge_topk(all_values[qs:qe],all_ids[qs:qe],scores,ds,k)
            all_values[qs:qe]=v;all_ids[qs:qe]=i
    return all_values,all_ids
