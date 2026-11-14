"""FP32 MaxSim reducer for contiguous [nq*32,ntokens] similarities.
Offsets must be strictly increasing document boundaries with offsets[0]=0,
offsets[-1]=ntokens, and each document length <= maxlen. exact_gt validates
those properties on CPU before constructing the CUDA tensor.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _reduce(S, O, OUT, NT: tl.constexpr, ND: tl.constexpr,
            QL: tl.constexpr, BLOCK: tl.constexpr):
    di = tl.program_id(0)
    qi = tl.program_id(1)
    start = tl.load(O + di)
    end = tl.load(O + di + 1)
    rr = tl.arange(0, QL)
    cc = tl.arange(0, BLOCK)
    vals = tl.load(S + (qi * QL + rr[:, None]).to(tl.int64) * NT + start + cc[None, :],
                   mask=start + cc[None, :] < end, other=-float('inf'))
    scores = tl.sum(tl.max(vals, axis=1), axis=0)
    tl.store(OUT + qi * ND + di, scores)


def maxsim(sim, offsets, nq, maxlen=180):
    if (sim.ndim != 2 or sim.dtype != torch.float32 or not sim.is_cuda
            or not sim.is_contiguous() or nq <= 0 or sim.shape[0] != nq * 32):
        raise ValueError('Expected contiguous CUDA FP32 sim[nq*32, ntokens]')
    if (offsets.ndim != 1 or offsets.numel() < 2 or not offsets.is_contiguous()
            or offsets.dtype not in (torch.int32, torch.int64) or offsets.device != sim.device):
        raise ValueError('Expected contiguous integer offsets on the same CUDA device')
    if maxlen <= 0:
        raise ValueError('maxlen must cover every positive document length')
    nd = offsets.numel() - 1
    out = torch.empty((nq, nd), dtype=torch.float32, device=sim.device)
    _reduce[(nd, nq)](sim, offsets, out, sim.shape[1], nd, 32,
                      triton.next_power_of_2(maxlen), num_warps=4)
    return out
