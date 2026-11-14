"""Exact offline reuse helpers; no search algorithm changes."""
import random
import numpy as np
import torch

def official_heldout_global_rows(lengths,seed=123):
    lengths=np.asarray(lengths,dtype=np.int64);n=len(lengths);docs=sorted(random.Random(seed).sample(range(n),min(n,1+int(16*np.sqrt(120*n)))))
    selected_lengths=lengths[docs];sample_offsets=np.r_[0,np.cumsum(selected_lengths)];sample_count=int(sample_offsets[-1]);heldout_count=min(50000,max(1,sample_count//20))
    torch.manual_seed(seed);positions=torch.randperm(sample_count)[-heldout_count:].numpy()
    selected_doc=np.searchsorted(sample_offsets[1:],positions,side='right');offsets=np.r_[0,np.cumsum(lengths)]
    rows=offsets[np.asarray(docs)[selected_doc]]+positions-sample_offsets[selected_doc]
    return rows,{'sampled_documents':len(docs),'sample_vectors':sample_count,'heldout_vectors':heldout_count,'seed':seed,'method':'exact virtual concatenation+same torch.randperm; materialize selected heldout only'}

def compress_with_existing_codes(codec,embs,codes):
    """Same official half subtraction and binarize, omitting only centroid argmax."""
    assert len(embs)==len(codes)
    output=[]
    for offset in range(0,len(codes),1<<18):
        batch=embs[offset:offset+(1<<18)]
        if codec.use_gpu:batch=batch.cuda().half()
        else:batch=batch.float()
        selected=codes[offset:offset+len(batch)]
        if codec.use_gpu:selected=selected.cuda()
        centers=codec.lookup_centroids(selected,out_device=batch.device)
        residual=batch-centers
        output.append(codec.binarize(residual).cpu())
    return codec.Embeddings(codes.cpu(),torch.cat(output))
