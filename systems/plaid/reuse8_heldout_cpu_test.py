"""Compare sparse reconstruction with actual concat/permutation reference."""
import random,json
import numpy as np
import torch
from pathlib import Path
from reuse8_support import official_heldout_global_rows
torch.set_num_threads(1);lens=np.random.default_rng(7).integers(1,181,size=100000,dtype=np.int64);offsets=np.r_[0,np.cumsum(lens)];docs=sorted(random.Random(123).sample(range(len(lens)),min(len(lens),1+int(16*np.sqrt(120*len(lens))))))
concat=np.concatenate([np.arange(offsets[d],offsets[d+1]) for d in docs]);torch.manual_seed(123);perm=torch.randperm(len(concat));n=min(50000,max(1,len(concat)//20));reference=concat[perm.numpy()][-n:]
actual,meta=official_heldout_global_rows(lens);assert np.array_equal(reference,actual)
p=Path(__file__).parent/'reuse8-heldout-cpu-check.json';p.write_text(json.dumps({'passed':True,'docs':len(lens),'reference_sample_tokens':len(concat),'heldout_rows':n,'exact_row_order_equal':True,'meta':meta},indent=2));print('VIRTUAL_HELDOUT_EXACT_REFERENCE_PASSED',len(concat),n)
