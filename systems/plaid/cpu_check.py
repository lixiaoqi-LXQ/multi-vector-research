import json
import torch
from colbert.search.strided_tensor import StridedTensor
from colbert.search.index_storage import IndexScorer
from colbert.modeling.colbert import ColBERT

torch.set_num_threads(2)
StridedTensor.try_load_torch_extensions(False)
IndexScorer.try_load_torch_extensions(False)
ColBERT.try_load_torch_extensions(False)
torch.manual_seed(321)
lengths = torch.tensor([2, 5, 1, 7], dtype=torch.int64)
scores = torch.rand(int(lengths.sum()), 32)
actual = ColBERT.segmented_maxsim(scores, lengths)
expected = torch.stack([v.max(0).values.sum() for v in scores.split(lengths.tolist())])
torch.testing.assert_close(actual, expected)
packed = torch.arange(int(lengths.sum()) * 3).view(-1, 3)
st = StridedTensor(packed, lengths, use_gpu=False)
ids = torch.tensor([3, 0, 2], dtype=torch.int32)
result, got_lens = st.lookup(ids)
expected_packed = torch.cat([packed.split(lengths.tolist())[i] for i in ids])
assert torch.equal(result, expected_packed)
assert torch.equal(got_lens, lengths[ids.long()])
print(json.dumps({'cpu_extensions':'passed', 'segmented_maxsim_max_error':(actual-expected).abs().max().item(), 'ragged_lookup':'passed', 'threads':torch.get_num_threads()}))
