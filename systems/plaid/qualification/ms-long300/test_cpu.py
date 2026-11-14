"""CPU-only preparation/selection/oracle/negative-comparison checks."""
import importlib.util,json,hashlib,tempfile
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('qualification',Path(__file__).with_name('qualify.py'));q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
m=q.inputs();d=np.load(q.DATA/'documents.npy',mmap_mode='r');l=np.load(q.DATA/'doclens.npy');offs=np.r_[0,np.cumsum(l,dtype=np.int64)];rows=np.load(q.DATA/'subset-row-to-source-row.npy');checks=[]
assert d.dtype==np.uint16 and d.shape==(63404,128) and len(l)==1003
for item in m['long_docs']:
 pid=item['subset_row'];assert l[pid]==300 and rows[pid]==item['source_row'];raw=np.array(d[offs[pid]:offs[pid+1]])
 assert hashlib.sha256(raw.tobytes()).hexdigest()==item['fp16_bit_sha256']
checks.append('prepared_shape_mapping_and_all_three_longdoc_fp16_sha')
# Tail-sensitive oracle control; truncations must be distinguishable.
a=np.zeros((300,128),dtype=np.float16);a[:256,1]=1;a[256:,0]=1
chosen=q.choose_tail_query(a,a);assert chosen['full']==32 and chosen['prefix256']==0 and chosen['prefix180']==0
assert chosen['token_index']>=256
checks.append('synthetic_tail_oracle_detects_both_truncations')
with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
 p=Path(tmp);modes=[]
 for i in range(3):
  sub=p/str(i);sub.mkdir();q.save(sub/'results.json',[{'ids':[1,2],'scores':[2.,1.]}]);modes.append(str(sub))
 q.compare(modes,p/'positive.json');q.save(Path(modes[2])/'results.json',[{'ids':[1,2],'scores':[2.,.9]}])
 try:q.compare(modes,p/'negative.json')
 except AssertionError:checks.append('placement_score_tamper_rejected')
 else:raise AssertionError('Tamper accepted')
q.save(Path(__file__).with_name('cpu-tests.json'),{'complete':True,'CUDA_imported':False,'checks':checks});print(checks)
