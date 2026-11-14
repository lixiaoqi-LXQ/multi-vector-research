"""Convert released FP16 bitpatterns to official Chimera FP32 binary layout."""
import argparse,json,hashlib,os
from pathlib import Path
import numpy as np
from exact_gt import load_prefix
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--docs',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--wait-download',action='store_true');p.add_argument('--resume-output',action='store_true')
a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
if (out/'manifest.json').exists():raise RuntimeError('Completed conversion exists; select a new output path')
lens,x=load_prefix(a.data,a.docs,a.wait_download)
partial=out/'embeddings.bin.partial';start=0;sample_count=0
try:
 if partial.exists():
  if not a.resume_output:raise RuntimeError('Partial exists; use --resume-output after validating provenance')
  with open(partial,'rb') as f: header=np.fromfile(f,dtype='<i4',count=2)
  assert np.array_equal(header,np.asarray(x.shape))
  payload=partial.stat().st_size-8;stride=x.shape[1]*4
  assert payload>=0 and payload%stride==0
  start=payload//stride;assert start<=len(x)
  # This session's interrupted append is attested against pinned source samples.
  if start:
   existing=np.memmap(partial,mode='r',dtype='<f4',offset=8,shape=(start,128))
   rows=np.unique(np.r_[0,start-1,np.random.default_rng(42).integers(0,start,size=1024)])
   for row in rows:
    assert np.array_equal(existing[row],np.array(x[int(row):int(row)+1],dtype=np.float32)[0])
   sample_count=len(rows);del existing
 with open(partial,'ab' if start else 'wb') as f:
  if not start:np.asarray(x.shape,dtype='<i4').tofile(f)
  for begin in range(start,len(x),262144):
   end=min(begin+262144,len(x))
   np.asarray(x[begin:end],dtype='<f4').tofile(f)
 assert partial.stat().st_size==8+len(x)*128*4
 os.replace(partial,out/'embeddings.bin')
 with open(out/'doclens.bin','wb') as f:
  np.asarray([len(lens)],dtype='<i4').tofile(f);lens.astype('<i4').tofile(f)
 q=np.load(Path(a.data)/'queries.npy',allow_pickle=False)
 with open(out/'queries.bin','wb') as f:
  np.asarray(q.shape,dtype='<i4').tofile(f);q.astype('<f4').tofile(f)
 m=vars(a)|{'docs':len(lens),'tokens':int(lens.sum()),'queries':len(q),'dimension':128,'source_precision':'released FP16 expanded exactly to FP32','corpus_source':x.identity,'resumed_after_token':start,'resume_sample_rows_checked':sample_count,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'complete':True}
 tmp=out/'manifest.json.partial';tmp.write_text(json.dumps(m,indent=2)+'\n');os.replace(tmp,out/'manifest.json');print(m)
finally:x.close()
