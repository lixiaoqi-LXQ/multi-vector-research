"""CPU arithmetic/tolerance oracle; does not replace pending GPU route qualification."""
from pathlib import Path
import json,math
import numpy as np
rng=np.random.default_rng(127)
def norm32(matrix):
 lanes=np.zeros((len(matrix),32),dtype=np.float32)
 for j in range(0,128,32):
  for lane in range(32):
   a=matrix[:,j+lane].astype(np.float64)
   lanes[:,lane]=(a*a+lanes[:,lane].astype(np.float64)).astype(np.float32)
 for offset in [16,8,4,2,1]:
  previous=lanes.copy();lanes[:,:32-offset]=(previous[:,:32-offset]+previous[:,offset:]).astype(np.float32)
 return lanes[:,0]
max_error=0.;cases=0
for rows in [1,2,3,33,129]:
 for queries in [1,2,32]:
  for scale in [.01,1.,8.]:
   x=(rng.normal(size=(rows,128))*scale).astype(np.float32);q=(rng.normal(size=(queries,128))*scale).astype(np.float32)
   if rows>1:x[1]=x[0]
   q[0]=x[0]
   dot=q@x.T;value=np.maximum(np.float32(0),((norm32(q)[:,None]+norm32(x)[None,:])+np.float32(-2)*dot).astype(np.float32))
   xd=x.astype(np.float64);qd=q.astype(np.float64);truth=np.sum((qd[:,None,:]-xd[None,:,:])**2,axis=2)
   tolerance=64*np.finfo(np.float32).eps*(np.sum(qd**2,axis=1)[:,None]+np.sum(xd**2,axis=1)[None,:]+2*np.abs(qd@xd.T)+1)
   error=np.abs(value-truth);assert np.all(error<=tolerance);max_error=max(max_error,float(error.max()))
   for k in sorted({1,min(7,rows),rows}):
    ix=np.argsort(value,axis=1)[:,:k]
    for qi in range(queries):
     cutoff=np.sort(truth[qi])[k-1];chosen=set(map(int,ix[qi]));assert all(truth[qi,c]<=cutoff+2*tolerance[qi,c] for c in chosen)
     assert all(c in chosen for c in range(rows) if truth[qi,c]<cutoff-2*tolerance[qi,c])
   cases+=1
result={'status':'PASS_CPU_EXPANDED_L2_FP64_TOLERANCE_ORACLE','cases':cases,'max_error':max_error,'GPU_used':False,'GPU_route_verified':False,'FP32_policy':'row norms FMA RN and warp tree, FP32 dot; GPU cuBLAS_PEDANTIC and selectK still pending'}
(Path(__file__).parent/'flat-numerics-cpu.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
