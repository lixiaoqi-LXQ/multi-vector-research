"""Real MS doclen300 diagnostic; CPU prepare is independent of CUDA imports.
GPU subcommands require an explicitly authorized root exec_guard/cohort window.
No idle/global-lock exception is granted by this helper.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
R=HERE.parents[1]
SOURCE=Path('/home/lxq/graduate/data/multivector-colbertv2/msmarco-v1-passage')
DATA=Path('/home/lxq/graduate/data/multivector-colbertv2/qualification/plaid-ms-long300-v1')
INDEXROOT=Path('/home/lxq/graduate/data/multivector-colbertv2/indexes')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2))
def inputs():
 m=json.loads((DATA/'manifest.json').read_text())
 for name,expected in m['artifacts_sha256'].items():assert sha(DATA/name)==expected,name
 return m

def prepare():
 assert os.environ.get('UMA_RUN_GPU_TESTS')!='1', 'CPU preparation must not inherit GPU test intent'
 DATA.mkdir(parents=True,exist_ok=False)
 lens=np.load(SOURCE/'doclens.npy',mmap_mode='r');offsets=np.r_[0,np.cumsum(lens,dtype=np.int64)];longrows=np.flatnonzero(lens==300)
 assert longrows.tolist()==[5617986,5617987,5617990]
 rows=np.unique(np.r_[np.arange(1000,dtype=np.int64),longrows]);lengths=np.asarray(lens[rows]);ranges=np.c_[offsets[rows],offsets[rows+1]]
 src=np.load(SOURCE/'documents.npy',mmap_mode='r');assert src.shape==(597909919,128) and src.dtype in (np.uint16,np.float16)
 bits=src.view(np.uint16);dst=np.lib.format.open_memmap(DATA/'documents.npy',mode='w+',dtype=np.uint16,shape=(int(lengths.sum()),128));pos=0;long=[]
 for i,(row,(lo,hi)) in enumerate(zip(rows,ranges)):
  part=np.asarray(bits[lo:hi]);dst[pos:pos+len(part)]=part
  if len(part)==300:long.append({'subset_row':i,'source_row':int(row),'source_token_range':[int(lo),int(hi)],'fp16_bit_sha256':hashlib.sha256(part.tobytes()).hexdigest()})
  pos+=len(part)
 dst.flush();del dst
 np.save(DATA/'doclens.npy',lengths);np.save(DATA/'doc_ids.npy',np.load(SOURCE/'doc_ids.npy',mmap_mode='r')[rows]);np.save(DATA/'subset-row-to-source-row.npy',rows);np.save(DATA/'source-token-ranges.npy',ranges)
 np.save(DATA/'queries.npy',np.load(SOURCE/'queries.npy',mmap_mode='r')[:32]);np.save(DATA/'queries_ids.npy',np.load(SOURCE/'queries_ids.npy',mmap_mode='r')[:32]);np.save(DATA/'query-rows.npy',np.arange(32,dtype=np.int64))
 artifacts={p.name:sha(p) for p in DATA.glob('*.npy')};parent=json.loads((SOURCE/'download-provenance.json').read_text())
 manifest={'schema':1,'kind':'derived real-source doclen300 diagnostic; not original prefix or full benchmark','source':str(SOURCE),'source_repository':parent['repository'],'source_revision':parent['revision'],'parent_provenance_sha256':sha(SOURCE/'download-provenance.json'),'docs':len(rows),'tokens':pos,'dim':128,'max_doclen':int(lengths.max()),'long_docs':long,'original_queries':32,'synthetic_queries_separate':True,'selection':'first1000 source rows plus every doclen300 source row','artifacts_sha256':artifacts,'source_payload_bytes_read':int(pos*128*2)}
 assert manifest['docs']==1003 and pos==63404 and all(np.load(DATA/'doclens.npy')[x['subset_row']]==300 for x in long)
 save(DATA/'manifest.json',manifest);save(HERE/'cpu-prepare-receipt.json',{'complete':True,'data':str(DATA),'manifest_sha256':sha(DATA/'manifest.json'),'manifest':manifest,'CUDA_imported':False})

def verify_frozen():
 f=json.loads((HERE/'source-freeze.json').read_text())
 for path,expected in f['source_sha256'].items():assert sha(Path(path))==expected,'Frozen source changed: '+path
 for path,expected in f['existing_extensions_sha256'].items():assert sha(Path(path))==expected,'Frozen extension changed: '+path
 assert sha(DATA/'manifest.json')==f['data_manifest_sha256']
 return f

def gpu_modules():
 verify_frozen()
 assert os.environ.get('UMA_RUN_GPU_TESTS')=='1','Root GPU slot authorization and UMA_RUN_GPU_TESTS=1 required'
 assert os.environ.get('PLAID_LONG300_AUTHORIZED')=='1','Root must explicitly mark this diagnostic slot'
 assert os.environ.get('OMP_NUM_THREADS')=='4' and os.environ.get('OMP_PLACES')=='{15},{16},{17},{18}'
 sys.path.insert(0,str(R))
 import torch,faiss
 torch.set_num_threads(4);torch.manual_seed(123)
 assert torch.cuda.is_available()
 return torch,faiss

def index(bits):return INDEXROOT/f'plaid-ms-long300-qa-b{bits}-k256-km4'
def build(bits):
 torch,faiss=gpu_modules();m=inputs();out=index(bits);out.mkdir(parents=True,exist_ok=False)
 from colbert.infra import ColBERTConfig
 from colbert.indexing.codecs.residual import ResidualCodec
 from colbert.indexing.collection_indexer import CollectionIndexer
 from colbert.indexing.utils import optimize_ivf
 raw=np.load(DATA/'documents.npy',mmap_mode='r').view(np.float16);emb=torch.from_numpy(np.array(raw));sample=emb[torch.randperm(len(emb))];held_n=min(50000,max(1,len(sample)//20));train,held=sample[:-held_n],sample[-held_n:]
 cfg=ColBERTConfig(dim=128,nbits=bits,gpus=1,kmeans_niters=4);cfg.total_visible_gpus=1
 if bits==4:
  resource=faiss.StandardGpuResources();resource.setTempMemory(128*1024**2);km=faiss.Kmeans(128,256,niter=4,verbose=True,seed=123,gpu=False);km.index=faiss.index_cpu_to_gpu(resource,0,km.index);km.train(train.float().numpy());centers=torch.nn.functional.normalize(torch.from_numpy(km.centroids.copy()),p=2,dim=-1).half();del km,resource
 else:
  assert bits==8 and (index(4)/'build.json').exists();centers=torch.load(index(4)/'centroids.pt',map_location='cpu')
 trainer=object.__new__(CollectionIndexer);trainer.config=cfg;trainer.use_gpu=True;trainer.verbose=0
 cutoffs,weights,avg=trainer._compute_avg_residual(centers.cuda(),held.cuda().half());codec=ResidualCodec(cfg,centers,avg,cutoffs,weights);codec.save(str(out));compressed=codec.compress(emb)
 assert len(compressed.codes)==m['tokens'];compressed.save(str(out/'0'));save(out/'doclens.0.json',np.load(DATA/'doclens.npy').tolist())
 if bits==8:assert torch.equal(compressed.codes,torch.load(index(4)/'0.codes.pt',map_location='cpu')), 'Same centers must produce same codes'
 optimize_ivf(torch.argsort(compressed.codes),torch.bincount(compressed.codes.long(),minlength=256),str(out),verbose=0)
 save(out/'metadata.json',{'config':cfg.export(),'num_chunks':1,'num_partitions':256,'num_embeddings':m['tokens'],'num_documents':m['docs'],'avg_doclen':m['tokens']/m['docs']})
 assert torch.cuda.max_memory_allocated()<2*1024**3
 save(out/'build.json',{'diagnostic':True,'source_manifest_sha256':sha(DATA/'manifest.json'),'bits':bits,'K_override':256,'kmeans_iters':4,'seed':123,'train':len(train),'heldout':len(held),'faiss_temp_mib':128 if bits==4 else 0,'centers_reused_from4':bits==8,'cpu_affinity':sorted(os.sched_getaffinity(0)),'omp_places':os.environ.get('OMP_PLACES'),'torch_peak_allocated':torch.cuda.max_memory_allocated(),'torch_peak_reserved':torch.cuda.max_memory_reserved(),'index_file_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'helper_sha256':sha(Path(__file__)),'source_freeze_sha256':sha(HERE/'source-freeze.json')})
 assert torch.cuda.max_memory_allocated()<2*1024**3

def oracle(q,d):return (d.astype(np.float64)@q.astype(np.float64).T).max(axis=0).sum()
def choose_tail_query(original,decoded):
 candidates=np.r_[original[256:],decoded[256:]].astype(np.float32);candidates/=np.linalg.norm(candidates,axis=1,keepdims=True);candidates=candidates.astype(np.float16)
 # All32 query vectors repeat one direction; compute each candidate dot once.
 scores=decoded.astype(np.float64)@candidates.astype(np.float64).T
 full=scores.max(0)*32;s256=scores[:256].max(0)*32;s180=scores[:180].max(0)*32;gaps=np.minimum(full-s256,full-s180);i=int(gaps.argmax())
 best={'q':np.repeat(candidates[i:i+1],32,axis=0),'full':float(full[i]),'prefix256':float(s256[i]),'prefix180':float(s180[i]),'gap':float(gaps[i]),'candidate':i,'query_source':'original_tail' if i<44 else 'decoded_tail','token_index':256+i%44}
 assert best['gap']>=.1 and best['gap']>5*.02, 'Tail query not discriminative'
 return best

def check(bits,placement,output):
 torch,_=gpu_modules();m=inputs();out=Path(output);out.mkdir(parents=True,exist_ok=False)
 import gc
 import seeded_search_runner as runner
 from types import SimpleNamespace
 from colbert.infra import ColBERTConfig
 scorer=runner.load_scorer(SimpleNamespace(index=str(index(bits)),device='cuda',index_placement=placement));cfg=ColBERTConfig(ncells=256,ndocs=4*m['docs'],centroid_score_threshold=0.,query_maxlen=32,gpus=1);cfg.total_visible_gpus=1
 from colbert.search.strided_tensor import StridedTensor
 padding_calls=[];original_pad=StridedTensor.as_padded_tensor
 def padded_hook(obj,*args,**kwargs):
  padded,mask=original_pad(obj,*args,**kwargs)
  padding_calls.append({'max_length':int(obj.lengths.max()),'max_stride':obj.max_stride,'inner_dims':list(obj.inner_dims),'padded_shape':list(padded.shape),'mask_shape':list(mask.shape),'mask_total':int(mask.sum()),'packed_lengths_total':int(obj.lengths.sum())})
  return padded,mask
 StridedTensor.as_padded_tensor=padded_hook
 lens=np.load(DATA/'doclens.npy');long=[x['subset_row'] for x in m['long_docs']];offsets=np.r_[0,np.cumsum(lens,dtype=np.int64)];calls=[];original_lookup=scorer.lookup_pids
 def hooked(pids,*args,**kwargs):
  ids=pids.detach().cpu().long().tolist() if torch.is_tensor(pids) else list(pids);result=original_lookup(pids,*args,**kwargs)
  lengths=result[1].detach().cpu().numpy();assert lengths.tolist()==lens[ids].tolist()
  calls.append({'pids':ids,'lengths':lengths.tolist(),'packed_count':len(result[0]),'decoded_shape':list(result[0].shape)});assert sum(lengths)==len(result[0]);return result
 scorer.lookup_pids=hooked
 masks={}
 ids=torch.tensor(long,dtype=torch.int32,device='cuda')
 for key in ['codes_strided','residuals_strided']:
  obj=getattr(scorer.embeddings_strided,key);padded,mask=obj.lookup(ids,output='padded');assert mask.sum(1).tolist()==[300]*3 and padded.shape[1]>=300
  masks[key]={'max_stride':obj.max_stride,'strides':obj.strides,'padded_shape':list(padded.shape),'mask_counts':mask.sum(1).tolist()}
 decoded,lengths=original_lookup(ids);assert lengths.tolist()==[300]*3 and tuple(decoded.shape)==(900,128);decoded=decoded.cpu().numpy().reshape(3,300,128)
 # Independent CPU residual decode of every vector, including the final44.
 codes=torch.load(index(bits)/'0.codes.pt',map_location='cpu');residuals=torch.load(index(bits)/'0.residuals.pt',map_location='cpu');centers=scorer.codec.centroids.cpu();weights=scorer.codec.bucket_weights.cpu();rev=scorer.codec.reversed_bit_map.cpu();lut=scorer.codec.decompression_lookup_table.cpu();cpu_decode=[]
 for j,pid in enumerate(long):
  lo,hi=offsets[pid:pid+2];rb=rev[residuals[lo:hi].long()].long();unpacked=lut[rb].reshape(300,128).long();d=centers[codes[lo:hi].long()]+weights[unpacked];d=torch.nn.functional.normalize(d,p=2,dim=-1).half().numpy();assert np.max(np.abs(d.astype(np.float32)-decoded[j].astype(np.float32)))<=.002
  cpu_decode.append({'pid':pid,'vectors':300,'tail_vectors':44,'max_abs_embedding_error':float(np.max(np.abs(d.astype(np.float32)-decoded[j].astype(np.float32))))})
 original_queries=np.load(DATA/'queries.npy');results=[]
 with torch.inference_mode():
  for qi,q in enumerate(original_queries):
   calls.clear();padding_calls.clear();rank=scorer.rank(cfg,torch.from_numpy(q[None].copy()).cuda().half());assert len(rank[0])==m['docs'] and set(rank[0])==set(range(m['docs']));assert np.isfinite(rank[1]).all()
   assert calls and all(pid in calls[-1]['pids'] for pid in long);assert calls[-1]['packed_count']==m['tokens'];assert padding_calls[-1]['max_length']==300 and padding_calls[-1]['mask_total']==m['tokens']
   results.append({'kind':'real_query_rank','query_row':qi,'ids':rank[0],'scores':rank[1],'lookup':calls[-1],'padding':list(padding_calls)})
   calls.clear();padding_calls.clear();forced=scorer.rank(cfg,torch.from_numpy(q[None].copy()).cuda().half(),pids=long);assert len(forced[0])==3;assert padding_calls[-1]['max_length']==300 and padding_calls[-1]['mask_total']==900
   for j,pid in enumerate(long):assert abs(forced[1][forced[0].index(pid)]-oracle(q.astype(np.float16),decoded[j]))<=.02
   results.append({'kind':'real_query_forced','query_row':qi,'ids':forced[0],'scores':forced[1],'lookup':calls[-1],'padding':list(padding_calls)})
  raw=np.load(DATA/'documents.npy',mmap_mode='r').view(np.float16);diagnostics=[]
  for j,pid in enumerate(long):
   lo,hi=offsets[pid:pid+2];best=choose_tail_query(raw[lo:hi],decoded[j]);q=best.pop('q');calls.clear();padding_calls.clear();rank=scorer.rank(cfg,torch.from_numpy(q[None].copy()).cuda(),pids=long);score=rank[1][rank[0].index(pid)]
   assert abs(score-best['full'])<=.02 and abs(score-best['prefix256'])>.1 and abs(score-best['prefix180'])>.1
   token_argmax=(decoded[j].astype(np.float64)@q.astype(np.float64).T).argmax(0);assert int(token_argmax.max())>=256
   results.append({'kind':'tail_diagnostic_forced','target_pid':pid,'ids':rank[0],'scores':rank[1],'lookup':calls[-1],'padding':list(padding_calls)})
   diagnostics.append({'target_pid':pid,**best,'official_score':score,'argmax_token_indices':token_argmax.tolist()});np.save(out/f'diagnostic-query-source-row{m["long_docs"][j]["source_row"]}.npy',q)
 torch.cuda.synchronize();assert torch.cuda.max_memory_allocated()<2*1024**3;save(out/'results.json',results);save(out/'qualification.json',{'complete':True,'scope':'real1003doc doclen300 qualification; no QPS/recall claim','bits':bits,'placement':placement,'manifest_sha256':sha(DATA/'manifest.json'),'helper_sha256':sha(Path(__file__)),'source_freeze_sha256':sha(HERE/'source-freeze.json'),'masks':masks,'independent_cpu_decompress':cpu_decode,'diagnostics':diagnostics,'original_query_count':32,'original_rank_records':32,'forced_real_records':32,'forced_diag_records':3,'all_longdocs_actually_scored':True,'cpu_affinity':sorted(os.sched_getaffinity(0)),'omp_places':os.environ.get('OMP_PLACES'),'torch_peak_allocated':torch.cuda.max_memory_allocated(),'torch_peak_reserved':torch.cuda.max_memory_reserved(),'payload_devices':{'codes':str(scorer.embeddings.codes.device),'residuals':str(scorer.embeddings.residuals.device),'ivf':str(scorer.ivf.tensor.device)}})
 assert torch.cuda.max_memory_allocated()<2*1024**3

def compare(paths,output):
 rows=[json.loads((Path(x)/'results.json').read_text()) for x in paths];assert len(rows)==3 and rows[0]==rows[1]==rows[2], 'Native/codeGPU/allpayload ID/score or lookup mismatch'
 save(Path(output),{'complete':True,'modes':paths,'all_returned_ids_scores_and_lookup_exact':True,'records_per_mode':len(rows[0]),'real_rank32':True,'forced_real32':True,'tail_forced3':True})

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True);sub.add_parser('prepare')
 b=sub.add_parser('build');b.add_argument('--bits',type=int,choices=[4,8],required=True)
 c=sub.add_parser('check');c.add_argument('--bits',type=int,choices=[4,8],required=True);c.add_argument('--placement',choices=['native','gpu-code-payload','gpu-all-payload'],required=True);c.add_argument('--output',required=True)
 x=sub.add_parser('compare');x.add_argument('--modes',nargs=3,required=True);x.add_argument('--output',required=True)
 a=p.parse_args()
 if a.cmd=='prepare':prepare()
 elif a.cmd=='build':build(a.bits)
 elif a.cmd=='check':check(a.bits,a.placement,a.output)
 else:compare(a.modes,a.output)
if __name__=='__main__':main()
