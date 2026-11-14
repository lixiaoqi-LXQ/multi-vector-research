"""CPU-only full coarse-index acceptance; never starts or signals GPU work."""
from pathlib import Path
import argparse,array,hashlib,json,struct,sys,time
import numpy as np
P=Path(__file__).resolve().parent
N,D,C,DOCS=266205513,128,131072,2428854
PAYLOADS=('centroids.carga','cluster_1bit.bin','doc_1bit.bin','doc_full.bin','doclens.bin','ivf.bin')
def need(value,message):
 if not value:raise ValueError(message)
def sha(path):
 with open(path,'rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stat(path):
 s=Path(path).stat();return {'device':s.st_dev,'inode':s.st_ino,'size':s.st_size,'mtime_ns':s.st_mtime_ns}
def check_ivf(labels,counts,positions,inv,token_positions,document_ids,lens,block=1<<20):
 n=len(labels);need(n>0,'Empty database is not a completed coarse index');need(len(inv)==n and len(token_positions)==n and len(document_ids)==n,'IVF/map length mismatch')
 need(positions[0]==0 and positions[-1]==n and np.all(positions[1:]>=positions[:-1]),'Invalid posting offsets')
 need(np.array_equal(np.diff(positions),counts),'Posting counts differ from all assignment labels')
 expected_doc=np.repeat(np.arange(len(lens),dtype=np.int32),lens);need(len(expected_doc)==n,'Document lengths do not cover the token database')
 seen=np.zeros((n+7)//8,dtype=np.uint8)
 for start in range(0,n,block):
  end=min(start+block,n);tokens=np.asarray(inv[start:end]);need(np.all(tokens<n),'IVF token ID out of range')
  lo=int(np.searchsorted(positions,start,side='right')-1);hi=int(np.searchsorted(positions,end-1,side='right')-1)
  ids=np.arange(lo,hi+1,dtype=np.uint32);lengths=np.minimum(positions[lo+1:hi+2],end)-np.maximum(positions[lo:hi+1],start);clusters=np.repeat(ids,lengths.astype(np.int64))
  need(np.array_equal(labels[tokens],clusters),'IVF token is in a cluster different from its committed assignment')
  need(np.array_equal(token_positions[tokens],np.arange(start,end,dtype=np.uint32)),'Token-to-cluster-position is not the inverse of IVF')
  need(np.array_equal(document_ids[start:end],expected_doc[tokens]),'Clustered document IDs disagree with original document lengths/order')
  np.bitwise_or.at(seen,tokens>>3,(np.uint32(1)<<(tokens&7)).astype(np.uint8))
 need(np.all(seen[:-1]==255) and int(seen[-1])==((1<<(n%8))-1 if n%8 else 255),'IVF does not cover each token exactly once')
 return {'all_tokens_checked':n,'all_labels_match_IVF':True,'inverse_token_map_exact':True,'clustered_document_ids_exact':True,'all_token_IDs_form_a_permutation':True}
def verify():
 started=time.time();s=json.loads((P/'spec.json').read_text());job=json.loads(Path(s['job_state']).read_text());need(job.get('status')=='complete' and job.get('returncode')==0,'Coarse builder is not terminal successful')
 cmd=s['command'];receipt=Path(cmd[cmd.index('--receipt-dir')+1]);result=json.loads((receipt/'result.json').read_text());need(result['complete']is True and result['returncode']==0,'Wrapper has not accepted the saved index')
 for path in [receipt/'supervision.json']:
  q=json.loads(path.read_text());need(q['owned_child_terminal_and_reaped']and q['returncode']==0,'Native was not terminal/reaped successful')
 index=Path(cmd[cmd.index('--index')+1]);cache=Path(cmd[cmd.index('--cache')+1]);build=json.loads((index/'build-provenance.json').read_text());need((build['docs'],build['tokens'],build['n_clusters'],build['ex_bits'])==(DOCS,N,C,3),'Build provenance shape mismatch')
 current={};frozen=build['index_identity']['files']
 for name in PAYLOADS:
  q=stat(index/name);need(q=={k:frozen[name][k]for k in q},'Final payload identity changed after wrapper SHA: '+name);current[name]=dict(q,sha256=frozen[name]['sha256'])
 with open(index/'doc_1bit.bin','rb')as f:need(struct.unpack('<5Q',f.read(40))==(N,D,C,3,D),'Index 4bit/header shape mismatch')
 need((index/'doc_full.bin').stat().st_size==64*N and(index/'cluster_1bit.bin').stat().st_size==28*N,'Serialized quantization payload size mismatch')
 lens=np.fromfile(index/'doclens.bin',dtype='<i4');need(lens[0]==DOCS and len(lens)==DOCS+1 and np.all(lens[1:]>0)and int(lens[1:].sum(dtype=np.int64))==N,'Final document lengths mismatch');need(sha(index/'doclens.bin')==sha(cmd[cmd.index('--doclens')+1]),'Final document lengths differ from input')
 lines=(cache/'state.txt').read_text().splitlines();need(lines[0]=='CHIMERA_ASSIGNMENT_CHECKPOINT_V1','Cache schema mismatch');need(list(map(int,lines[3].split()))==[N,D,C,3,65536,42,42,1,150],'Cache parameters differ from full coarse construction')
 identity=json.loads(Path(str(cache)+'.identity.json').read_text());need(identity['identity_sha256']==lines[1]==result['identity_sha256'],'Cache/source immutable binding mismatch');need(lines[2]==hashlib.sha256(lens[1:].tobytes()).hexdigest(),'Cache doclens payload digest mismatch')
 done,digest,number=lines[6].split();need(int(done)==N and int(number)==(N+65535)//65536 and len(lines)==7+int(number),'Not all assignment batches committed')
 for name,line in [('centroids.f32',4),('assignment.cagra',5)]:
  size,h=lines[line].split();need((cache/name).stat().st_size==int(size)and sha(cache/name)==h,'Checkpoint centroid/graph digest mismatch: '+name)
 labels=np.empty(N,dtype=np.uint32);counts=np.zeros(C,dtype=np.uint64);full=hashlib.sha256();cursor=0
 for line in lines[7:]:
  start,count,h=line.split();start,count=int(start),int(count);need(start==cursor and count==min(65536,N-cursor),'Assignment prefix has gap or wrong final batch')
  payload=(cache/'labels'/f'{start}.u32').read_bytes();need(len(payload)==count*4 and hashlib.sha256(payload).hexdigest()==h,'Assignment label chunk SHA/size mismatch');block=np.frombuffer(payload,dtype='<u4');need(np.all(block<C),'Assignment centroid ID out of range');labels[start:start+count]=block;counts+=np.bincount(block,minlength=C).astype(np.uint64);full.update(payload);cursor+=count
 need(cursor==N and full.hexdigest()==digest,'Full concatenated assignment digest mismatch')
 ivf=index/'ivf.bin';need(ivf.stat().st_size==8+N*4+(C+1)*8,'IVF serialized byte count mismatch')
 with open(ivf,'rb')as f:need(struct.unpack('<Q',f.read(8))[0]==N,'IVF token header mismatch')
 inv=np.memmap(ivf,mode='r',dtype='<u4',offset=8,shape=(N,));positions=np.memmap(ivf,mode='r',dtype='<u8',offset=8+N*4,shape=(C+1,));cluster=index/'cluster_1bit.bin';inverse=np.memmap(cluster,mode='r',dtype='<u4',offset=N*24,shape=(N,));docids=np.memmap(cluster,mode='r',dtype='<i4',offset=N*20,shape=(N,));checks=check_ivf(labels,counts,positions,inv,inverse,docids,lens[1:])
 return {'status':'PASS_FULL_COARSE_LABELS_PAYLOAD_IDENTITIES_AND_IVF','GPU_used':False,'performance_claim':False,'docs':DOCS,'tokens':N,'clusters':C,'total_bits':4,'assignment_chunks':int(number),'full_labels_sha256':digest,'cache_state_sha256':sha(cache/'state.txt'),'all_assignment_labels_range_checked':True,'payload_identities':current,'payload_SHA_source':'actual wrapper sixfile full hashes; unchanged inode/device/size/mtime confirmed independently','IVF_checks':checks,'validator_sha256':sha(__file__),'elapsed_seconds':time.time()-started,'saved_rotated_centroid_semantic_GPU_gate':'still required separately before formal search'}
def cpu_test():
 n=33;c=7;lens=np.array([1,3,7,11,11],dtype=np.int32);labels=np.array([((i*3)%c)for i in range(n)],dtype=np.uint32);counts=np.bincount(labels,minlength=c).astype(np.uint64);positions=np.r_[np.uint64(0),np.cumsum(counts,dtype=np.uint64)];inv=np.argsort(labels,kind='stable').astype(np.uint32);inverse=np.empty(n,np.uint32);inverse[inv]=np.arange(n,dtype=np.uint32);docs=np.repeat(np.arange(len(lens),dtype=np.int32),lens)[inv]
 check_ivf(labels,counts,positions,inv,inverse,docs,lens,block=5);rejected=0
 for which in range(4):
  badinv=inv.copy();badmap=inverse.copy();baddocs=docs.copy();badpos=positions.copy()
  if which==0:badinv[0]=n
  elif which==1:badinv[0]=badinv[1]
  elif which==2:badmap[0]=n
  else:baddocs[0]=99
  try:check_ivf(labels,counts,badpos,badinv,badmap,baddocs,lens,block=5)
  except ValueError:rejected+=1
 need(rejected==4,'Corrupt IVF fixtures were accepted');return {'status':'PASS_CPU_SMALL_IVF_VALIDATOR','valid_cases':1,'corruption_cases_rejected':4,'GPU_used':False}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--cpu-test',action='store_true');a=p.parse_args()
 if a.cpu_test:print(json.dumps(cpu_test()));raise SystemExit
 if not a.run:print(json.dumps({'prepared_only':True,'GPU_used':False,'full_dataset_read':False}));raise SystemExit
 result=verify();target=P/'full-CPU-acceptance.json';need(not target.exists(),'Retain existing acceptance record');target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'tokens':N,'assignment_chunks':result['assignment_chunks']}))
