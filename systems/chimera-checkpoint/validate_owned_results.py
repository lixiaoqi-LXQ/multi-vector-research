#!/usr/bin/env python3
"""Independent CPU audit of actual committed label payloads and saved document files."""
from pathlib import Path
import array,hashlib,json,time
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gpu-owned-device-qualification'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def same(a,b):
 if a.stat().st_size!=b.stat().st_size:return False
 with open(a,'rb') as x,open(b,'rb') as y:
  while True:
   u,v=x.read(8*1024*1024),y.read(8*1024*1024)
   if u!=v:return False
   if not u:return True

def cache(p):
 s=(p/'state.txt').read_text().splitlines();assert s[0]=='CHIMERA_ASSIGNMENT_CHECKPOINT_V1'
 assert list(map(int,s[3].split()))==[1087049,128,8192,3,65536,42,42,1,150]
 done,full,count=s[6].split();assert int(done)==1087049 and int(count)==17 and len(s)==24
 for name,line in [('centroids.f32',4),('assignment.cagra',5)]:
  size,digest=s[line].split();assert (p/name).stat().st_size==int(size) and sha(p/name)==digest
 digest=hashlib.sha256();cursor=0;chunks=[]
 for line in s[7:]:
  start,n,h=line.split();start,n=int(start),int(n);assert start==cursor and n==min(65536,1087049-cursor)
  f=p/'labels'/f'{start}.u32';b=f.read_bytes();labels=array.array('I');labels.frombytes(b)
  assert len(labels)==n and max(labels)<8192 and hashlib.sha256(b).hexdigest()==h
  digest.update(b);chunks.append({'start':start,'count':n,'sha256':h});cursor+=n
 assert cursor==1087049 and digest.hexdigest()==full
 return {'identity_sha256':s[1],'state_sha256':sha(p/'state.txt'),'centroid_sha256':s[4].split()[1],'graph_sha256':s[5].split()[1],'completed_tokens':cursor,'full_labels_sha256':full,'chunks':chunks}
def main():
 p=json.loads((ROOT/'owned-continuation-plan.json').read_text());base=Path(p['base_cache'])
 caches={k:cache(Path(str(base)+'-'+k)) for k in ['A','B']};assert caches['A']==caches['B']
 q=json.loads((OUT/'q1-prefix-validation.json').read_text());assert caches['A']['chunks'][0]['sha256']==q['prefix_sha256']
 indices={}
 for branch in ['A','B']:
  args=p['branches']['branchA' if branch=='A' else 'branchBresume']['command'];indices[branch]=Path(args[args.index('--index')+1])
 indices['fresh4']=Path('/home/lxq/graduate/data/multivector-colbertv2/indexes/chimera/lotte-prefix-10000-c8192-b4-owned-device-qualification')
 indices['fresh8']=Path('/home/lxq/graduate/data/multivector-colbertv2/indexes/chimera/lotte-prefix-10000-c8192-b8-owned-device-qualification')
 names=['doclens.bin','doc_1bit.bin','doc_full.bin','cluster_1bit.bin','ivf.bin','centroids.carga']
 hashes={k:{n:{'bytes':(v/n).stat().st_size,'sha256':sha(v/n)} for n in names} for k,v in indices.items()}
 checks={}
 for other in ['B','fresh4']:
  checks['A_vs_'+other]={n:same(indices['A']/n,indices[other]/n) for n in names}
 for branch in ['A','B']:
  meta=json.loads((indices[branch]/'build-provenance.json').read_text());raw=Path(meta['vector_input']);s=raw.stat();x=meta['input_files']['vector_input']
  assert meta['input_mode']=='fp16_npy' and [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]==[x['device'],x['inode'],x['size'],x['mtime_ns']]
  assert meta['fresh_independent_build_time_measured'] is False and meta['binary_sha256']=='8d95bcf8c3fa9628022620d85649eb6f30598a5830aed470b0b25a55b5e113c5'
 passed=all(checks[pair][n] for pair in checks for n in ['doclens.bin','doc_1bit.bin','doc_full.bin'])
 receipt={'status':'PASS_EXACT_LABELS_AND_REQUIRED_DOCUMENT_PAYLOADS' if passed else 'FAIL_REQUIRED_DOCUMENT_PAYLOAD_MISMATCH','all_committed_prefix_and_full_labels_exact':True,'A_B_caches':caches,'index_paths':{k:str(v) for k,v in indices.items()},'payloads':hashes,'direct_streaming_byte_comparisons':checks,'derived8_source4_metadata_rawstat_compatible':True,'validator_sha256':sha(__file__),'created_at_unix':time.time()}
 target=OUT/'continuation-payload-validation.json';assert not target.exists();target.write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({'status':receipt['status'],'full_labels_sha256':caches['A']['full_labels_sha256'],'byte_comparisons':checks},indent=2));raise SystemExit(0 if passed else 1)
if __name__=='__main__':main()
