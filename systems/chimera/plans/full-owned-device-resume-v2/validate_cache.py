from pathlib import Path
import json,hashlib,time,array,os
P=Path(__file__).resolve().parent;C=Path('/home/lxq/graduate/data/multivector-colbertv2/chimera-build-cache/lotte-pooled-dev-full-c2097152-b4-owned-device-v1');t=time.time();lines=(C/'state.txt').read_text().splitlines();assert lines[0]=='CHIMERA_ASSIGNMENT_CHECKPOINT_V1';assert lines[1]=='4d2d271ae0584a6692ca89e854dcf9119557e4de583f5cbd39046badead0a285';n,d,c,b,batch,*_=map(int,lines[3].split());done,digest,count=lines[6].split();done=int(done);count=int(count);assert (n,d,c,b,batch,done,count)==(266205513,128,2097152,3,65536,130351104,1989);assert digest=='-'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
files=[]
for name,line in zip(('centroids.f32','assignment.cagra'),lines[4:6]):
 size,h=line.split();p=C/name;assert p.stat().st_size==int(size) and sha(p)==h;files.append({'path':str(p),'bytes':int(size),'sha256':h})
rolling=hashlib.sha256();cursor=0;rows=[]
for line in lines[7:]:
 start,length,h=line.split();start=int(start);length=int(length);assert start==cursor and length==min(batch,n-cursor);p=C/'labels'/f'{start}.u32';raw=p.read_bytes();assert len(raw)==length*4 and hashlib.sha256(raw).hexdigest()==h;a=array.array('I');a.frombytes(raw);assert max(a)<c;rolling.update(raw);rows.append({'start':start,'count':length,'sha256':h,'min':min(a),'max':max(a)});cursor+=length
assert cursor==done and len(rows)==count
committed={f'{x["start"]}.u32' for x in rows};orphans=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in (C/'labels').iterdir() if p.name not in committed]
x={'passed':True,'GPU_used':False,'cache':str(C),'identity':lines[1],'state_sha256':sha(C/'state.txt'),'completed_tokens':done,'total_tokens':n,'committed_batches':count,'prefix_concatenated_sha256':rolling.hexdigest(),'matrix_graph':files,'labels':rows,'uncommitted_orphans_preserved':orphans,'elapsed_s':time.time()-t,'interruption_cause':'unknown; original process chain absent without terminal receipt'};(P/'cache-validation.json').write_text(json.dumps(x,indent=2));print(json.dumps({k:v for k,v in x.items() if k!='labels'}))
