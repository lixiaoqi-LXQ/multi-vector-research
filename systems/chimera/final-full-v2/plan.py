"""CPU-only template, identity finalization and cal3 selection. Never launches GPU."""
from pathlib import Path
import argparse,json,hashlib,sys,statistics,struct,copy,csv
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;BASE=ROOT.parent;DATA=Path('/home/lxq/graduate/data/multivector-colbertv2');sys.path.insert(0,str(ROOT))
from search_modes_v2 import selection_family,modes,validate_points
from run_search_grid_v2 import identity_checks,index_identity,omp_budget

def need(v,m):
 if not v:raise ValueError(m)
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def key(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
FREEZE=BASE/'chimera-protocol/search-runner-v2/freeze.json'
EXPECTED='64957121b7edd97197802d461fd4f0785d101b69d7ddf382547afb70b1eef18b'
def verify_tools():
 need(sha(FREEZE)==EXPECTED,'Runner freeze changed; review latest contract before regeneration')
 for p,h in read(FREEZE)['files'].items():need(sha(p)==h,'Frozen runner dependency changed: '+p)
 return read(FREEZE)
def generate(out):
 verify_tools();out=Path(out);out.mkdir(exist_ok=False)
 original=read(ROOT/'plans/lotte-pooled-dev-full-c2097152-b4/k100-candidate-witness.json');catalog=[]
 alg=BASE/'chimera-alg-ablation/build/gpu_search';kernel=ROOT/'kernel-ablation/build/gpu_search'
 need(sha(alg)=='c54539b5143f63ad96ab0118f165dded7861d9c5af180767cd74c66d7bd4945d','Main candidate binary changed');need(sha(kernel)=='5e2ad985ec323f9e517b13bf7f298f73a365e8c67bc24a725c0c870f461f76bc','Kernel candidate changed')
 datasets=[('lotte','lotte-pooled-dev-full',266205513,2428854,2931,2097152,'lotte-calibration-seed42','lotte-pooled-dev-full-c2097152-b4-owned-device-checkpoint-v1'),('ms','msmarco-v1-passage-full',597909919,8841823,6980,4194304,'msmarco-calibration-seed42','msmarco-v1-passage-full-c4194304-b4-owned-sorted-gather-v1')]
 for tag,dataset,n,docs,nq,c,split,idx4 in datasets:
  for bits in (4,8):
   index=DATA/'indexes/chimera'/idx4
   if bits==8:index=index.with_name(idx4.replace('-b4-','-b8-')+'-cpu-requantized')
   for k in ((10,100) if bits==4 else (100,)):
    s=copy.deepcopy(original);sp=DATA/'splits'/split;gt=DATA/'groundtruth'/dataset;s.update(dataset=dataset,index=str(index),binary=str(alg),binary_sha256=sha(alg),search_source_manifest_sha256=sha(BASE/'chimera-alg-ablation/source-manifest.json'),expected_queries=nq,nq=100,k=k,warmup=32,rounds=1,threads=20,cpu_affinity='0-19',algorithm_variant='full',kernel_mode='full',low_memory=(tag=='ms' and bits==8),goals=([.90,.95] if bits==4 else [.99]),template_status='AWAITING_ROOT_ACCEPTED_FULL_INDEX',expected_docs=docs)
    s.update(query=str(sp/'queries.bin'),query_rows=str(sp/'calibration-rows.npy'),split_manifest=str(sp/'manifest.json'),heldout_rows_source=str(sp/'heldout-rows.npy'),gt=str(gt/('calibration-seed42' if tag=='ms' else 'calibration')/'groundtruth.tsv'),gt_ids=str(gt/('calibration-seed42' if tag=='ms' else 'calibration')/'ids.npy'),gt_manifest=str(gt/'manifest.json'),full_gt_ids=str(gt/'ids.npy'),full_gt=str(gt/'groundtruth.tsv'),full_query=str(DATA/'chimera-inputs'/dataset/'queries.bin'))
    s['expected_index']=dict(N=n,D=128,paddedD=128,n_clusters=c,ex_bits=bits-1,total_bits=bits);s['runner_freeze_sha256']=EXPECTED;s['selection_policy']='cal100 screening; finalists repeated3 independent processes; all3 quality and median QPS; freeze before full3; heldout never used to select';s['finalizer_requirements']=['root acceptance receipt binding all6 payload SHA','saved rotated dataset audit + published sidecar','complete build provenance','derived8 CPU requant provenance and source4 inheritance','completed GT/cal100 exact projection']
    families=[('main',s)]
    if tag=='lotte' and bits==4 and k==100:
     for mode in ('full','no-lut','no-aggregation','neither'):
      v=copy.deepcopy(s);v.update(binary=str(kernel),binary_sha256=sha(kernel),search_source_manifest_sha256=sha(ROOT/'kernel-ablation/source-manifest.json'),kernel_mode=mode);families.append(('fig8-kernel-'+mode,v))
     for mode in ('full','no-cr','no-ivf-pg','neither'):
      v=copy.deepcopy(s);v['algorithm_variant']=mode
      if mode in ('no-ivf-pg','neither'):v['index']=str(DATA/'indexes/chimera/lotte-pooled-dev-full-c131072-b4-fig8-coarse-v1');v['expected_index']['n_clusters']=131072;v['expected_coarse_clusters']=131072
      if mode in ('no-cr','neither'):v['k_refine_semantics']='ignored positive CLI placeholder; full actual CG document set proceeds to DS'
      families.append(('fig8-alg-'+mode,v))
    for family,v in families:
     name=f'{tag}-b{bits}-k{k}-{family}';points0=read(ROOT/f'plans/lotte-pooled-dev-full-c2097152-b4/k{k}-stage0.json')['points'];witness=read(ROOT/f'plans/lotte-pooled-dev-full-c2097152-b4/k{k}-candidate-witness.json')['points'];stages=[('screen',points0),('candidate-witness',witness),('fullbit-upper',[dict(p,k_full_bit=f) for p in witness[-2:] for f in (10000,30000)])]
     for stage,points in stages:
      q=copy.deepcopy(v);unique=[]
      for p in points:
       p=dict(p);p['k_refine']=min(p['k_refine'],docs);p['k_full_bit']=min(p['k_full_bit'],docs)
       if mode_cr(q):p['k_full_bit']=min(p['k_full_bit'],p['k_refine'])
       else:p['k_refine']=1
       if p not in unique:unique.append(p)
      q.update(points=unique,stage=stage,staging_gate=('first independent cal100 screen' if stage=='screen' else 'run only if target absent or a measured neighborhood needs validation; preserve failed points'))
      validate_points(q,512,docs);file=out/name/(stage+'.template.json');write(file,q);catalog.append({'family':name,'stage':stage,'template':str(file),'index':q['index'],'mode':modes(q),'goals':q['goals']})
 write(out/'catalog.json',{'templates':catalog,'GPU_launched':False,'runner_freeze_sha256':EXPECTED,'status':'templates only; missing index/GT/audit never replaced with dummy identity','execution_sequence':['finalize only root-accepted index and complete GT projection','staged cal100 screen and adequate candidate witness if needed','same-family local refinement if useful','up to3 eligible distinct finalists per target, each3 independent cal100 repetitions','freeze all3-qualified median-QPS winner per target','full queries3 repetitions; report full and heldout quality','record missing targets/heldout failures without heldout tuning'],'controls':['same parameters CPU1/4/10fast/20all cal-only','same parameters chunks1/2/4/6/8/10 cal-only; report actual Khat']})
def mode_cr(s):return s['algorithm_variant'] not in ('no-cr','neither')
def file_stat(p):
 s=Path(p).stat();return {'device':s.st_dev,'inode':s.st_ino,'size':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
def verified_index(accept,index,witness,acceptance_sha):
 names=sorted(accept['payload_sha256']);before={n:file_stat(index/n) for n in names}
 if witness.exists():
  w=read(witness);need(w['acceptance_sha256']==acceptance_sha and w['index']==str(index),'Identity witness acceptance/index changed');need(w['payload_stat']==before,'Accepted index stat/inode/ctime changed; investigate before a new full-hash acceptance');ident=w['index_identity']
 else:
  # Exactly one full read of each file. No second index_identity() hash pass.
  with open(index/'doc_1bit.bin','rb') as f:n,d,c,ex,pad=struct.unpack('<5Q',f.read(40))
  files={}
  for name in names:
   h=sha(index/name);need(h==accept['payload_sha256'][name],'Accepted payload mismatch: '+name);v=before[name];files[name]={k:v[k] for k in ('size','mtime_ns','inode','device')};files[name]['sha256']=h
  need(before=={n:file_stat(index/n) for n in names},'Index changed during first full hash');ident=dict(N=n,D=d,paddedD=pad,n_clusters=c,ex_bits=ex,total_bits=ex+1,files=files)
  payload={'index':str(index),'acceptance_sha256':acceptance_sha,'payload_stat':before,'index_identity':ident,'first_hash_complete':True,'reuse_policy':'exact device/inode/size/mtime/ctime plus pinned acceptance SHA; no repeated payload hash'}
  with witness.open('x') as f:json.dump(payload,f,indent=2)
 for name in names:
  need(ident['files'][name]['sha256']==accept['payload_sha256'][name],'Cached digest differs from root acceptance')
  need(all(ident['files'][name][k]==before[name][k] for k in ('size','mtime_ns','inode','device')),'Cached identity stat changed')
 with open(index/'doc_1bit.bin','rb') as f:header=struct.unpack('<5Q',f.read(40))
 need(header==(ident['N'],ident['D'],ident['n_clusters'],ident['ex_bits'],ident['paddedD']),'Cached index header changed')
 return ident

def finalize(template,acceptance,out):
 verify_tools();s=read(template);a=read(acceptance);index=Path(s['index']);need(a.get('root_accepted') is True and a.get('index')==str(index),'Explicit root-accepted exact full index required');need(set(a['payload_sha256'])=={'centroids.carga','cluster_1bit.bin','doc_1bit.bin','doc_full.bin','doclens.bin','ivf.bin'},'Need six accepted payload SHA')
 witness=Path(str(Path(acceptance).resolve())+'.verified-index.json');ident=verified_index(a,index,witness,sha(acceptance))
 side=index/'ROUTING_DATASET_AUDIT.json';need(sha(side)==a['routing_sidecar_sha256'],'Root-accepted routing sidecar changed');meta=read(side);need(meta.get('status')=='APPROVED_SAVED_ROTATED_DATASET_AUDIT' and meta.get('index')==str(index),'Published routing sidecar not approved for exact index');audit=a['routing_dataset_audit_receipt'];need(meta.get('routing_dataset_audit_receipt')==audit and meta.get('routing_dataset_audit_sha256')==a['routing_dataset_audit_sha256'],'Sidecar/audit binding mismatch');need(sha(audit)==a['routing_dataset_audit_sha256'],'Accepted audit changed');s.update(routing_dataset_audit_receipt=audit,routing_dataset_audit_sha256=a['routing_dataset_audit_sha256'])
 need(sha(index/'build-provenance.json')==a['build_provenance_sha256'],'Accepted provenance changed');build=read(index/'build-provenance.json')
 if s['expected_index']['total_bits']==8:
  need(build.get('fresh_8bit_build') is False,'Full8 must be qualified CPU-derived index');derived_fields=a.get('derived_spec_fields',{});need(set(derived_fields)=={'allow_derived_index','approved_requantize_tool_sha256','approved_requantize_qualification_sha256'},'Exact approved derived fields required');s.update(derived_fields);need(s.get('allow_derived_index') is True,'Explicit derived-index opt-in required')
 need(set(ident['files'])==set(a['payload_sha256']),'Index payload names changed');cache=index/'artifact-manifest.json'
 if cache.exists():need(read(cache)==ident,'Existing artifact manifest differs from accepted identity')
 else:
  with cache.open('x') as f:json.dump(ident,f,indent=2)
 identity_checks(s,ident,100,s['expected_docs']);s.update(template_status='FINALIZED_ROOT_ACCEPTED_INDEX',root_index_acceptance=str(Path(acceptance).resolve()),root_index_acceptance_sha256=sha(acceptance),finalized_index_identity=ident);write(out,s)
def load_runs(paths):
 values=[];family=None;seen=set();goals=None
 for p in map(Path,paths):
  p=p.resolve();need(p not in seen,'Repeated run path');seen.add(p);m=read(p/'prepared.json');r=read(p/'release.json');s=read(p/'summary.json');accepted=read(p/'accepted-summary.json');need(accepted.get('output_contract_passed') is True,'Independent accept_run_v2 output validation required');need(not m['spec'].get('profile_range',False) and m['spec'].get('warmup')==32,'Only warmup32 non-profile calibration');need(m['spec']['nq']==100,'Require actual cal100 spec');need(len(s)==len(m['spec']['points']) and sorted(key(x['point']) for x in s)==sorted(key(x) for x in m['spec']['points']),'Expected point set mismatch');need(all(x.get('actual_search_modes',{}).get('verified') is True for x in s),'Missing actual mode validation');need(m['spec']['phase']=='calibration' and m['queries']==100 and m['spec']['rounds']==1,'Only independent cal100 rounds1');need(r.get('complete') is True and len(s)==m['expected_searches']==r['completed_searches'],'Incomplete cal run');f=selection_family(m)
  if family is None:family=f;goals=m['spec']['goals']
  need(m['spec']['goals']==goals,'Mixed target sets')
  need(f==family,'Family mismatch: mode/binary/index/GT/query/resource/load/warmup');values.append((p,m,s))
 return values,family
def finalists(paths,out):
 runs,family=load_runs(paths);out=Path(out);out.mkdir(exist_ok=False);pool={};goals=runs[0][1]['spec']['goals'];metric='recall'+str(family['k'])
 for p,m,rows in runs:
  for row in rows:
   if row[metric]>=min(goals):pool.setdefault(key(row['point']),[]).append((row,p,m))
 chosen={}
 for goal in goals:
  eligible=[max(v,key=lambda z:z[0]['qps']) for v in pool.values() if any(x[0][metric]>=goal for x in v)]
  for row,p,m in sorted(eligible,key=lambda z:z[0]['qps'],reverse=True)[:3]:chosen[key(row['point'])]=(row,p,m)
 for number,(row,p,m) in enumerate(chosen.values()):
  for repeat in range(3):s=copy.deepcopy(m['spec']);s.update(points=[row['point']],finalist_id=f'f{number:02d}',finalist_repeat=repeat,screening_source=str(p));write(out/f'f{number:02d}-repeat{repeat}.json',s)
 write(out/'screening-ledger.json',{'family':family,'missing_targets':[g for g in goals if not any(x[0][metric]>=g for v in pool.values() for x in v)],'finalists':len(chosen),'GPU_launched':False})
def freeze_final(paths,out):
 runs,family=load_runs(paths);groups={};metric='recall'+str(family['k']);goals=runs[0][1]['spec']['goals'];out=Path(out);out.mkdir(exist_ok=False)
 for p,m,rows in runs:
  need(len(rows)==1 and m['spec'].get('finalist_repeat') in (0,1,2),'Require actual finalist repetitions');groups.setdefault(key(rows[0]['point']),[]).append((p,m,rows[0]))
 for group in groups.values():need(len(group)==3 and {x[1]['spec']['finalist_repeat'] for x in group}=={0,1,2},'Each finalist needs3 distinct complete cal100 runs')
 selected=[];missing=[]
 for goal in goals:
  eligible=[g for g in groups.values() if all(x[2][metric]>=goal for x in g)]
  if not eligible:missing.append({'target':goal,'status':'no finalist meets target in all3 cal repetitions'});continue
  g=max(eligible,key=lambda g:statistics.median(x[2]['qps'] for x in g));p,m,row=g[0];s=copy.deepcopy(m['spec']);binding=[{'run':str(p),'prepared_sha256':sha(p/'prepared.json'),'summary_sha256':sha(p/'summary.json'),'release_sha256':sha(p/'release.json'),'accepted_summary_sha256':sha(p/'accepted-summary.json'),'raw_receipts_sha256':{q.name:sha(q) for q in sorted(p.iterdir()) if q.is_file() and q.suffix in ('.tsv','.log','.json','.jsonl')}} for p,m,row in g]
  s.update(calibration_query=s['query'],calibration_gt=s['gt'],calibration_gt_ids=s['gt_ids'],calibration_rows=s['query_rows'],phase='formal',query=s['full_query'],gt=s['full_gt'],gt_ids=s['full_gt_ids'],nq=s['expected_queries'],rounds=3,heldout_rows=s['heldout_rows_source'],selection_frozen_from={'three_independent_cal100_runs':binding,'selection_family':family,'metric':metric,'threshold':goal,'median_qps':statistics.median(x[2]['qps'] for x in g)},quality_targets=[{'metric':metric,'threshold':goal}],quality_target={'metric':metric,'threshold':goal});s.pop('query_rows',None);write(out/f'formal-{metric}-{goal:.2f}.json',s);selected.append(goal)
  controls=copy.deepcopy(m['spec']);controls['points']=[dict(row['point'],num_chunks=c) for c in (1,2,4,6,8,10)];controls['comparison']='fixed parameter chunk cal only; actual Khat separately profiled';write(out/f'chunk-cal-{goal:.2f}.json',controls)
  for label,t,aff in [('1core7',1,'7'),('4fast',4,'5-8'),('10fast',10,'5-9,15-19'),('20all',20,'0-19')]:
   c=copy.deepcopy(m['spec']);c.update(threads=t,cpu_affinity=aff,points=[row['point']],resource_configuration_control=True);c.pop('omp_places',None);c['omp_places']=omp_budget(c,aff,t)['OMP_PLACES'];write(out/f'cpu-cal-{goal:.2f}-{label}.json',c)
 write(out/'frozen-selection.json',{'family':family,'selected_targets':selected,'missing':missing,'heldout_policy':'Evaluate once after freeze; failed target remains failed; never select or retry using heldout','GPU_launched':False})
def ledger(paths,out):
 records=[]
 for p in map(Path,paths):
  m=read(p/'prepared.json');s=m['spec'];release=read(p/'release.json') if (p/'release.json').exists() else {};rows=read(p/'summary.json') if (p/'summary.json').exists() else []
  for row in rows:
   latency_stats={}
   if s['phase']=='formal':
    file=p/(row['round_name']+'.tsv');lat=[float(x['latency_ms']) for x in csv.DictReader(file.open(),delimiter='\t') if x['rank']=='1'];need(len(lat)==s['expected_queries'],'Formal latency count mismatch');latency_stats={'mean_ms':statistics.mean(lat),'p95_ms':float(np.percentile(lat,95)),'p99_ms':float(np.percentile(lat,99)),'qps':1000/statistics.mean(lat),'scope':'one full round'}
   metric='recall'+str(s['k']);targets=s.get('quality_targets',[{'metric':metric,'threshold':g} for g in s.get('goals',[])])
   records.append({'run':str(p),'point':row['point'],'round_name':row.get('round_name'),'phase':s['phase'],'formal_round_statistics':latency_stats,'family':selection_family(m),'complete':release.get('complete',False),'metrics':{k:v for k,v in row.items() if 'recall' in k},'targets':[dict(x,observed=row.get(('heldout_' if s.get('heldout_rows') else '')+x['metric']),passed=row.get(('heldout_' if s.get('heldout_rows') else '')+x['metric'],-1)>=x['threshold']) for x in targets],'heldout_failure_action':'retain failed target; no retuning or replacement from heldout','performance_review':'requires interference acceptance; cal medians are selection statistics only'})
  if not release.get('complete'):records.append({'run':str(p),'status':'incomplete','phase':s['phase']})
 write(out,{'all_observed_points':records,'GPU_launched':False})

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
 g=sub.add_parser('generate');g.add_argument('--out',required=True)
 f=sub.add_parser('finalize');f.add_argument('--template',required=True);f.add_argument('--root-acceptance',required=True);f.add_argument('--out',required=True)
 for name in ('finalists','freeze-final','ledger'):
  q=sub.add_parser(name);q.add_argument('--runs',nargs='+',required=True);q.add_argument('--out',required=True)
 a=p.parse_args()
 if a.action=='generate':generate(a.out)
 elif a.action=='finalize':finalize(a.template,a.root_acceptance,a.out)
 elif a.action=='finalists':finalists(a.runs,a.out)
 elif a.action=='freeze-final':freeze_final(a.runs,a.out)
 else:ledger(a.runs,a.out)
if __name__=='__main__':main()
