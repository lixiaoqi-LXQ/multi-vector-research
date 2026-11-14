from pathlib import Path
import tempfile,json,copy,shutil
import plan
P=Path(__file__).resolve().parent;F=P/'cpu-fixtures-v2';F.mkdir(exist_ok=False);base=plan.read(next((P/'templates-v3').rglob('screen.template.json')));base['warmup']=32;base['points']=base['points'][:1];base['finalist_id']='f00'
def fixture(name,repeat):
 p=F/name;p.mkdir();s=copy.deepcopy(base);s['finalist_repeat']=repeat
 m={'spec':s,'queries':100,'expected_searches':1,'binary_sha256':'binaryA','index_identity':{'N':1,'total_bits':4},'k':s['k'],'query_payload_sha256':'queryA','gt_ids_sha256':'gtA','GT_manifest':{'docs':1},'GT_manifest_sha256':'manifestA','query_rows_manifest_sha256':'splitA','CPU_affinity':'0-19','OpenMP_environment':{'OMP_NUM_THREADS':'20'},'load_mode':'original_owning_arrays'}
 row={'point':s['points'][0],'queries':100,'k':s['k'],'recall10':.97,'recall100':.97,'qps':100+repeat,'actual_search_modes':{'verified':True}}
 for n,x in [('prepared.json',m),('release.json',{'complete':True,'completed_searches':1}),('summary.json',[row]),('accepted-summary.json',{'output_contract_passed':True})]:plan.write(p/n,x)
 return p
runs=[fixture('r'+str(x),x) for x in range(3)];plan.freeze_final(runs,F/'valid');assert plan.read(F/'valid/frozen-selection.json')['selected_targets']==base['goals'];count=1
cases=[('missing',runs[:2],None,None),('duplicate',runs+[runs[0]],None,None)]
for field in ('binary_sha256','query_payload_sha256','gt_ids_sha256','index_identity','CPU_affinity','OpenMP_environment','load_mode'):
 cases.append((field,runs,field,'wrong'))
for name,paths,field,value in cases:
 backup=None
 if field:
  p=runs[2]/'prepared.json';backup=p.read_text();m=json.loads(backup);m[field]=value;plan.write(p,m)
 try:
  try:plan.freeze_final(paths,F/('bad-'+name))
  except (ValueError,KeyError):count+=1
  else:raise AssertionError('Accepted invalid '+name)
 finally:
  if backup:p.write_text(backup)
for label,change in [('profile',lambda s:s.update(profile_range=True)),('wrongmode',lambda s:s.update(kernel_mode='no-lut')),('wrongpoints',lambda s:s.update(points=[])),('repeat',lambda s:s.update(finalist_repeat=1)),('warmup',lambda s:s.update(warmup=5))]:
 p=runs[2]/'prepared.json';backup=p.read_text();m=json.loads(backup);change(m['spec']);plan.write(p,m)
 try:
  try:plan.freeze_final(runs,F/('bad-'+label))
  except (ValueError,KeyError):count+=1
  else:raise AssertionError('Accepted '+label)
 finally:p.write_text(backup)
try:plan.finalize(next((P/'templates-v3').rglob('screen.template.json')),P/'missing-acceptance.json',F/'bad-finalized.json')
except FileNotFoundError:count+=1
else:raise AssertionError('Missing full index acceptance accepted')
plan.write(P/'CPU-validation.json',{'passed':True,'GPU_used':False,'checks':count,'synthetic_selection_fixtures':str(F),'finalizer_rejects_unpublished_full_index':True,'runner_freeze_sha256':plan.EXPECTED});print(count,'CPU checks PASS')
