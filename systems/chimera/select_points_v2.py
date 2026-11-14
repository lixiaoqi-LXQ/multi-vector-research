"""Select from calibration-only observations; emit frozen formal/diagnostic specs."""
import argparse,json,hashlib
from pathlib import Path
from run_search_grid_v2 import need,sha,omp_budget
from search_modes_v2 import selection_family

def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--out',required=True);p.add_argument('--neighbors-for',type=float);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);points=[];family=None
 for folder in map(Path,a.runs):
  meta=json.loads((folder/'prepared.json').read_text());release=json.loads((folder/'release.json').read_text());summary=json.loads((folder/'summary.json').read_text());spec=meta['spec']
  need(spec.get('phase')=='calibration' and meta['queries']==100 and spec.get('rounds')==1,'Selection accepts only independent cal100, never heldout/formal results')
  need(release.get('complete') is True and release.get('completed_searches')==meta['expected_searches']==len(summary),'Incomplete calibration grid')
  key=selection_family(meta)
  if family is None:family=key
  need(key==family,'Cannot mix mode/binary/index/bitwidth/K/query/GT/calibration/resource families in selection')
  for result in summary:
   need(result['queries']==100 and result['k']==spec['k'],'Invalid calibration point')
   points.append(dict(result=result,spec=spec,run=str(folder),prepared_sha256=sha(folder/'prepared.json'),summary_sha256=sha(folder/'summary.json')))
 k=family['k'];metric='recall'+str(k);goals=([.90,.95] if k==10 else [.90,.95,.99]);selected=[];missing=[]
 for goal in goals:
  eligible=[x for x in points if x['result'][metric]>=goal]
  if not eligible:missing.append(dict(metric=metric,threshold=goal,status='no measured calibration point reached target'));continue
  x=max(eligible,key=lambda x:x['result']['qps']);selected.append(dict(metric=metric,threshold=goal,source=x))
 groups={}
 for choice in selected:
  x=choice['source'];sig=json.dumps(dict(index=x['spec']['index'],point=x['result']['point'],k=k,threads=x['spec'].get('threads',20),cpu_affinity=x['spec'].get('cpu_affinity','0-19'),omp_places=x['spec'].get('omp_places')),sort_keys=True);g=groups.setdefault(sig,dict(source=x,targets=[]));g['targets'].append(dict(metric=metric,threshold=choice['threshold']))
 for number,g in enumerate(groups.values()):
  x=g['source'];s=dict(x['spec']);s.update(calibration_query=s['query'],calibration_gt=s['gt'],calibration_gt_ids=s['gt_ids'],calibration_rows=s['query_rows'],phase='formal',query=s['full_query'],gt=s['full_gt'],gt_ids=s['full_gt_ids'],nq=int(s.get('expected_queries',2931)),rounds=3,heldout_rows=s['heldout_rows_source'],threads=x['spec'].get('threads',20),cpu_affinity=x['spec'].get('cpu_affinity','0-19'),points=[x['result']['point']],selection_frozen_from=dict(run=x['run'],prepared_sha256=x['prepared_sha256'],summary_sha256=x['summary_sha256'],point=x['result']['point'],calibration_metrics=x['result']),quality_targets=g['targets'],quality_target=max(g['targets'],key=lambda t:t['threshold']),selection_note='Fastest eligible measured cal100 point; heldout is evaluated once and never drives retries or selection')
  s.pop('query_rows',None);write(out/f'formal-{number:02d}.json',s)
  # Prepare only: run chunk cal first, then select c1/cbest for paired formal runs.
  chunk=dict(x['spec']);chunk['points']=[dict(x['result']['point'],num_chunks=c) for c in (1,2,4,6,8,10)];chunk['comparison']='Same index/nprobe/krefine/kfull/CPU; actual extra fullbit work must be disclosed';write(out/f'chunk-calibration-{number:02d}.json',chunk)
  for label,threads,affinity in [('1core7',1,'7'),('4fast',4,'5-8'),('10fast',10,'5-9,15-19'),('20all',20,'0-19')]:
   diag=dict(x['spec']);diag.pop('omp_places',None);diag.update(phase='calibration',threads=threads,cpu_affinity=affinity,points=[x['result']['point']],comparison='Fixed index/parameters/chunks/cal100 rows; CPU organization control. Primary paper budget remains20all; freeze alternative configuration before heldout.',resource_configuration_control=True);diag['omp_places']=omp_budget(diag,affinity,threads)['OMP_PLACES'];write(out/f'cpu-diagnostic-{number:02d}-{label}.json',diag)
 if a.neighbors_for is not None:
  eligible=[x for x in points if x['result'][metric]>=a.neighbors_for]
  if eligible:
   x=max(eligible,key=lambda x:x['result']['qps']);base=x['result']['point'];neighbors=[];nd=x['spec']['expected_index']['N'];docs=json.loads((Path(x['run'])/'prepared.json').read_text())['docs']
   for field in ('nprobe','k_refine','k_full_bit'):
    for scale in (.5,2):
     q=dict(base);q[field]=max(1 if field=='nprobe' else k,int(q[field]*scale))
     q['nprobe']=min(q['nprobe'],512);q['k_refine']=min(q['k_refine'],docs);q['k_full_bit']=min(q['k_full_bit'],q['k_refine']);q['cagra_itopk_size']=min(q['nprobe']+32,544)
     if q!=base and q not in neighbors:neighbors.append(q)
   follow=dict(x['spec'],points=neighbors,local_refinement_of=dict(metric=metric,threshold=a.neighbors_for,point=base));write(out/'calibration-local-neighbors.json',follow)
 write(out/'selection.json',dict(calibration_only=True,family=family,selected=[dict(metric=x['metric'],threshold=x['threshold'],run=x['source']['run'],point=x['source']['result']['point'],calibration_metrics=x['source']['result']) for x in selected],missing=missing,formal_groups=len(groups),scope='No GPU work launched. Missing targets and subsequent heldout failures remain explicit.'))
 print(out/'selection.json')
if __name__=='__main__':main()
