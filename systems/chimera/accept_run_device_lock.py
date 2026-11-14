"""Independently recheck a complete configured run before reporting aggregate QPS."""
import argparse,json,csv,statistics
from pathlib import Path
import numpy as np
from search_modes_v2 import selection_family,modes,verify_mode_log,validate_points
from run_search_grid_device_lock import need,sha,check,check_index_unchanged
from routing_audit import reject_quarantined,verify_routing_dataset

def main():
 p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();root=Path(a.run);prepared=json.loads((root/'prepared.json').read_text());reject_quarantined(prepared['spec']['index']);release=json.loads((root/'release.json').read_text());s=prepared['spec'];expected=prepared['expected_searches']
 need(prepared['point_parameter_semantics']==validate_points(s,prepared['nprobe_capacity_limit'],prepared['docs']),'Parameter effect semantics changed')
 need(prepared['selection_family']==selection_family(prepared),'Selection family identity changed');need(prepared['search_modes']==modes(s),'Mode identity changed')
 for detail in prepared['runner_source_snapshot'].values():need(sha(detail['path'])==detail['sha256'],'Runner source snapshot changed')
 need(sha(root/'frozen-exec-guard')==prepared['exec_guard_sha256'],'Frozen exec guard changed')
 need(release.get('complete') is True and release['completed_searches']==release['expected_searches']==expected,'Run did not finish its configured searches')
 need(sha(prepared['binary_snapshot'])==prepared['binary_sha256'],'Frozen executable changed')
 need(sha(s['query'])==prepared['query_sha256'] and sha(s['gt_ids'])==prepared['gt_ids_sha256'],'Input/GT artifact identity changed')
 check_index_unchanged(s['index'],prepared['index_identity']);routing=verify_routing_dataset(s,prepared['index_identity']);need(prepared.get('routing_dataset_validation')==routing,'Routing audit/guard binding differs from frozen run');rows=[];all_lat=[]
 for pi,point in enumerate(s['points']):
  samples=[]
  for rnd in range(s.get('rounds',1)):
   name=f'point{pi:02d}-round{rnd}';supervision=json.loads((root/(name+'-supervision.json')).read_text());need(supervision['owned_child_terminal_and_reaped'] and supervision['output_lock_inherited_by_child'] and supervision['returncode']==0 and supervision['requested_signal'] is None,'Search child supervision failed');file=root/(name+'.tsv');r=check(file,s['gt_ids'],prepared['queries'],prepared['docs'],s.get('k',100),s.get('heldout_rows'));r.update(actual_search_modes=verify_mode_log(root/(name+'.log'),s),round=rnd,point=point);samples.append(r)
   latency=[]
   with open(file) as f:
    for line in csv.DictReader(f,delimiter='\t'):
     if line['rank']=='1':latency.append(float(line['latency_ms']))
   need(len(latency)==prepared['queries'],'Missing per-query latency');all_lat.extend(latency)
  rows.append(dict(point=point,rounds=samples,qps_mean=statistics.mean(x['qps'] for x in samples),qps_round_range=[min(x['qps'] for x in samples),max(x['qps'] for x in samples)],mean_of_round_p99_ms=statistics.mean(x['p99_ms'] for x in samples),pooled_p99_ms=float(np.percentile(all_lat[-prepared['queries']*len(samples):],99))))
 aggregate=[]
 for group in rows:
  samples=group['rounds'];quality={}
  for target in s.get('quality_targets',[]):
   metric=target['metric'];mean=statistics.mean(x[metric] for x in samples);quality[metric+'@'+str(target['threshold'])]={'mean_full_recall':mean,'threshold':target['threshold'],'passed':mean>=target['threshold']}
  aggregate.append({'point':group['point'],'rounds':len(samples),'queries_per_round':prepared['queries'],'mean_full_recall10':statistics.mean(x['recall10'] for x in samples),'mean_round_QPS':statistics.mean(x['qps'] for x in samples),'quality':quality})
 need(not s.get('heldout_rows'),'New full-query protocol forbids heldout qualification')
 result=dict(full_query_aggregate=aggregate,aggregate_quality_policy='full2931 mean recall across configured rounds; per-round flags do not determine aggregate pass',search_modes=prepared['search_modes'],selection_family=prepared['selection_family'],output_contract_passed=True,routing_dataset_validation=routing,performance_acceptance='pending interference telemetry review',phase=s.get('phase','diagnostic'),k=s.get('k',100),queries_per_round=prepared['queries'],round_count=s.get('rounds',1),groups=rows,index_identity=prepared['index_identity'],query_input_placement=prepared['query_input_placement'],scope='Observed configured queries only. Three-round range is not a confidence interval; mean of round p99 and pooled p99 are named separately. Interference telemetry still requires review.')
 (root/'accepted-summary.json').write_text(json.dumps(result,indent=2));print(root/'accepted-summary.json')
if __name__=='__main__':main()
