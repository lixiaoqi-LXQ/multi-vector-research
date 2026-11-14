"""CPU-only boundary/phase/identity policy checks; no GPU or build launch."""
import copy,json
from pathlib import Path
import policy
H=Path(__file__).parent;spec=json.loads((H/'spec.json').read_text());G=1024**3;checks=[]
def r(mem=60*G,count=0,stage='assignment',holder=True,disk=100*G,source=True,anon=0,nvml=0):return policy.stop_reason(spec,mem,count,stage,holder,disk,source,anon,nvml)
assert r() is None;checks.append('normal_assignment_allowed')
assert r(mem=32*G)=='hard_low_MemAvailable';assert r(mem=32*G+1) is None;checks.append('hard32GiB_exact_boundary')
assert r(mem=39*G,count=1) is None;assert r(mem=39*G,count=2)=='soft_low_MemAvailable_two_samples';checks.append('soft40GiB_two_samples')
for stage in ['left_assignment','assignment_complete_commit','unconfirmed','MS_native_missing_or_changed','MS_log_replaced_or_truncated']:assert r(stage=stage).startswith('MS_Chimera_left');checks.append('reject_stage_'+stage)
assert r(holder=False).startswith('unknown_or_stale');assert r(source=False)=='source_stat_changed';assert r(anon=28*G+1)=='own_RssAnon_exceeded_budget';assert r(nvml=4*G+1)=='own_NVML_exceeded_budget';checks.append('reject_unknownholder_sourcechange_ownbudget')
assert policy.parse_ms_stage('[BUILD_CHECKPOINT] assignment_committed completed_tokens=10 total_tokens=100')=='assignment'
assert policy.parse_ms_stage('[BUILD_CHECKPOINT] assignment_committed completed_tokens=100 total_tokens=100')=='assignment_complete_commit'
for marker in ['assignment_complete','rotated_graph_build_start','rotated_graph_complete','encoding_start','index_ready_for_save']:assert policy.parse_ms_stage('[BUILD_CHECKPOINT] '+marker)=='left_assignment'
checks.append('all_MS_phase_exit_markers_rejected')
# Same split-line assembly used by the runtime watcher: no dropped marker.
pending='[BUILD_CHECKPOINT] assign';delta='ment_complete\n';lines=(pending+delta).split('\n');last=lines.pop();assert last=='' and policy.parse_ms_stage('\n'.join(lines))=='left_assignment';checks.append('split_log_marker_preserved')
assert spec['launch_min_MemAvailable_bytes']==100719476736 and spec['launch_min_MemAvailable_bytes']>=72*G;checks.append('uniform_bytes_budget_gate')
assert policy.holder(spec)['actual_global_write_holder']==3474856
bad=copy.deepcopy(spec);bad['allowed_guardian_cohorts'][0]['guardian']['starttime_ticks']+=1
try:policy.holder(bad)
except AssertionError:checks.append('stale_guardian_identity_rejected')
else:raise AssertionError('Stale guardian accepted')
bad=copy.deepcopy(spec);bad['allowed_guardian_cohorts']=[]
try:policy.holder(bad)
except AssertionError:checks.append('unknown_actual_holder_rejected')
else:raise AssertionError('Unknown holder accepted')
# Both queued handoff cohorts have real matching guardian/launcher/native identities.
for item in spec['allowed_guardian_cohorts']:
 assert all(policy.matches(item[role]) for role in ['guardian','launcher','native'])
checks.append('three_live_cohort_pairs_match_PID_start_cmdline')
(H/'cpu-tests.json').write_text(json.dumps({'complete':True,'GPU_used':False,'build_launched':False,'checks':checks},indent=2));print(len(checks),'CPU policy checks PASS')
