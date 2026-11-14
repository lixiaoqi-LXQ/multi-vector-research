"""CPU-only stage/resource/holder policy for the specifically authorized MS4 build cohort."""
import hashlib,json,os,re,shutil
from pathlib import Path
GIB=1024**3

def process(pid):
 try:
  data=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return {'pid':pid,'starttime_ticks':int(data[19]),'state':data[0],'cmdline_sha256':hashlib.sha256(Path(f'/proc/{pid}/cmdline').read_bytes()).hexdigest()}
 except (FileNotFoundError,ProcessLookupError):return None

def matches(expected):
 got=process(expected['pid'])
 return got is not None and got['state']!='Z' and all(got[k]==expected[k] for k in ['pid','starttime_ticks','cmdline_sha256'])

def terminal_native_proof(cohort,identity_reader=process,json_reader=lambda p:json.loads(Path(p).read_text())):
 # A reused or zombie PID is not the original reaped child.
 if identity_reader(cohort['native']['pid']) is not None:raise AssertionError('Native PID remains present or was reused')
 bound=cohort['native_terminal_binding'];job=json_reader(bound['job_state']);prepared=json_reader(bound['prepared']);proc=json_reader(bound['process']);receipt=json_reader(bound['supervision'])
 assert job['receipt_dir']==bound['receipt_dir'] and job['index']==bound['index'],'Source job/receipt binding differs'
 for role in ['launcher','wrapper']:
  assert job[role]['pid']==cohort[role]['pid'] and job[role]['start_ticks']==cohort[role]['starttime_ticks'],'Source job role differs'
 assert prepared['identity_sha256']==bound['prepared_identity_sha256'],'Prepared source identity differs'
 assert proc['pid']==cohort['native']['pid'],'Source process native differs'
 assert hashlib.sha256(json.dumps(proc['command'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==bound['native_command_sha256'],'Source native argv differs'
 assert proc['command']==prepared['command'],'Prepared/process argv differs'
 assert receipt['child_pid']==cohort['native']['pid'] and receipt['returncode']==0,'Not the bound successful native'
 assert receipt['status']=='process_exited' and receipt['requested_signal'] is None and receipt['error'] is None,'Native interrupted or supervision failed'
 assert receipt['owned_child_terminal_and_reaped'] is True and receipt['termination_escalated'] is False,'Native not cleanly reaped'
 return {'accepted':'native_rc0_reaped_wrapper_hashing','supervision':bound['supervision'],'native_pid':cohort['native']['pid'],'source_identity_sha256':prepared['identity_sha256']}

def validate_holder_cohort(cohort,matcher=matches,identity_reader=process,json_reader=lambda p:json.loads(Path(p).read_text())):
 for role in ['guardian','launcher','wrapper']:assert matcher(cohort[role]),('Holder cohort identity does not match',role,cohort[role])
 if matcher(cohort['native']):return {'accepted':'native_live'}
 assert cohort['name'] in ['coarse','fine'],'MS holder native must remain live'
 return terminal_native_proof(cohort,identity_reader,json_reader)

def holder(spec):
 path=Path(spec['global_lock']);st=path.stat();key=f'{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}';pids=[]
 for line in Path('/proc/locks').read_text().splitlines():
  f=line.split()
  if '->' not in f and len(f)>5 and f[1:4]==['FLOCK','ADVISORY','WRITE'] and f[5]==key:pids.append(int(f[4]))
 assert len(pids)==1,('Global WRITE holder count changed',pids)
 cohort=next((x for x in spec['allowed_guardian_cohorts'] if x['guardian']['pid']==pids[0]),None)
 assert cohort is not None,('Unknown global WRITE holder',pids)
 proof=validate_holder_cohort(cohort)
 return {'actual_global_write_holder':pids[0],'role':cohort['name'],'identities':cohort,'native_lifecycle_proof':proof,'this_job_claims_global_lease':False}

def ms_stage(path):
 # Read only the bounded diagnostic log. Search all existing markers on first read;
 # subsequent polls inspect appended bytes via the caller's file offset.
 text=Path(path).read_text(errors='replace')
 return parse_ms_stage(text)

def parse_ms_stage(text):
 forbidden=['assignment_complete','rotated_graph_build_start','rotated_graph_complete','encoding_start','index_ready_for_save']
 if any('[BUILD_CHECKPOINT] '+x in text for x in forbidden):return 'left_assignment'
 matches_=list(re.finditer(r'assignment_committed completed_tokens=(\d+) total_tokens=(\d+)',text))
 if not matches_:return 'unconfirmed'
 last=matches_[-1]
 return 'assignment' if int(last[1])<int(last[2]) else 'assignment_complete_commit'

def resource_snapshot(spec):
 mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemFree:','MemTotal:'))}
 return {'memory_bytes':mem,'disk_free_bytes':shutil.disk_usage(spec['data_root']).free}

def stop_reason(spec,available,soft_count,stage,holder_ok,disk,source_ok=True,own_anon=0,own_nvml=0):
 if not source_ok:return 'source_stat_changed'
 if not holder_ok:return 'unknown_or_stale_actual_global_holder_cohort'
 if stage!='assignment':return 'MS_Chimera_left_authorized_assignment_phase:'+stage
 if available<=spec['runtime_hard_MemAvailable_bytes']:return 'hard_low_MemAvailable'
 if available<spec['runtime_soft_MemAvailable_bytes'] and soft_count>=2:return 'soft_low_MemAvailable_two_samples'
 if disk<spec['runtime_min_disk_free_bytes']:return 'low_disk_free'
 if own_anon>spec['runtime_own_RssAnon_max_bytes']:return 'own_RssAnon_exceeded_budget'
 if own_nvml>spec['runtime_own_NVML_max_bytes']:return 'own_NVML_exceeded_budget'
 return None

def source_stat(path):
 s=Path(path).stat();return {'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}

def verify(spec,fresh=True):
 for path,expected in spec['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
 for path,expected in spec['bound_metadata_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
 assert source_stat(spec['source_file'])==spec['source_stat'],'Verified raw source stat changed'
 proof=json.loads(Path(spec['full_source_verification_receipt']).read_text());assert proof['status']=='VERIFIED_COMPLETE' and proof['actual_sha256']==spec['source_file_sha256'] and proof['file_after']==spec['source_stat']
 gt=json.loads(Path(spec['GT_manifest']).read_text());assert gt['complete'] and gt['docs_done']==8841823 and gt['nq']==6980 and gt['scanned_prefix_sha256']==spec['GT_scanned_payload_sha256']
 assert matches(spec['MS_native']),'MS native must remain live; terminalproof substitution forbidden'
 stage=ms_stage(spec['MS_assignment_log']);assert stage=='assignment',stage
 h=holder(spec);resources=resource_snapshot(spec)
 assert resources['memory_bytes']['MemAvailable']>=spec['launch_min_MemAvailable_bytes'],('Launch memory gate',resources)
 assert resources['disk_free_bytes']>=spec['launch_min_disk_free_bytes'],('Launch disk gate',resources)
 if fresh:
  assert not Path(spec['index']).exists(),'Fresh output exists; preserve it and seek a new reviewed attempt'
  assert not Path(spec['run_output']).exists(),'Fresh run receipt directory exists'
 return {'status':'CPU_PREFLIGHT_PASS','holder':h,'resources':resources,'MS_stage':stage,'raw_source_not_rehashed':True,'operation':'completeMS4 offline cohort construction only; no calibration/QPS'}
