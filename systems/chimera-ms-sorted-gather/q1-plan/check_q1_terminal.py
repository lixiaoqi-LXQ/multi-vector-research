"""Read-only Q1 terminal/prefix and queued-guardian eligibility report; never kills."""
from pathlib import Path
import argparse,json,hashlib,struct,subprocess
import preflight
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--spec',default=str(P/'spec.json'));p.add_argument('--verify-cache-artifacts',action='store_true');a=p.parse_args();s=json.loads(Path(a.spec).read_text());state=Path(s['job_state'])
if not state.exists():print(json.dumps({'status':'NOT_STARTED','GPU_used':False}));raise SystemExit
job=json.loads(state.read_text());assert job['status']=='checkpoint_stopped' and job['returncode']==3 and job['intentional_checkpoint_stop']
receipt=Path(job['receipt_dir']);result=json.loads((receipt/'result.json').read_text());sup=json.loads((receipt/'supervision.json').read_text());assert result['intentional_stop'] and result['returncode']==3 and not result['complete'] and sup['owned_child_terminal_and_reaped'] and sup['returncode']==3
cmd=s['command'];cache=Path(cmd[cmd.index('--cache')+1]);tokens=(cache/'state.txt').read_text().split();assert tokens[0]=='CHIMERA_ASSIGNMENT_CHECKPOINT_V1';assert list(map(int,tokens[3:12]))==[597909919,128,4194304,3,65536,42,42,1,150];assert int(tokens[16])==65536 and tokens[17]=='-' and int(tokens[18])==1 and list(map(int,tokens[19:21]))==[0,65536];label=cache/'labels/0.u32';b=label.read_bytes();assert len(b)==262144 and hashlib.sha256(b).hexdigest()==tokens[21];assert all(x[0]<4194304 for x in struct.iter_unpack('<I',b));assert hashlib.sha256((cache/'state.txt').read_bytes()).hexdigest()==result['checkpoint_state_sha256'];assert not Path(cmd[cmd.index('--index')+1]).exists() and not Path(cmd[cmd.index('--spill-full-codes')+1]).exists()
verified_cache_artifacts={}
if a.verify_cache_artifacts:
 for name,size_token,sha_token in [('centroids.f32',12,13),('assignment.cagra',14,15)]:
  file=cache/name;assert file.stat().st_size==int(tokens[size_token]);before=file.stat()
  with open(file,'rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  after=file.stat();assert digest==tokens[sha_token] and (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns);verified_cache_artifacts[name]={'bytes':after.st_size,'sha256':digest}
gid=job['continuation_guardian_identity'];now=preflight.process(gid['pid']);report=Path(s['base'])/'guardian-state.json';g=json.loads(report.read_text());locks=subprocess.check_output(['lslocks','-n','-o','PID,MODE,PATH'],text=True);holders=[int(row.split()[0]) for row in locks.splitlines() if row.split() and row.split()[-1]==s['lease_path'] and '*' not in row.split()[1]]
eligible=now is not None and now['state']!='Z' and now['start_ticks']==gid['start_ticks'] and g['pid']==gid['pid'] and g['phase']=='queued' and gid['pid'] not in holders
print(json.dumps({'status':'PASS_FULL_MS_Q1_DURABLE_PREFIX','completed_tokens':65536,'total_tokens':597909919,'cache':str(cache),'guardian_identity':gid,'actual_lease_holders':holders,'exact_queued_nonholder_guardian_eligible_for_root_termination':eligible,'action':'Root rechecks exact guardian identity and actual nonholder state immediately before SIGTERM; if holding, allow natural release. This helper never sends a signal.','full_graph_digest':'recorded by native commit; full cache digests revalidated by core on resume','verified_cache_artifacts':verified_cache_artifacts,'GPU_used':False},indent=2))
