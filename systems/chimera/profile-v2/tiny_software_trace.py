"""One root-authorized tiny software CUDA trace; original HES capture remains intact."""
import fcntl,json,os,sys,uuid
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import tiny_diagnostic as tiny
def main():
 D=tiny.D;f=tiny.read(D/'software-trace-freeze.json')
 for p,h in f['files'].items():tiny.need(tiny.sha(p)==h,'Software diagnostic freeze changed: '+p)
 frozen,actors=tiny.validate_freeze();plan=tiny.read(D/'plan.json');tiny.need(tiny.read(D/'QA-acceptance.json')['status']=='PASS_TINY_MAIN_VS_OBSERVATION_32_IDS_F32_BITS','Main/obs QA must pass');out=D/'nsys32-sw';out.mkdir(exist_ok=False);cfgpath=out/'tree-config.json';lockpath=out/'diagnostic-output.lock';nonce=uuid.uuid4().hex
 old=tiny.read(D/'nsys32/tree-config.json');cfg=dict(old,socket_name='chimera-tiny-sw-'+nonce,token=nonce,lock_path=str(lockpath),leaf_receipt=str(out/'leaf-lease.json'),tree_receipt=str(out/'tree-supervision.json'));cfg['engine_command']=tiny.command(plan,D/'observation-search',out/'rows.tsv',True);cfg['launch_command']=list(old['launch_command']);cfg['launch_command'][cfg['launch_command'].index('--trace=cuda,nvtx')]='--trace=cuda-sw,nvtx';cfg['launch_command'][cfg['launch_command'].index('--output')+1]=str(out/'trace');cfg['launch_command'][-1]=str(cfgpath)
 # Apart from fresh namespace paths, only the trace backend option differs.
 normalized=[v.replace('/nsys32-sw/','/nsys32/')for v in cfg['launch_command']];normalized[normalized.index('--trace=cuda-sw,nvtx')]='--trace=cuda,nvtx';tiny.need(normalized==old['launch_command'],'Unexpected command mutation');tiny.exclusive_json(cfgpath,cfg)
 env=dict(os.environ,**plan['environment']);env.pop('GOMP_CPU_AFFINITY',None);os.environ['PLAID_DEVICE_LOCK']=str(lockpath);env['PLAID_DEVICE_LOCK']=str(lockpath)
 with open(lockpath,'a')as lock,open(out/'output.log','x')as log:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);child=tiny.DiagnosticPopen([sys.executable,str(P/'tree_guard.py'),str(cfgpath),str(lock.fileno())],env=env,stdout=log,stderr=tiny.subprocess.STDOUT,pass_fds=(lock.fileno(),));tiny.exclusive_json(out/'process.json',dict(pid=child.pid,start_ticks=int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]),command=cfg['launch_command'],actual_cohort=actors,software_trace_freeze_sha256=tiny.sha(D/'software-trace-freeze.json'),diagnostic_output_lease=str(lockpath),global_device_lease_owned_by_coarse_guardian=3474856,not_a_formal_profile=True,timeout_s=175,cleanup_s=10,performance_claim=False))
  try:rc=child.wait(timeout=210)
  finally:
   if child.poll()is None:child.terminate_and_reap()
 r=tiny.read(out/'tree-supervision.json');tiny.need(rc==0 and r['all_owned_descendants_gone']and not r['timed_out'],'Tiny software NSYS tree failed');value=tiny.compare('observation','nsys32-sw');tiny.exclusive_json(out/'execution.json',dict(status='TINY_SOFTWARE_TRACE_EXECUTED_AWAITING_FORMAT_PARSE',**value,trace_sha256=tiny.sha(out/'trace.nsys-rep'),tree_sha256=tiny.sha(out/'tree-supervision.json'),diagnostic_lease_only=True,performance_claim=False));print(json.dumps(dict(stage='nsys32-sw',passed=True,guard_pid=child.pid,leaf_pid=tiny.read(out/'leaf-lease.json')['pid'],all_owned_descendants_gone=True,performance_claim=False)),flush=True)
if __name__=='__main__':main()
