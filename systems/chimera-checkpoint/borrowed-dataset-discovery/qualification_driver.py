#!/usr/bin/env python3
"""Authorized prefix qualification orchestration; does not acquire the device lease."""
import argparse, fcntl, hashlib, json, os, signal, subprocess, sys, threading, time
from pathlib import Path
from supervision import supervise_child
from run_checkpoint_build import exclusive_json
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gpu-qualification'

def check_freeze():
    frozen=json.loads((ROOT/'gpu-freeze.json').read_text())
    for name,item in frozen['files'].items():
        p=ROOT/name;s=p.stat()
        assert (s.st_size,s.st_ino,s.st_mtime_ns)==(item['bytes'],item['inode'],item['mtime_ns']),name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],name

def command_for(phase):
    args=list(json.loads((ROOT/'wrapper-cpu-preflight.json').read_text())['future_command_without_run_flag'])
    args[0]=sys.executable
    if phase=='q1':return [*args,'--run'],3
    plan=json.loads((ROOT/'roundtrip-fixture/plan.json').read_text())
    if phase=='roundtrip':return plan['command'],0
    raise ValueError('Unknown phase')

def run(phase,command,expected):
    check_freeze();OUT.mkdir(exist_ok=True);record=OUT/(phase+'-launch.json');assert not record.exists()
    env=os.environ.copy();env.update(UMA_RUN_GPU_TESTS='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OMP_PLACES='{15},{16},{17},{18}',OMP_PROC_BIND='true',OMP_DYNAMIC='FALSE',OMP_MAX_ACTIVE_LEVELS='1',CUDA_CACHE_PATH=str(OUT/'cuda-cache'))
    if phase=='roundtrip':
        plan=json.loads((ROOT/'roundtrip-fixture/plan.json').read_text())
        env.update(plan['required_environment'])
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();timed_out=threading.Event();timer=None
    with open(OUT/(phase+'-driver.lock'),'a') as lock,open(OUT/(phase+'-driver.log'),'x') as log:
        # A local fixture/output lock only. The root-owned offline device lease is unchanged.
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def started(child):
            nonlocal timer
            exclusive_json(record,{'phase':phase,'supervised_pid':child.pid,'command':command,'environment':{k:env[k] for k in ['UMA_RUN_GPU_TESTS','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OMP_PLACES','OMP_PROC_BIND','OMP_DYNAMIC','OMP_MAX_ACTIVE_LEVELS','CUDA_CACHE_PATH']},'timeout_s':165,'termination_grace_s':10,'expected_returncode':expected,'offline_overlap_pids':[2765928,2877509],'driver_sha256':source_sha,'started_at_unix':time.time()})
            def expire():
                if child.poll() is None:
                    timed_out.set();os.kill(os.getpid(),signal.SIGTERM)
            timer=threading.Timer(165,expire);timer.start()
        try:
            result=supervise_child(command,env=env,log=log,output_lock=lock,exec_guard=ROOT/'exec_guard',receipt_path=OUT/(phase+'-driver-supervision.json'),write_receipt=exclusive_json,on_started=started,terminate_grace_s=10)
        finally:
            if timer is not None:timer.cancel();timer.join()
    passed=result['returncode']==expected and result['requested_signal'] is None and not timed_out.is_set()
    exclusive_json(OUT/(phase+'-driver-result.json'),dict(phase=phase,passed=passed,timed_out=timed_out.is_set(),supervision=result))
    print(json.dumps({'phase':phase,'passed':passed,'returncode':result['returncode'],'timed_out':timed_out.is_set()}),flush=True)
    if not passed:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['q1','roundtrip']);a=p.parse_args();command,expected=command_for(a.phase);run(a.phase,command,expected)
