"""Resume the frozen MS GT in the authorized offline cohort, with exact-child supervision."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R = Path('/home/lxq/graduate/ai-workspace/research/multivector')
sys.path.insert(0, str(R/'chimera-checkpoint'))
from supervision import supervise_child
from gt_state import atomic_json, file_sha

OUT = Path('/home/lxq/graduate/data/multivector-colbertv2/groundtruth/msmarco-v1-passage-full')
BASE = R/'data/ms-gt-resume-v3'
STATE = BASE/'job-state.json'
GUARD = R/'chimera/runs/full-lotte-owned-device-checkpoint-v1/frozen-guard_full.py'
EXEC = R/'chimera-checkpoint/exec_guard'
ARM_PID, ARM_START, ARM_GUARDIAN = 3233295, 245395633, 3228697
PYTHON = R/'plaid/.venv/bin/python'


def ident(pid):
    raw = Path(f'/proc/{pid}/stat').read_text()
    fields = raw[raw.rfind(')')+2:].split()
    return dict(pid=pid, start_ticks=int(fields[19]), state=fields[0])


def owners():
    rows = subprocess.check_output(['lslocks','-n','-o','PID,MODE,PATH'], text=True)
    return [int(v[0]) for row in rows.splitlines() if len(v:=row.split()) == 3
            and v[2] == '/tmp/uma-paper-device0.lock' and '*' not in v[1]]


def exclusive(path, value):
    with open(path, 'x') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def main():
    BASE.mkdir(exist_ok=False)
    for name, digest in {'exact_gt.py':'5826f0156af8c026759f5ac27e9dd881a30e30d0ef6e4917a7079c23bfaac51c',
                         'gt_state.py':'40525a882dd1f539100cc36faa2bbd0e7c43a620bade053612b80d875a48d5c5',
                         'segmented_maxsim.py':'04f14e7c35e5ee5262545f5dfe4aeb1645c8b429e5bbc49f3ac3c9c8bbe9e653'}.items():
        assert file_sha(R/'scripts'/name) == digest, name
    assert file_sha(EXEC) == 'b73678278e797e04e29ccf5a5146e7af8e4d9e05d74a5281891e0bb9567749da'
    assert ident(ARM_PID)['start_ticks'] == ARM_START
    assert json.loads((OUT/'manifest.json').read_text()).get('complete') is False
    command = ['taskset','-c','5,6,8,9',str(PYTHON),str(R/'scripts/exact_gt.py'),
               '--data','/home/lxq/graduate/data/multivector-colbertv2/msmarco-v1-passage',
               '--docs','8841823','--queries','0','--out',str(OUT),'--resume',str(OUT),
               '--query-batch','32','--token-block','131072','--k','100','--threads','4','--reducer','triton']
    job = dict(status='starting', launcher=ident(os.getpid()), wrapper=None, receipt_dir=str(BASE),
               binary=str(PYTHON), index=str(OUT), command=command, started_at_unix=time.time(),
               prior_job=str(R/'data/gt-ms-offline-job-v2.json'), prior_last_logged_docs=6190643,
               durable_checkpoint_docs=6189179, purpose='resume exact full GT; no performance measurement',
               supervisor_sha256=file_sha(R/'chimera-checkpoint/supervision.py'),
               exec_guard_sha256=file_sha(EXEC), guardian_source_sha256=file_sha(GUARD),
               script_sha256=file_sha(__file__), atomic_checkpoint_full_prefix_validation='performed by unchanged exact_gt --resume')
    atomic_json(STATE, job)
    with open(BASE/'guardian.log','w') as log:
        guardian = subprocess.Popen([sys.executable,str(GUARD),'--state',str(STATE)], stdout=log,stderr=subprocess.STDOUT)
    job['continuation_guardian_pid'] = guardian.pid
    atomic_json(STATE, job)
    while True:
        assert guardian.poll() is None, 'GT continuation guardian exited before launch'
        holding = owners()
        if guardian.pid in holding:
            break
        if holding == [ARM_GUARDIAN] and ident(ARM_PID)['start_ticks'] == ARM_START and ident(ARM_PID)['state'] != 'Z':
            gp = BASE/'guardian-state.json'
            if gp.exists() and json.loads(gp.read_text()).get('phase') == 'queued':
                break
        time.sleep(.1)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0',UMA_RUN_GPU_TESTS='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
               TRITON_CACHE_DIR=str(R/'builds/triton-ms-gt'))
    for key in ['OMP_PLACES','GOMP_CPU_AFFINITY','OMP_PROC_BIND','OMP_THREAD_LIMIT']:
        env.pop(key, None)
    def started(child):
        job.update(status='running', wrapper=ident(child.pid))
        atomic_json(STATE, job)
        exclusive(BASE/'process.json', dict(pid=child.pid, command=command))
        print(json.dumps(dict(launcher_pid=os.getpid(), child_pid=child.pid, guardian_pid=guardian.pid)),flush=True)
    with open(BASE/'output.lock','a') as lock, open(BASE/'gt.log','w') as log:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result = supervise_child(command,env=env,log=log,output_lock=lock,exec_guard=EXEC,
                                 receipt_path=BASE/'supervision.json',write_receipt=exclusive,on_started=started)
    job.update(status='complete' if result['returncode'] == 0 else 'failed',
               returncode=result['returncode'], finished_at_unix=time.time())
    atomic_json(STATE, job)
    if result['returncode'] != 0:
        raise RuntimeError('GT resumed child failed; checkpoint and log retained')
    from accept_ms_gt import accept
    accepted = accept(OUT/'calibration-seed42')
    atomic_json(BASE/'full-output-acceptance.json',accepted)
    print(json.dumps(dict(status='PASS_FULL_OUTPUT_ACCEPTANCE', docs=accepted['docs'])),flush=True)


if __name__ == '__main__':
    main()
