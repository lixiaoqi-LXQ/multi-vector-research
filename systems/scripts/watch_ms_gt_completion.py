"""Observe the existing exact GT PID; never start or signal GPU work."""
import json
import time
from pathlib import Path
from gt_state import atomic_json, file_sha
from accept_ms_gt import accept

R = Path('/home/lxq/graduate/ai-workspace/research/multivector')
PID, START = 2877509, 244562069
OUTPUT = Path('/home/lxq/graduate/data/multivector-colbertv2/groundtruth/msmarco-v1-passage-full/calibration-seed42')
RECEIPT = R/'data/gt-ms-completion-observer.json'


def identity():
    try:
        text = Path(f'/proc/{PID}/stat').read_text()
        fields = text[text.rfind(')')+2:].split()
        return dict(pid=PID, state=fields[0], start_ticks=int(fields[19]))
    except FileNotFoundError:
        return None


def main():
    initial = identity()
    if initial is None or initial['start_ticks'] != START:
        raise RuntimeError('Expected existing GT process identity is absent')
    evidence = {'status':'observing_existing_GT', 'pid':PID, 'start_ticks':START,
                'started_unix':time.time(), 'GPU_used':False,
                'script_sha256':file_sha(__file__), 'validator_sha256':file_sha(Path(__file__).with_name('accept_ms_gt.py')),
                'original_launch_record':str(R/'data/gt-ms-offline-job-v2.json'),
                'original_supervisor_observation':{'session':59913, 'exit_code':143,
                    'GT_process_continued_with_parent_pid':1},
                'signals_sent':False, 'new_GPU_work_started':False}
    atomic_json(RECEIPT, evidence)
    while True:
        current = identity()
        if current is None or current['start_ticks'] != START or current['state'] == 'Z':
            break
        time.sleep(2)
    evidence.update(observed_terminal_unix=time.time(), terminal_identity=current,
                    child_exit_code=None, child_exit_code_reason='Original supervisor terminated; observer is not the process parent')
    try:
        accepted = accept(OUTPUT)
        evidence.update(status='PASS_FULL_OUTPUT_ACCEPTANCE', calibration_projection=str(OUTPUT),
                        acceptance=accepted)
    except BaseException as exc:
        evidence.update(status='OUTPUT_ACCEPTANCE_FAILED', error=repr(exc))
        atomic_json(RECEIPT, evidence)
        raise
    atomic_json(RECEIPT, evidence)
    print(json.dumps({'status':evidence['status'], 'calibration_projection':str(OUTPUT)}), flush=True)


if __name__ == '__main__':
    main()
