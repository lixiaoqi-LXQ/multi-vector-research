from pathlib import Path
import hashlib,json,shutil
P=Path(__file__).resolve().parent;R=P.parents[1];candidate=P.parent;old=R/'chimera/plans/full-owned-device-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['run_checkpoint_build.py','supervision.py','exec_guard.c','exec_guard','source-manifest.json']:
 target=P/'controls'/name;assert not target.exists();shutil.copy2(candidate/name,target)
shutil.copy2(candidate/'build/gpu_build',P/'controls/gpu_build')
shutil.copy2(candidate/'ms-provenance.json',P/'provenance.json')
for name in ['launch_full.py','guard_full.py']:shutil.copy2(old/name,P/name)
p=P/'launch_full.py';s=p.read_text();s=s.replace("base=Path(s['base']);base.mkdir(exist_ok=False);import shutil", "import preflight\npreflight.verify(s)\nbase=Path(s['base']);base.mkdir(exist_ok=False);import shutil")
s=s.replace("job['continuation_guardian_pid']=guardian.pid;write(job)","job['continuation_guardian_pid']=guardian.pid;job['continuation_guardian_identity']=ident(guardian.pid);write(job)")
s=s.replace("if gt_live() and holders==[s['existing_GT_guardian_pid']]:", "if gt_live() and preflight.cohort_live(s) and holders==[s['existing_GT_guardian_pid']]:")
s=s.replace("if guardian.poll() is not None:raise RuntimeError('Guardian died before wrapper launch')", "if guardian.poll() is not None:raise RuntimeError('Guardian died before wrapper launch')\n  preflight.verify_resources(s)")
s=s.replace("status='complete' if rc==0 else 'interrupted' if requested else 'failed',returncode=rc", "status='complete' if rc==0 else 'checkpoint_stopped' if rc==3 else 'interrupted' if requested else 'failed',intentional_checkpoint_stop=(rc==3),returncode=rc")
p.write_text(s)
# All engine/control files copied here remain exact candidate bytes.
print('Copied reviewed controls; private launcher adds only resource/cohort checks and stop3 status')
