"""Authorized future-search phase transition; frozen index builder is unchanged."""
import hashlib,json,os,sys,time
from pathlib import Path
manifest=Path('/home/lxq/graduate/ai-workspace/research/multivector/plaid/scale-runs/lotte-full-b4-km4-gpu-all-payload/phase-transition.json')
m=json.loads(manifest.read_text())
assert sys.argv[1:]==m['search_phase']['argv'], 'Unexpected search phase argv'
for path,expected in m['search_phase']['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected, 'Search source changed before launch: '+path
env=dict(os.environ);env['UMA_RUN_GPU_TESTS']='1';env['PLAID_SEARCH_PHASE_MANIFEST']=str(manifest)
worker=m['search_phase']['worker']
receipt={'time':time.time(),'pid':os.getpid(),'source_verified':True,'phase_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'argv_before_exec':sys.argv,'argv_after_exec':[worker,*sys.argv[1:]],'UMA_RUN_GPU_TESTS':'1'}
(manifest.parent/'search-phase-start.json').write_text(json.dumps(receipt,indent=2))
os.execve(sys.executable,[sys.executable,worker,*sys.argv[1:]],env)
