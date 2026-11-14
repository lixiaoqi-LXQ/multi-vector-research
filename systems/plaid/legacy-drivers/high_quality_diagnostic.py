"""Run only under the shared device lock after resource handoff."""
import json, os, subprocess, sys, time
from pathlib import Path
root=Path(__file__).resolve().parent
data=Path('/home/lxq/graduate/data/multivector-colbertv2/lotte-pooled-dev')
index=Path('/home/lxq/graduate/data/multivector-colbertv2/indexes/plaid-lotte-prefix10000-b4-km20')
gt=Path('/home/lxq/graduate/data/multivector-colbertv2/groundtruth/lotte-prefix-10000')
splits=Path('/home/lxq/graduate/data/multivector-colbertv2/splits/lotte-calibration-seed42')
env=dict(os.environ)
env['PATH']=str(root/'.venv/bin')+os.pathsep+env.get('PATH','')
env.update(PYTHONPATH=str(root/'faiss-gpu-env/lib/python3.13/site-packages'),LD_LIBRARY_PATH=str(root/'faiss-gpu-env/lib'),CUDA_HOME='/usr/local/cuda-13.0',TORCH_CUDA_ARCH_LIST='12.1',MAX_JOBS='2',TORCH_EXTENSIONS_DIR=str(root/'torch-extensions-cuda'))
def run(cmd,log):
    with open(root/log,'w') as f: subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
idle=[]
for _ in range(10):
    value=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    idle.append({'time':time.time(),'utilization_gpu':int(value)})
    time.sleep(1)
(root/'km20-idle-before.json').write_text(json.dumps(idle,indent=2))
assert max(x['utilization_gpu'] for x in idle)<=5, 'Device had active compute in idle window'
common=['--data',str(data),'--index',str(index)]
if not (index/'build-provenance.json').exists():
    run([sys.executable,str(root/'runner.py'),'build',*common,'--documents',str(data/'documents.npy.partial'),'--docs','10000','--bits','4','--kmeans-iters','20','--build-device','cuda','--kmeans-device','cuda','--threads','4'],'build-n10000-b4-km20.log')
points=[(32,8192),(64,16384),(128,40000),(16384,40000)]
for cells,ndocs in points:
    name=f'cuda-n10000-km20-c{cells}-d{ndocs}-t0'
    output=root/'results'/(name+'-screen100')
    run(['taskset','-c','7',sys.executable,str(root/'runner.py'),'search',*common,'--device','cuda','--queries','100','--query-rows',str(splits/'calibration-rows.npy'),'--rounds','1','--warmup','10','--threads','1','--ncells',str(cells),'--ndocs',str(ndocs),'--threshold','0','--output',str(output)],name+'-screen.log')
    run([sys.executable,str(root/'summarize.py'),str(output),'--groundtruth',str(gt)],name+'-screen-summary.log')
    summary=json.loads((output/'summary.json').read_text())
    recall=summary['rounds'][0]['recall@100']
    print(name,'screen R100',recall,flush=True)
    if recall>=.90:
        full=root/'results'/(name+'-full2931')
        run(['taskset','-c','7',sys.executable,str(root/'runner.py'),'search',*common,'--device','cuda','--queries','2931','--rounds','3','--warmup','32','--threads','1','--ncells',str(cells),'--ndocs',str(ndocs),'--threshold','0','--output',str(full)],name+'-full.log')
        run([sys.executable,str(root/'summarize.py'),str(full),'--groundtruth',str(gt),'--heldout-rows',str(splits/'heldout-rows.npy')],name+'-full-summary.log')
        print((full/'summary.json').read_text(),flush=True)
        break
else:
    print('4bit exhaustive compressed-scoring ceiling on first100 queries below R100 .90; preserve as failed quality point.',flush=True)
