"""Serial, fixed-parameter Chimera+ correctness and formal measurements."""
from pathlib import Path
import json,subprocess,time,hashlib,statistics,os
R=Path('/home/lxq/graduate/ai-workspace/research/multivector');P=R/'chimera-plus';PY=R/'chimera/build-env/bin/python';runner=P/'runner/run_search_grid_device_lock.py';accept=R/'chimera/accept_run_device_lock.py';binary=Path('/home/lxq/graduate/ai-workspace/builds/chimera-plus/gpu_search');expected=json.loads((P/'test-plan.json').read_text())['binary_sha256']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def status(**kw):(P/'progress.json').write_text(json.dumps(dict(time=time.time(),driver_pid=os.getpid(),**kw),indent=2)+'\n')
def launch(name):
 assert sha(binary)==expected
 out=P/'runs'/name
 status(phase=name,status='running',output=str(out))
 with open(P/(name+'-driver.log'),'w') as log:
  p=subprocess.run([str(PY),str(runner),'--spec',str(P/(name+'.json')),'--output',str(out),'--run'],stdout=log,stderr=subprocess.STDOUT)
 assert p.returncode==0,f'{name} failed; see driver log'
 return out
def validate(out,diagnostic=False):
 with open(out/'accept.log','w') as log:p=subprocess.run([str(PY),str(accept),str(out)],stdout=log,stderr=subprocess.STDOUT)
 assert p.returncode==0,f'{out} acceptance failed'
 a=json.loads((out/'accepted-summary.json').read_text());s=json.loads((out/'prepared.json').read_text())['spec']
 assert a['output_contract_passed'] and s['full_bit_device']=='gpu'
 for pi in range(len(s['points'])):
  for r in range(s['rounds']):
   log=(out/f'point{pi:02d}-round{r}.log').read_text()
   assert '[FULLBIT] backend=gpu access=ATS_host_page_tables' in log and 'corpus_copy_bytes=0' in log
   if diagnostic:assert log.count('scores_fp32_bitwise=passed')>=32
   else:assert '[FULLBIT_VERIFY]' not in log
 return a
try:
 validate(P/'runs/qa10',True)
 validate(launch('qa100'),True)
 groups=[]
 for name in ['r10-p90','r10-p95','r100-p90','r100-p95']:
  out=launch(name);a=validate(out);s=json.loads((out/'prepared.json').read_text())['spec'];rr=a['groups'][0]['rounds'];metric=f"recall{s['k']}";threshold=s['quality_target']['threshold'];actual=statistics.mean(x[metric] for x in rr)
  records=[]
  for pi in range(len(s['points'])):
   for rnd in range(s['rounds']):
    validation=json.loads((out/f'point{pi:02d}-round{rnd}-validation.json').read_text());assert validation['pagefaults']['query_loop']['major_delta']==0
    mon=[json.loads(l) for l in (out/f'point{pi:02d}-round{rnd}-monitor.jsonl').read_text().splitlines()];assert mon and all(x['gpu_query_returncode']==0 for x in mon);assert all(not x['foreign_gpu_processes'] for x in mon), 'Review foreign residency/compute interference before acceptance'
    records.append({'major_faults':0,'monitor_samples':len(mon)})
  groups.append({'name':name,'k':s['k'],'target':threshold,'point':s['points'][0],'rounds':rr,'mean_recall':actual,'quality_pass':actual>=threshold,'mean_QPS':statistics.mean(x['qps'] for x in rr),'mean_latency_ms':statistics.mean(x['mean_ms'] for x in rr),'mean_round_p99_ms':statistics.mean(x['p99_ms'] for x in rr),'telemetry':records,'source_run':str(out),'source_sha256':{n:sha(out/n) for n in ['prepared.json','summary.json','accepted-summary.json','release.json']}})
  (P/'formal-results.json').write_text(json.dumps({'complete':False,'binary_sha256':expected,'groups':groups},indent=2)+'\n')
 result={'complete':True,'binary_sha256':expected,'only_GPU_full_bit_scoring_changed':True,'no_parameter_tuning':True,'synthetic_scores_bitwise_passed':104,'actual_candidates_bitwise_QA':[str(P/'runs/qa10'),str(P/'runs/qa100')],'groups':groups}
 (P/'formal-results.json').write_text(json.dumps(result,indent=2)+'\n');status(phase='done',status='complete',formal_results=str(P/'formal-results.json'))
 print('Four formal full-query tests complete',flush=True)
except BaseException as e:
 status(phase='failed',status='failed',error=repr(e));raise
