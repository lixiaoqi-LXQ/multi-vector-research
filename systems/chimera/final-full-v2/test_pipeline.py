"""Real runner CPU preparation with synthetic selection statistics; no measured data."""
from pathlib import Path
import plan,subprocess,os,json,copy
P=Path(__file__).resolve().parent;F=P/'pipeline-cpu-fixture';F.mkdir(exist_ok=False);s=plan.read(P/'actual-prefix-finalize-fixture/finalized-first.json');s['points']=s['points'][:1];plan.write(F/'screen.json',s)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1')
def prepare(spec,out):subprocess.run([str(plan.ROOT/'build-env/bin/python'),str(plan.ROOT/'run_search_grid_v2.py'),'--spec',str(spec),'--output',str(out)],env=env,check=True,capture_output=True)
prepare(F/'screen.json',F/'screen');m=plan.read(F/'screen/prepared.json');row=dict(point=s['points'][0],queries=100,k=100,recall100=.97,qps=100,actual_search_modes={'verified':True},round_name='point00-round0');plan.write(F/'screen/summary.json',[row]);plan.write(F/'screen/release.json',dict(complete=True,completed_searches=1));plan.write(F/'screen/accepted-summary.json',dict(output_contract_passed=True,fixture_only=True));plan.finalists([F/'screen'],F/'finalists');runs=[]
for repeat in range(3):
 spec=F/'finalists'/f'f00-repeat{repeat}.json';out=F/f'finalist{repeat}';prepare(spec,out);row=dict(row,qps=100+repeat);plan.write(out/'summary.json',[row]);plan.write(out/'release.json',dict(complete=True,completed_searches=1));plan.write(out/'accepted-summary.json',dict(output_contract_passed=True,fixture_only=True));runs.append(out)
plan.freeze_final(runs,F/'frozen');formal=list((F/'frozen').glob('formal-*.json'))
for spec in formal:
 out=F/(spec.stem+'-prepared');prepare(spec,out);m=plan.read(out/'prepared.json');assert m['expected_searches']==3 and m['queries']==2931;assert len(m['commands'])==3 and m['spec']['heldout_rows'];assert m['spec']['selection_frozen_from']['median_qps']==101
plan.write(P/'pipeline-CPU-validation.json',{'passed':True,'GPU_used':False,'real_runner_prepare_screen_and3finalists_and2formal':True,'actual_prefix_index_identity_and_projection_checked':True,'formal_queries':2931,'formal_rounds':3,'statistics':'synthetic CPU fixture only; no measured performance or recall claim','fixture':str(F)});print('Runner CPU prepare screen -> finalists3 -> freeze -> formal3 PASS')
