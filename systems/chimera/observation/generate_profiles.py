"""Prepare six actual-work profile captures from one final calibrated point. No GPU."""
import argparse,json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser();p.add_argument('--final-spec',required=True);p.add_argument('--out',required=True);a=p.parse_args();s=json.loads(Path(a.final_spec).read_text());assert len(s['points'])==1,'Freeze one eligible calibrated point first';out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
for count in (1,2,4,6,8,10):
 folder=out/f'chunk{count}';folder.mkdir();point=dict(s);point.update(binary=str(R/'observation/build/gpu_search'),phase='profile_diagnostic_not_QPS',points=[dict(s['points'][0],num_chunks=count)]);file=folder/'point.json';file.write_text(json.dumps(point,indent=2));subprocess.run([str(R/'build-env/bin/python'),str(R/'prepare_profile.py'),'--spec',str(file),'--out',str(folder/'capture'),'--queries','32'],check=True)
 sqlite=folder/'trace.sqlite';export=['/usr/local/cuda/bin/nsys','export','--type=sqlite','--output',str(sqlite),str(folder/'capture/trace.nsys-rep')];parse=[str(R/'build-env/bin/python'),str(R/'observation/parse_khat.py'),'--sqlite',str(sqlite),'--kfull',str(point['points'][0]['k_full_bit']),'--out',str(folder/'actual-Khat.json')];(folder/'offline-export-parse.json').write_text(json.dumps({'export_command':export,'parse_command':parse,'no_commands_executed':True},indent=2))
print(out)
