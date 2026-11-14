"""Explicit reviewed detached top-level start; no work in default plan mode."""
from pathlib import Path
import argparse,json,hashlib,os,subprocess,sys
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--approved-spec-sha256');a=p.parse_args();spec=P/'spec.json';digest=hashlib.sha256(spec.read_bytes()).hexdigest();command=[sys.executable,str(P/'launch_full.py'),'--run']
if not a.run:print(json.dumps({'prepared_only':True,'GPU_used':False,'spec_sha256':digest,'detached_top_level_command':command}));raise SystemExit
if a.approved_spec_sha256!=digest:raise ValueError('Exact reviewed spec SHA required')
handle=P/'detached-start.json';log=P/'detached-launcher.log'
if handle.exists() or log.exists():raise FileExistsError('Retain previous start evidence; use a reviewed new attempt')
with open(log,'x') as f:child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
try:
 text=Path(f'/proc/{child.pid}/stat').read_text();start_ticks=int(text[text.rfind(')')+2:].split()[19])
except FileNotFoundError:start_ticks=None
record={'launcher_pid':child.pid,'launcher_start_ticks':start_ticks,'launcher_command':command,'spec_sha256':digest,'detached_new_session':True,'wrapper_native_PDEATHSIG_scope':'owned wrapper/native track launcher; top-level launcher detached from invoking shell'}
with open(handle,'x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
directory=os.open(P,os.O_RDONLY|os.O_DIRECTORY);os.fsync(directory);os.close(directory);print(json.dumps(record))
