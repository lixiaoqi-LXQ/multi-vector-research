"""Hold inherited device-lock description until exact supervised compute is gone."""
from pathlib import Path
import argparse,os,json,time
p=argparse.ArgumentParser();p.add_argument('--fd',type=int,required=True);p.add_argument('--state',required=True);a=p.parse_args();state=Path(a.state)
def live(i):
 if not i:return False
 try:
  raw=Path(f"/proc/{i['pid']}/stat").read_text();v=raw[raw.rfind(')')+2:].split();return v[0]!='Z' and int(v[19])==i['start_ticks']
 except (FileNotFoundError,ProcessLookupError):return False
while True:
 try:s=json.loads(state.read_text())
 except FileNotFoundError:time.sleep(.2);continue
 active=live(s.get('current_child')) or live(s.get('native'))
 if not active and (s.get('status') in ('complete','failed','interrupted','lock_busy') or not live(s.get('driver'))):break
 time.sleep(.2)
os.close(a.fd)
