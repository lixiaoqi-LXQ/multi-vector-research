import sys,time,json,os
from pathlib import Path
sys.path.insert(0,'/home/lxq/graduate/ai-workspace/research/multivector/plaid')
from future_telemetry import Telemetry
p=Path('/home/lxq/graduate/ai-workspace/research/multivector/plaid/ai-workspace/r100-device-lock-v1/native-cal-continuation-nvml.jsonl')
ctx={'stage':'native-cal-dispatch-paused','known_child_pids':[648455,648460],'child_pid':648460}
m=Telemetry(p,ctx).start()
try:
 while Path('/proc/648460').exists():
  if (Path('/home/lxq/graduate/ai-workspace/research/multivector/plaid/ai-workspace/r100-device-lock-v1/cal-output/native-cal/selection-frozen.json')).exists():break
  time.sleep(.5)
finally:m.close()
