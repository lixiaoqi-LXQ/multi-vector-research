import sys,os
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from child_supervision import SupervisedPopen
p=SupervisedPopen([sys.executable,sys.argv[2],sys.argv[3]])
Path(sys.argv[4]).write_text(str(p.pid))
p.wait()
