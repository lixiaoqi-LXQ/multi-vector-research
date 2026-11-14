import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from child_supervision import SupervisedPopen
p=SupervisedPopen([sys.executable,sys.argv[2],sys.argv[3],sys.argv[4]])
sys.exit(p.wait())
