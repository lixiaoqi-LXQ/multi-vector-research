import sys
sys.path.insert(0,'/home/lxq/graduate/ai-workspace/research/multivector/plaid')
from child_supervision import SupervisedPopen
p=SupervisedPopen([sys.executable,'/home/lxq/graduate/ai-workspace/research/multivector/plaid/child-supervision-cpu-check/leaf.py',sys.argv[1]])
p.wait()
