"""Plot interval-union stage accounting from existing NSYS evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path('/home/lxq/graduate/ai-workspace/research/multivector')
d=json.loads((ROOT/'report/lotte-r10-nsys-stage-projection.json').read_text())
records={x['label']:x for x in d['records']}
names=['chimera-R10p90','plaid-native-R10p90','plaid-gpuall-R10p90']
labels=['Chimera query','PLAID native rank','PLAID gpu-all rank']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(10.6,3.0),layout='constrained')
for i,name in enumerate(names):
 r=records[name]; n=r['outer_count'];gpu=r['GPU_activity_in_outer_ns']/n/1e6
 cpu=r.get('CPU_fullbit_union_ns',0)/n/1e6
 overlap=r.get('CPU_GPU_overlap_ns',0);assert overlap==0
 other=r['outer_union_ns']/n/1e6-gpu-cpu
 left=0
 for j,(v,color,label) in enumerate([(gpu,'#315b8a','GPU kernel / copy / memset'),(cpu,'#cc8540','CPU 4bit scoring'),(other,'#a8afb8','Other host / submission / wait')]):
  if v:
   ax.barh(i,v,left=left,color=color,label=label if i==0 else None,height=.58)
   if v>.42:ax.text(left+v/2,i,f'{v:.3f}',ha='center',va='center',color='white' if j==0 else 'black',fontsize=9)
   left+=v
 ax.text(left+.07,i,f'{left:.3f} ms',ha='left',va='center',fontsize=9)
ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.set_xlim(0,11)
ax.set_xlabel('Mean diagnostic time per query (ms)');ax.spines[['top','right','left']].set_visible(False)
ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
ax.legend(loc='lower right',frameon=False,fontsize=8)
fig.suptitle('Recall@10 target 0.90: measured stage accounting',fontsize=12)
fig.supxlabel('Same 32 calibration queries; scopes: Chimera query / PLAID rank (sorting + host lists included).\n'
               'Interval unions include memset; nested spans are not added. Diagnostic timing, separate from formal QPS.',fontsize=8)
for ext in ['svg','png']:fig.savefig(ROOT/f'report-assets/lotte-r10-breakdown-v1.{ext}',dpi=200)
plt.close(fig)
