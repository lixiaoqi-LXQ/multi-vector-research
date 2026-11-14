"""Plot six single-calibration diagnostics separately from formal results."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/lxq/graduate/ai-workspace/research/multivector')
d=json.loads((ROOT/'chimera/fine-completion-v1/r10-controls-cal-v1/completion-summary.json').read_text());assert d['complete'] and len(d['rows'])==6
r={x['configuration']:x for x in d['rows']}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(10.6,3.0),layout='constrained')
for ax,keys,labels,title in [
 (axes[0],['cpu-1core7-chunk1','cpu-4fast-chunk1','cpu-10fast-chunk1','cpu-20all-chunk1'],['1 fast','4 fast','10 fast','20 all'],'CPU threads (chunks=1)'),
 (axes[1],['cpu-20all-chunk1','cpu-20all-chunk2','cpu-20all-chunk4'],['1','2','4'],'Chunks (20 threads)')]:
 vals=[r[k]['calibration_qps'] for k in keys]
 bars=ax.bar(labels,vals,color=['#315b8a' if v!=max(vals) else '#378579' for v in vals],width=.6)
 for bar,v in zip(bars,vals):ax.annotate(f'{v:.1f}',(bar.get_x()+bar.get_width()/2,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
 ax.set_ylim(0,1300);ax.set_ylabel('Calibration QPS');ax.set_title(title,fontsize=11);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.supxlabel('Same 100 calibration queries, one repetition; all Recall@10 = 0.910.\n'
 'Fixed nprobe=32 / refine=1000 / full=20; diagnostic results, not full-query formal QPS.',fontsize=8)
for ext in ['svg','png']:fig.savefig(ROOT/f'report-assets/lotte-r10-controls-v1.{ext}',dpi=200)
plt.close(fig)
