"""Render current main-report figures from accepted evidence; no experiments."""
from pathlib import Path
import json,statistics,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
R=Path('/home/lxq/graduate/ai-workspace/research/multivector');D=R/'report';A=R/'report-assets'
raw=json.loads((D/'lotte-r10-full-query-records.json').read_text());look={(x['label'],x['target_recall10']):x for x in raw['records']}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
f,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for j,(key,label,col) in enumerate([('Chimera','Chimera','#315b8a'),('PLAID native','PLAID official GPU','#cc8540')]):
 for metric,ax,offset,width,hatch in [('qps',axes[0],(j-.5)*.3,.3,''),('mean_ms',axes[1],(2*j-1.5)*.19,.17,''),('p99_ms',axes[1],(2*j-0.5)*.19,.17,'///')]:
  vs=[[r[metric] for r in look[key,t]['rounds']] for t in (.9,.95)];y=np.array([statistics.mean(v) for v in vs]);lo=np.array([min(v) for v in vs]);hi=np.array([max(v) for v in vs]);xx=np.arange(2)+offset
  ax.bar(xx,y,width,color=col,hatch=hatch,edgecolor='white',linewidth=.7,label=label if metric=='qps' else None,yerr=[y-lo,hi-y],capsize=2,error_kw={'elinewidth':.8})
  for x,z,h in zip(xx,y,hi):ax.annotate(f'{z:.1f}' if metric=='qps' else f'{z:.2f}',(x,h),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
for ax,title in zip(axes,['QPS','Mean latency and P99 (ms)']):
 ax.set_xticks([0,1],['0.90','0.95']);ax.set_xlabel('Recall@10 target');ax.set_title(title);ax.set_ylim(0,ax.get_ylim()[1]*1.18);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
axes[1].legend(handles=[Patch(facecolor='#999999',label='Mean'),Patch(facecolor='#999999',edgecolor='white',hatch='///',label='P99')],loc='upper left',frameon=False,ncol=2,fontsize=9)
f.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2,frameon=False);f.supxlabel('LoTTE 4bit | B1 | CPU20 | 2931 queries x 3 rounds\nBars: mean across rounds (including per-round P99); whiskers: round range',fontsize=9);f.savefig(A/'lotte-throughput.png',dpi=180);plt.close(f)
old={x['label']:x for x in json.loads((D/'lotte-r10-nsys-stage-projection.json').read_text())['records']};c=old['chimera-R10p90'];new=json.loads((R/'chimera/profile-v2/r10p95-overlap-v1/chimera-E-c1/NSYS-overlap-Khat-breakdown.json').read_text())['summary'];n=32
rs=[('R10 target 0.90',c['outer_union_ns']/n/1e6,c['GPU_activity_in_outer_ns']/n/1e6,c['CPU_fullbit_union_ns']/n/1e6),('R10 target 0.95',new['query_NVTX_ns']/n/1e6,new['GPU_kernel_copy_set_union_ns']/n/1e6,new['CPU_fullbit_union_ns']/n/1e6)]
f,ax=plt.subplots(figsize=(9,3.2),layout='constrained')
for i,(label,T,G,C) in enumerate(rs):
 left=0
 for val,col,name in [(G,'#287f8e','GPU activity'),(C,'#dfac54','CPU fullbit'),(T-G-C,'#c2c7ce','Other host / wait')]:
  ax.barh(i,val,left=left,color=col,label=name if i==0 else None)
  ax.text(left+val/2,i,f'{val:.3f}',ha='center',va='center',fontsize=10);left+=val
ax.set_yticks(range(2),[r[0] for r in rs]);ax.invert_yaxis();ax.set_xlabel('Observed mean query span (ms)');ax.set_title('Chimera breakdown: chunks=1, CPU20, 32 diagnostic queries');f.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False);f.savefig(A/'lotte-bottlenecks.png',dpi=180);plt.close(f)
p=D/'lotte-r10-brief.md';s=p.read_text();a=s.index('## 瓶颈分析：');b=s.index('## chunks对比：',a);part=s[a:b]
part=part.replace('## 瓶颈分析：0.90与0.95','## Chimera瓶颈分析：0.90与0.95').replace('配置均使用CPU20、B1。Chimera顺序：`nprobe / refine / full / itopk / chunks`；PLAID：`ncells / ndocs / threshold`。','配置均使用CPU20、B1，参数顺序：`nprobe / refine / full / itopk / chunks`。')
part='\n'.join(x for x in part.split('\n') if not x.startswith('| PLAID') and not x.startswith('- **PLAID：'))
part=part.replace('GPU活动包含kernel、copy、memset；其余区间包含主机整理、提交和等待。Chimera计完整query；PLAID计rank，上传、half转换、Top-k切片及调用方同步在rank之外。PLAID无CPU fullbit阶段，其主机工作计入其余区间。','GPU活动包含kernel、copy、memset；其余区间包含主机整理、提交和等待。统计完整query范围，重叠为0。')
s=s[:a]+part+s[b:];s=s.replace('误差线为三轮范围；p99为各轮p99的均值。','左图为QPS，右图实心柱为平均延迟、斜线柱为P99；误差线为三轮范围，P99取各轮P99的均值。')
p.write_text(s)
for link in re.findall(r'\]\(([^)]+)\)',s):
 if not link.startswith('http'):assert (p.parent/link).exists(),link
assert 'PLAID' not in s[s.index('## Chimera瓶颈分析：'):s.index('## chunks对比：')]
print('Updated main report: QPS + mean/P99 latency; Chimera-only breakdown.')
