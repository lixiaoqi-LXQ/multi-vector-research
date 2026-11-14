from pathlib import Path
import json,statistics,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path('/home/lxq/graduate/ai-workspace/research/multivector');D=R/'report';A=R/'report-assets'
raw=json.loads((D/'lotte-r10-full-query-records.json').read_text());rows=[x for x in raw['records'] if x['label'] in ['Chimera','PLAID native']];look={(x['label'],x['target_recall10']):x for x in rows}
ratios=[statistics.mean(r['qps'] for r in look['Chimera',t]['rounds'])/statistics.mean(r['qps'] for r in look['PLAID native',t]['rounds']) for t in (.9,.95)]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
f,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for j,(key,label,col) in enumerate([('Chimera','Chimera','#315b8a'),('PLAID native','PLAID official GPU','#cc8540')]):
 for ax,m,title in zip(axs,['qps','p99_ms'],['QPS','Mean of round p99 (ms)']):
  vs=[[r[m] for r in look[key,t]['rounds']] for t in (.9,.95)];y=np.array([statistics.mean(v) for v in vs]);lo=np.array([min(v) for v in vs]);hi=np.array([max(v) for v in vs]);xx=np.arange(2)+(j-.5)*.3
  ax.bar(xx,y,.3,color=col,label=label,yerr=[y-lo,hi-y],capsize=3);ax.set_xticks([0,1],['0.90','0.95']);ax.set_xlabel('Recall@10 target');ax.set_title(title);ax.set_ylim(0,max(ax.get_ylim()[1],max(hi)*1.2));ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
  if m=='qps':
   for x,z,h in zip(xx,y,hi):ax.annotate(f'{z:.1f}',(x,h),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
f.legend(*axs[0].get_legend_handles_labels(),loc='outside upper center',ncol=2,frameon=False);f.supxlabel('LoTTE 4bit | B1 | CPU20 | 2931 queries x 3 rounds; whiskers: round range',fontsize=9);f.savefig(A/'lotte-throughput.png',dpi=180);plt.close(f)
old={x['label']:x for x in json.loads((D/'lotte-r10-nsys-stage-projection.json').read_text())['records']};new=json.loads((R/'chimera/profile-v2/r10p95-overlap-v1/chimera-E-c1/NSYS-overlap-Khat-breakdown.json').read_text())['summary']
ch=old['chimera-R10p90'];pl=old['plaid-native-R10p90'];n=32
r=[('Chimera 0.90',ch['outer_union_ns']/n/1e6,ch['GPU_activity_in_outer_ns']/n/1e6,ch['CPU_fullbit_union_ns']/n/1e6),('Chimera 0.95',new['query_NVTX_ns']/n/1e6,new['GPU_kernel_copy_set_union_ns']/n/1e6,new['CPU_fullbit_union_ns']/n/1e6),('PLAID 0.90',pl['outer_union_ns']/n/1e6,pl['GPU_activity_in_outer_ns']/n/1e6,0)]
f,axes=plt.subplots(1,2,figsize=(10,3.2),layout='constrained')
for ax,subset,title in [(axes[0],r[:2],'Chimera: full query'),(axes[1],r[2:],'PLAID official GPU: rank')]:
 for i,(name,T,G,C) in enumerate(subset):
  left=0
  for val,col,lab in [(G,'#287f8e','GPU activity'),(C,'#dfac54','CPU fullbit'),(T-G-C,'#c2c7ce','Other host / wait')]:
   ax.barh(i,val,left=left,color=col,label=lab if i==0 else None)
   if val>.18:ax.text(left+val/2,i,f'{val:.3f}',ha='center',va='center',fontsize=9)
   left+=val
 ax.set_yticks(range(len(subset)),[x[0] for x in subset]);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('Observed mean diagnostic span (ms)')
f.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False);f.suptitle('32 diagnostic queries, CPU20; interval unions',fontsize=12);f.savefig(A/'lotte-bottlenecks.png',dpi=180);plt.close(f)
p=D/'lotte-r10-brief.md';s=p.read_text();s='\n'.join(l for l in s.split('\n') if not l.startswith('| PLAID gpu-all |'))
s=s.replace('Chimera分别比PLAID最快已测配置快4.80/10.79倍。',f'Chimera分别比官方PLAID GPU路径快{ratios[0]:.2f}/{ratios[1]:.2f}倍。')
s=s.replace('PLAID native / gpu-all','PLAID（官方GPU）').replace('PLAID native','PLAID（官方GPU）').replace('lotte-r10-full-query-v3.png','lotte-throughput.png').replace('lotte-r10-breakdown-v2.png','lotte-bottlenecks.png')
s=s.replace('**PLAID：0.90下主要机会在主机路径。** gpu-all缩短的99.31ms中，94.85ms来自非GPU活动区间。0.95尚无阶段轨迹。','**PLAID：0.90下主机路径占比较高。** rank中63.5%的时间无GPU活动；0.95尚无阶段轨迹。')
s=s.replace('## Recall@10：端到端性能','PLAID统一采用官方GPU路径，调用原生`IndexScorer.rank`。\n\n## Recall@10：端到端性能')
p.write_text(s)
assert 'gpu-all' not in s
for l in re.findall(r'\]\(([^)]+)\)',s):
 if not l.startswith('http'):assert (p.parent/l).exists(),l
state=R/'state.json';d=json.loads(state.read_text());d['reporting_baseline']='official PLAID GPU native only; gpu-all retained as historical evidence and excluded from report';state.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print('official-only report updated; ratios',ratios)
