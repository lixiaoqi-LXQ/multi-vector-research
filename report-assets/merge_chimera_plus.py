"""Update the one user report from complete fixed-parameter Chimera+ results."""
from pathlib import Path
import json,statistics,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
R=Path('/home/lxq/graduate/ai-workspace/research/multivector');D=R/'report';A=R/'report-assets';P=R/'chimera-plus'
new=json.loads((P/'formal-results.json').read_text());assert new['complete'] and len(new['groups'])==4
base10=json.loads((D/'lotte-r10-full-query-records.json').read_text())
base100=json.loads((R/'chimera/r100-v1/completion-summary.json').read_text());plaid100=json.loads((R/'plaid/ai-workspace/r100-native-device-lock-v2/final-acceptance.json').read_text())
look={}
for x in base10['records']:
 if x['label'] not in ['Chimera','PLAID native']:continue
 look[10,x['target_recall10'],x['label']]={'recall':x['full_recall10'],'rounds':x['rounds'],'point':x['point']}
for x in base100['groups']:look[100,x['target'],'Chimera']={'recall':x['mean_full_recall100'],'rounds':[dict(r,qps=r['QPS']) for r in x['rounds']],'point':x['point']}
for target,x in zip([.9,.95],plaid100['records']):look[100,target,'PLAID native']={'recall':x['mean_R100'],'rounds':x['rounds']}
for g in new['groups']:
 assert g['quality_pass'];look[g['k'],g['target'],'Chimera+']={'recall':g['mean_recall'],'rounds':g['rounds'],'point':g['point']}
for k in (10,100):
 for target in (.9,.95):assert look[k,target,'Chimera+']['point']==look[k,target,'Chimera']['point']
labels=[('Chimera','Chimera','#315b8a'),('Chimera+','Chimera+ GPU fullbit','#378579'),('PLAID native','PLAID official GPU','#cc8540')]
def mean(row,key):return statistics.mean(x[key] for x in row['rounds'])
def plot(k):
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 f,axes=plt.subplots(1,2,figsize=(12,4.1),layout='constrained')
 for j,(key,label,col) in enumerate(labels):
  for metric,ax,offset,width,hatch in [('qps',axes[0],(j-1)*.24,.23,''),('mean_ms',axes[1],(2*j-2.5)*.13,.12,''),('p99_ms',axes[1],(2*j-1.5)*.13,.12,'///')]:
   vs=[[r[metric] for r in look[k,t,key]['rounds']] for t in (.9,.95)];y=np.array([statistics.mean(v) for v in vs]);lo=np.array([min(v) for v in vs]);hi=np.array([max(v) for v in vs]);xx=np.arange(2)+offset
   ax.bar(xx,y,width,color=col,hatch=hatch,edgecolor='white',linewidth=.6,label=label if metric=='qps' else None,yerr=[y-lo,hi-y],capsize=2,error_kw={'elinewidth':.7})
   for x,z,h in zip(xx,y,hi):ax.annotate(f'{z:.0f}' if metric=='qps' else f'{z:.2f}',(x,h),xytext=(0,4),textcoords='offset points',ha='center',fontsize=7)
 for ax,title in zip(axes,['QPS','Mean latency and P99 (ms)']):
  ax.set_xticks([0,1],['0.90','0.95']);ax.set_xlabel(f'Recall@{k} target');ax.set_title(title);ax.set_ylim(0,ax.get_ylim()[1]*1.18);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
 axes[1].legend(handles=[Patch(facecolor='#999',label='Mean'),Patch(facecolor='#999',edgecolor='white',hatch='///',label='P99')],loc='upper left',frameon=False,ncol=2,fontsize=9)
 f.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=3,frameon=False);f.supxlabel('LoTTE 4bit | B1 | CPU budget20 | 2931 queries x 3 rounds | fixed search parameters\nBars: mean across rounds (including round P99); whiskers: round range',fontsize=9)
 name='lotte-throughput.png' if k==10 else 'lotte-throughput-r100.png';f.savefig(A/name,dpi=180);plt.close(f);return name
figs={k:plot(k) for k in (10,100)}
def table(k):
 lines=[f'| 系统 | 目标R{k} | 实际R{k} | 平均QPS | 平均延迟ms | 各轮P99均值ms |','|---|---:|---:|---:|---:|---:|']
 for t in (.9,.95):
  for key,label,col in labels:
   r=look[k,t,key];label='PLAID（官方GPU）' if key=='PLAID native' else key
   lines.append(f"| {label} | {t:.2f} | {r['recall']:.6f} | {mean(r,'qps'):.2f} | {mean(r,'mean_ms'):.3f} | {mean(r,'p99_ms'):.3f} |")
 return '\n'.join(lines)
p=D/'lotte-r10-brief.md';s=p.read_text();a=s.index('## Recall@10：');b=s.index('## Chimera瓶颈分析：',a)
gains={k:[mean(look[k,t,'Chimera+'],'qps')/mean(look[k,t,'Chimera'],'qps') for t in (.9,.95)] for k in (10,100)}
newsection='''## Recall@10：端到端性能

'''+f"**同参替换精排后，Chimera+在0.90/0.95目标下的QPS分别为原Chimera的{gains[10][0]:.2f}/{gains[10][1]:.2f}倍。**\n\n"+table(10)+f"\n\n![R10端到端QPS与平均/P99延迟](../report-assets/{figs[10]})\n\n"
newsection+='''每点全2,931条查询×3轮，参数不重选。平均延迟、QPS及P99分别取三轮均值，误差线为三轮范围。原Chimera结果来自前批正式测试，Chimera+为本轮同参测试。

## Chimera+：只替换完整4bit精排

**GPU直接读取现有4bit编码并计算MaxSim；路由、候选筛选、1bit评分及CPU最终Top-k不变。** GB10实际使用ATS主机页表访问已有编码/因子，没有复制第二份整库，也没有展开成FP32。

GPU保留CPU的FP32 FMA和归约顺序。104个合成候选、真实LoTTE四配置的22,780个候选评分（含预热）均与CPU逐位一致。正式计时关闭逐位复查，包含GPU精排、必要传输及完成同步。

| 返回k / 目标 | nprobe | refine | full-bit | CAGRA itopk | chunks |
|---|---:|---:|---:|---:|---:|
| 10 / 0.90 | 32 | 1000 | 20 | 64 | 1 |
| 10 / 0.95 | 128 | 1000 | 50 | 160 | 1 |
| 100 / 0.90 | 64 | 3000 | 300 | 96 | 1 |
| 100 / 0.95 | 256 | 3000 | 300 | 288 | 1 |

独立源码仓库：`/home/lxq/graduate/new-idea/multi-vector/Chimera`，分支`Chimera+`，提交`c97f24c`。`baseline`分支保留原ARM实验源码。

'''
s=s[:a]+newsection+s[b:];s=s.replace('## Chimera瓶颈分析：0.90与0.95','## 原Chimera瓶颈分析：0.90与0.95').replace('相同32条固定查询的诊断轨迹，单位ms/查询。Chimera主配置均为chunks1。','以下为原Chimera的32条固定查询轨迹，单位ms/查询，均为chunks1；本轮未采集Chimera+阶段轨迹。')
a=s.index('## Recall@100：');b=s.index('## 数据证据',a)
s=s[:a]+f"## Recall@100：端到端性能\n\n**Chimera+在0.90/0.95目标下的QPS分别为原Chimera的{gains[100][0]:.2f}/{gains[100][1]:.2f}倍。** 所有点均达标，搜索参数与原Chimera一致。\n\n"+table(100)+f"\n\n![R100端到端QPS与平均/P99延迟](../report-assets/{figs[100]})\n\n"+'PLAID参数：0.90为`2 / 4096 / 0.45`，0.95为`4 / 8192 / 0.45`（ncells / ndocs / threshold）。\n\n'+s[b:]
s=s.replace('## Chimera微调：chunks与CPU线程数','## 原Chimera微调：chunks与CPU线程数')
s=s.replace('| CPU任务粒度 | 合并小任务或复用线程池，减少反复进入OpenMP区域 |','| CPU任务粒度 | 原Chimera可减少OpenMP小任务开销；Chimera+已移除CPU精排 |')
s=s.replace('| 小缓冲区零拷贝 | 让GPU直接写共享候选/评分缓冲区，减少显式复制与往返同步 |','| UMA数据访问 | Chimera+已用ATS直接读取4bit编码；候选/评分小缓冲区的传输仍可优化 |')
s=s.replace('CPU已直接读取完整4bit文档编码；FP32展开仅用于构建。上述为待验证优化，本轮未修改搜索算法。','原Chimera由CPU读取完整4bit编码；Chimera+已将这段评分迁到GPU。FP32展开仅用于构建，其他优化方向本轮未改。')
s += '\n[Chimera+四组正式结果与每轮数据](../chimera-plus/formal-results.json) · [Chimera+源码与二进制身份](../chimera-plus/implementation-identity.json)\n'
p.write_text(s)
for l in re.findall(r'\]\(([^)]+)\)',s):
 if not l.startswith('http'):assert (p.parent/l).exists(),l
print('Updated same report and both mean/P99 charts; gains',gains)
