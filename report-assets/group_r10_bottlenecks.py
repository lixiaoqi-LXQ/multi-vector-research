from pathlib import Path
import json,numpy as np,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/lxq/graduate/ai-workspace/research/multivector');D=R/'report';A=R/'report-assets'
old={x['label']:x for x in json.loads((D/'lotte-r10-nsys-stage-projection.json').read_text())['records']}
p=R/'chimera/profile-v2/r10p95-overlap-v1';new=json.loads((p/'chimera-E-c1/NSYS-overlap-Khat-breakdown.json').read_text())['summary'];c4=json.loads((p/'chimera-E-c4/NSYS-overlap-Khat-breakdown.json').read_text())['summary']
rows=[]
for name,label,target,scope in [('chimera-R10p90','Chimera',.9,'query'),('plaid-native-R10p90','PLAID native',.9,'rank'),('plaid-gpuall-R10p90','PLAID gpu-all',.9,'rank')]:
 x=old[name];n=x['outer_count'];rows.append(dict(label=label,target=target,scope=scope,T=x['outer_union_ns']/n/1e6,G=x['GPU_activity_in_outer_ns']/n/1e6,C=x.get('CPU_fullbit_union_ns',0)/n/1e6,O=x.get('CPU_GPU_overlap_ns',0)/n/1e6))
rows.insert(1,dict(label='Chimera',target=.95,scope='query',T=new['query_NVTX_ns']/32e6,G=new['GPU_kernel_copy_set_union_ns']/32e6,C=new['CPU_fullbit_union_ns']/32e6,O=new['GPU_CPU_fullbit_overlap_ns']/32e6))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
f,axes=plt.subplots(1,2,figsize=(11,3.3),layout='constrained')
for ax,rs,title in [(axes[0],rows[:2],'Chimera: full query'),(axes[1],rows[2:],'PLAID: rank, R10 target 0.90')]:
 for i,x in enumerate(rs):
  vals=[x['G']-x['O'],x['C']-x['O'],x['O'],x['T']-x['G']-x['C']+x['O']];left=0
  for j,(v,color,lab) in enumerate(zip(vals,['#287f8e','#dfac54','#8b7ab0','#c2c7ce'],['GPU only','CPU fullbit only','Overlap','Other host / wait'])):
   ax.barh(i,v,left=left,color=color,label=lab if i==0 else None)
   if v>.18:ax.text(left+v/2,i,f'{v:.3f}',ha='center',va='center',fontsize=9)
   left+=v
 ax.set_yticks(range(len(rs)),[f"R10 {x['target']:.2f}" if x['label']=='Chimera' else x['label'].replace('PLAID ','') for x in rs]);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('Observed mean diagnostic span (ms)')
f.legend(*axes[0].get_legend_handles_labels(),ncol=4,loc='outside lower center',frameon=False,fontsize=9);f.suptitle('Recall@10 bottlenecks: main configurations, CPU20, 32 diagnostic queries',fontsize=12)
f.savefig(A/'lotte-r10-breakdown-v2.png',dpi=180);plt.close(f)
brief=D/'lotte-r10-brief.md';s=brief.read_text();bak=D/'lotte-r10-brief-before-grouping.md'
if not bak.exists():bak.write_text(s)
head=s.split('## 2. 瓶颈与 breakdown')[0]
chunks=s.split('## 3. 0.95配置：chunks全量比较与UMA分析\n')[1].split('### 0.95配置的瓶颈与 breakdown')[0]
uma=s.split('### UMA上可优化的空间')[1].split('## 4. 历史补充短测')[0]
end=s.split('## 4. 历史补充短测：线程与chunk')[1]
section='''## 2. 瓶颈配置与 breakdown：0.90 / 0.95

### 对应配置

两档目标的瓶颈配置集中如下；主配置均为CPU20、B1。Chimera参数顺序为 `nprobe / refine / full / itopk / chunks`，PLAID为 `ncells / ndocs / threshold`。

| 系统 / 放置 | R10目标0.90配置 | R10目标0.95配置 | 32条诊断轨迹 |
|---|---|---|---|
| Chimera | 32 / 1000 / 20 / 64 / 1 | 128 / 1000 / 50 / 160 / 1 | 两档均已采集；另测0.95 chunks4 |
| PLAID native | 1 / 1024 / 0.45 | 4 / 4096 / 0.45 | 已采集0.90；0.95尚无轨迹 |
| PLAID gpu-all | 1 / 1024 / 0.45 | 4 / 4096 / 0.45 | 已采集0.90；0.95尚无轨迹 |

### 两档主配置的阶段耗时

以下为相同32条固定诊断查询的已采集轨迹；正式召回与QPS使用上节全部2,931条查询×3轮。Chimera两档均为chunks=1。

| 系统 / 放置 | R10目标 | 观测范围 | 总跨度ms/条 | GPU活动ms/条 | CPU完整4bit评分ms/条 | 重叠ms/条 | 其余区间ms/条 |
|---|---:|---|---:|---:|---:|---:|---:|
'''
for x in rows:
 section+=f"| {x['label']} | {x['target']:.2f} | {x['scope']} | {x['T']:.3f} | {x['G']:.3f} | {x['C']:.3f} | {x['O']:.3f} | {x['T']-x['G']-x['C']+x['O']:.3f} |\n"
section+='''| PLAID native | 0.95 | rank | — | — | — | — | — |
| PLAID gpu-all | 0.95 | rank | — | — | — | — | — |

![0.90与0.95瓶颈配置并列](../report-assets/lotte-r10-breakdown-v2.png)

GPU活动为kernel、copy、memset的区间并集；其余区间包含主机整理、提交及等待。PLAID没有独立CPU fullbit阶段，表中该列为0；其主机工作计入其余区间。Chimera范围为完整query；PLAID为原生rank，包含排序和主机列表转换，查询上传/half转换、Top-k切片和调用方同步在rank之外。

**Chimera两档主配置都存在CPU/GPU串行。** 0.90中GPU活动占46.5%、CPU精排占40.7%；0.95中分别为55.2%、32.8%，重叠均为0。0.95精排目标从20篇升到50篇，路由nprobe从32升到128；这组参数变化同时增加GPU和CPU区域耗时。各组来自独立软件追踪，正式速度比较以上节QPS为准。

**PLAID在0.90的主机路径开销明确。** native rank中63.5%的时间无GPU活动；gpu-all使32条rank总跨度从306.673ms降到207.362ms，GPU活动仅从111.809ms降到107.345ms。99.311ms差额中94.847ms来自非GPU活动区间缩短。0.95已有正式QPS，但尚无对应阶段追踪，暂不作该档瓶颈归因。

**compute / memory判断：当前证据指向执行组织和主机开销，尚缺有效DRAM带宽及CPU PMU计数来判断饱和度。** 异常occupancy未参与判断。0.95追踪保留“可能未采集全部事件”的告警；32个query及各stage调用计数已核对，阶段时间限定为已采集轨迹。

### 0.95多分块对照：为什么默认chunks=5没有更快

固定0.95选中参数，只改变chunks。chunks1与4的观测版均与各自主程序的320个结果ID/FP32评分逐位一致。

| chunks | 总跨度ms/条 | GPU活动ms/条 | CPU精排ms/条 | 重叠ms/条 | wait_chunk跨度ms/条 | 每query CPU精排调用 / 文档数 |
|---|---:|---:|---:|---:|---:|---|
| 1 | 1.879 | 1.037 | 0.616 | 0.000 | 0.291 | 1次 / 50篇 |
| 4 | 2.989 | 1.049 | 1.898 | 0.206 | 0.119 | 4次 / 12、13、12、13篇 |

![0.95分块的重叠与开销](../report-assets/lotte-r10-95-overlap.png)

**多分块产生重叠，但CPU精排碎片化代价更大。** 两种设置都精排50篇；chunks4每次仅12–13篇，却进入20线程OpenMP区域。CPU区域由0.616ms增至1.898ms，超过0.206ms重叠收益。kernel launch从23增至29次/query，event synchronize从1增至4次；同步次数增加，但wait_chunk总跨度下降，变慢不能归为等待时间增加。图中重叠只计一次，wait_chunk不另加到端到端时间。

## 3. chunks正式性能对比（0.95，同参）

'''
s=head+section+chunks+'\n## 4. UMA优化方向\n'+uma+'\n## 5. 历史补充短测：线程与chunk'+end
s=s.replace('实际收益见下方轨迹。','对应的阶段轨迹见第2节。')
brief.write_text(s)
# Link detailed report to unified comparison, keeping historical provenance.
p=D/'lotte-r10-report.md';t=p.read_text();t=t.replace('其中已合并0.95选参、chunks=1/4/5全量三轮、CPU/GPU breakdown与UMA优化方向。','其中已合并0.95选参、chunks=1/4/5全量三轮、CPU/GPU breakdown与UMA优化方向；0.90与0.95的瓶颈配置及阶段耗时已集中在同一节并排比较。');p.write_text(t)
for link in re.findall(r'\]\(([^)]+)\)',s):
 if not link.startswith('http'):assert (brief.parent/link).exists(),link
print('grouped 0.90/0.95 configs and bottlenecks in section2; links verified')
