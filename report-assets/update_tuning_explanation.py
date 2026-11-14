from pathlib import Path
import json,statistics,re,shutil
R=Path('/home/lxq/graduate/ai-workspace/research/multivector');D=R/'report';p=D/'lotte-r10-brief.md';s=p.read_text()
ctrl=json.loads((R/'chimera/fine-completion-v1/r10-controls-cal-v1/completion-summary.json').read_text());rs=ctrl['rows'];cpus=[x for x in rs if x['chunks']==1]
a=s.index('## chunks对比：');b=s.index('## UMA优化方向',a);existing=s[a:b].replace('## chunks对比：0.95，同参','### chunks：0.95全量三轮')
# Put measured contrasts and explanation in one place, avoiding duplicated causal paragraphs.
existing=existing.split('**原因：')[0]
section='''## Chimera微调：chunks与CPU线程数

**结论：并行收益取决于任务粒度；当前小候选精排下，增加chunks或线程数会增加调度成本。**

### CPU线程与chunks：已有100条查询短测

固定`nprobe32 / refine1000 / full20 / itopk64`，同100条查询、各一次，所有Recall@10均为0.910。以下是诊断短测，未替代全量三轮结果。

| CPU线程 | 核集合 | chunks | QPS | 平均延迟ms |
|---:|---|---:|---:|---:|
'''
for x in rs:
 section+=f"| {x['threads']} | {x['CPU_affinity']} | {x['chunks']} | {x['calibration_qps']:.2f} | {x['calibration_mean_ms']:.3f} |\n"
shutil.copyfile(R/'report-assets/lotte-r10-controls-v1.png',R/'report-assets/lotte-tuning.png')
section+='''
![线程与chunks微调](../report-assets/lotte-tuning.png)

**CPU线程：4个快核短测最快，20线程反而下降。** CPU精排按文档分配任务，使用`OpenMP schedule(dynamic,1)`；此处full目标仅20篇，任务数少，增线程很快遇到并行度上限。更多线程增加任务领取、唤醒及结束屏障开销；文档长度不等还带来尾部等待。20核配置还包含不同性能的CPU核，因此线程数与核类型同时变化。源码支持这些机制，但短测没有分别隔离每项成本；不能据此判定“多核本身更慢”。

**chunks：20线程下，1→2→4的QPS持续下降。** 这组短测未记录实际精排量，下面用0.95的完整三轮和轨迹进一步解释。

'''+existing+'''
### 为什么开启流水线没有更快？

**流水线节省的重叠时间，小于切碎任务增加的时间。** 对本次32条诊断查询，chunks1→4时：

| 改变 | 每查询时间变化 |
|---|---:|
| CPU精排区域 | +1.281 ms |
| GPU活动 | +0.012 ms |
| 其余区间 | +0.022 ms |
| 新增CPU/GPU重叠 | −0.206 ms |
| 总查询跨度 | **+1.110 ms** |

两种设置实际都精排50篇；四块分别12/13/12/13篇，每块仍进入20线程OpenMP区域。GPU启动从23增至29次/query，CPU精排调用从1增至4次。事件同步次数增加，但wait_chunk总时间下降，变慢主要体现在CPU精排区域。

此外，chunks只让后续GPU文档评分与前一块CPU精排重叠；路由、候选生成和精炼不在这段流水线中。块数太多会缩小每块工作，并增加流水线启动、排空及调度成本。

**两种调参相互影响。** chunks越多，每次交给CPU的文档越少，越容易出现“线程多于任务”。更大的候选规模可能适合更多块或线程；当前证据支持保持主配置chunks1、CPU20，并将减少CPU小任务开销作为优化方向。4线程是否适合作为正式配置，需要同一目标下重新做全量三轮验证。

'''
s=s[:a]+section+s[b:]
# Incorporate the completed official R100 measurements into the same report.
a=s.index('## Recall@100：');b=s.index('## 数据证据',a)
c=json.loads((R/'chimera/r100-v1/completion-summary.json').read_text());pl=json.loads((R/'plaid/ai-workspace/r100-native-device-lock-v2/final-acceptance.json').read_text());assert c['complete'] and pl['accepted'] and pl['official_native_only']
r100='''## Recall@100：端到端性能

**两系统均已完成全量三轮。** CPU20、B1、实际Top-100，无留出集。

| 系统 | 目标R100 | 平均R100 | 平均QPS | 平均延迟ms | 各轮P99均值ms |
|---|---:|---:|---:|---:|---:|
'''
for i,g in enumerate(c['groups']):
 r100+=f"| Chimera | {g['target']:.2f} | {g['mean_full_recall100']:.6f} | {g['mean_round_QPS']:.2f} | {statistics.mean(x['mean_ms'] for x in g['rounds']):.3f} | {statistics.mean(x['p99_ms'] for x in g['rounds']):.3f} |\n"
 q=pl['records'][i];r100+=f"| PLAID（官方GPU） | {g['target']:.2f} | {q['mean_R100']:.6f} | {q['mean_QPS']:.2f} | {statistics.mean(x['mean_ms'] for x in q['rounds']):.3f} | {q['mean_round_p99_ms']:.3f} |\n"
r100+='\nChimera参数：0.90为`64 / 3000 / 300 / 96 / 1`，0.95为`256 / 3000 / 300 / 288 / 1`。PLAID参数：0.90为`2 / 4096 / 0.45`，0.95为`4 / 8192 / 0.45`（ncells / ndocs / threshold）。\n\n'
s=s[:a]+r100+s[b:]
s=s.replace('[Recall@100正式结果](../chimera/r100-v1/completion-summary.json)','[Chimera R100结果](../chimera/r100-v1/completion-summary.json) · [PLAID R100结果](../plaid/ai-workspace/r100-native-device-lock-v2/final-acceptance.json) · [线程/chunks短测明细](../chimera/fine-completion-v1/r10-controls-cal-v1/completion-summary.json)')
s=s.replace('| 搜索设置 | 4bit索引，B1逐条搜索，实际返回Top-10；目标Recall@10为0.90、0.95 |','| 搜索设置 | 4bit索引，B1逐条搜索；分别实际返回Top-10和Top-100；召回目标0.90、0.95 |').replace('| 计时范围 | 预编码主机FP32查询→主机Top-10结果完成，包含查询上传；排除模型编码、索引加载、预热和写盘 |','| 计时范围 | 预编码主机FP32查询→主机Top-k结果完成，包含查询上传；排除模型编码、索引加载、预热和写盘 |')
p.write_text(s)
for l in re.findall(r'\]\(([^)]+)\)',s):
 if not l.startswith('http'):assert (p.parent/l).exists(),l
print('Same report updated: CPU and chunk tuning, causal limits, completed official R100.')
