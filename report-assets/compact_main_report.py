from pathlib import Path
import json,re
R=Path('/home/lxq/graduate/ai-workspace/research/multivector');D=R/'report';p=D/'lotte-r10-brief.md';s=p.read_text()
backup=R/'memory/lotte-main-before-compact.md'
if not backup.exists():backup.write_text(s)
settings=s.split('## 实验平台与测试设置')[1].split('**结论：')[0]
formal=s.split('| 系统 / 放置 | 目标R10')[1].split('![全量三轮QPS')[0]
configs=s.split('| 系统 / 放置 | R10目标0.90配置')[1].split('### 两档主配置')[0]
stages=s.split('| 系统 / 放置 | R10目标 | 观测范围')[1].split('![0.90与0.95')[0]
chunks=s.split('| chunks | 平均Recall@10')[1].split('![chunks对比]')[0]
text='''# LoTTE 实验报告

## 实验平台与设置
'''+settings+'''
召回与QPS均统计全部2,931条查询，取三轮算术平均；无留出集。每轮QPS为查询数除以同步搜索总耗时。

## Recall@10：端到端性能

**结论：目标0.90/0.95下，Chimera分别比PLAID最快已测配置快4.80/10.79倍。**

| 系统 / 放置 | 目标R10'''+formal+'''
![端到端QPS与p99](../report-assets/lotte-r10-full-query-v3.png)

误差线为三轮范围；p99为各轮p99的均值。旧Chimera未达标点保留在原始证据中。

## 瓶颈分析：0.90与0.95

配置均使用CPU20、B1。Chimera顺序：`nprobe / refine / full / itopk / chunks`；PLAID：`ncells / ndocs / threshold`。

| 系统 / 放置 | R10目标0.90配置'''+configs+'''
### 阶段分解

相同32条固定查询的诊断轨迹，单位ms/查询。Chimera主配置均为chunks1。

| 系统 / 放置 | R10目标 | 观测范围'''+stages+'''
![两档配置瓶颈](../report-assets/lotte-r10-breakdown-v2.png)

GPU活动包含kernel、copy、memset；其余区间包含主机整理、提交和等待。Chimera计完整query；PLAID计rank，上传、half转换、Top-k切片及调用方同步在rank之外。PLAID无CPU fullbit阶段，其主机工作计入其余区间。

- **Chimera：两档都存在CPU/GPU串行。** 0.95增大路由与精排规模后，GPU和CPU区域耗时均增加。
- **PLAID：0.90下主要机会在主机路径。** gpu-all缩短的99.31ms中，94.85ms来自非GPU活动区间。0.95尚无阶段轨迹。
- **算力或带宽是否饱和：证据尚不足。** 当前明确的问题是执行组织与主机开销；缺有效DRAM带宽及CPU PMU计数。软件轨迹时间用于定位，正式性能看上节QPS。

## chunks对比：0.95，同参

**结论：保留chunks1，默认5更慢。** 参数固定为`128 / 1000 / 50 / 160`，仅改变chunks。

| chunks | 平均Recall@10'''+chunks+'''
![chunks正式性能](../report-assets/lotte-r10-95-chunks.png)

| chunks | CPU精排文档/查询 | CPU区域ms/查询 | CPU/GPU重叠ms/查询 |
|---|---|---:|---:|
| 1 | 50篇，一次调用 | 0.616 | 0 |
| 4 | 12/13/12/13篇，四次调用 | 1.898 | 0.206 |

**原因：精排量仍为50篇，任务变碎的代价超过重叠收益。** 每chunk仅12–13篇仍调用20线程；GPU活动基本持平，CPU区域明显变长。同步次数增多，但wait_chunk时间下降，不能归为等待增加。

## UMA优化方向

| 方向 | 下一步 |
|---|---|
| CPU任务粒度 | 合并小任务或复用线程池，减少反复进入OpenMP区域 |
| 小缓冲区零拷贝 | 让GPU直接写共享候选/评分缓冲区，减少显式复制与往返同步 |
| 事件复用 | 复用工作区事件，减少创建与销毁 |
| 跨查询协同 | 单独测试路由与前一查询CPU精排的流水线吞吐 |

CPU已直接读取完整4bit文档编码；FP32展开仅用于构建。上述为待验证优化，本轮未修改搜索算法。

## Recall@100：进度与已完成结果

**Chimera两档正式三轮已完成；PLAID正在选参。** 继续使用相同LoTTE、CPU20、B1，实际返回Top-100。

| 系统 | 目标R100 | 三轮平均R100 | 三轮平均QPS |
|---|---:|---:|---:|
| Chimera | 0.90 | 0.917085 | 431.16 |
| Chimera | 0.95 | 0.952890 | 314.43 |
| PLAID native / gpu-all | 0.90、0.95 | 进行中 | 进行中 |

Chimera参数：0.90为`64 / 3000 / 300 / 96 / 1`，0.95为`256 / 3000 / 300 / 288 / 1`（顺序同上）。PLAID完成后直接补充本报告。

## 数据证据

[Recall@10正式结果](lotte-r10-full-query-records.json) · [0.95选参与chunks明细](lotte-r10-95-chunks-data.json) · [0.90阶段数据](lotte-r10-nsys-stage-projection.json) · [0.95阶段数据](../chimera/profile-v2/r10p95-overlap-v1/E-c1-c4-overlap-breakdown.json) · [Recall@100正式结果](../chimera/r100-v1/completion-summary.json)
'''
# Verify actual R100 parameters rather than copying assumed values.
for goal in ('0.90','0.95'):
 q=json.loads((R/f'chimera/r100-v1/formal-r100-{goal}-full3.json').read_text())['points'][0]
 print(goal,q)
p.write_text(text)
for l in re.findall(r'\]\(([^)]+)\)',text):
 if not l.startswith('http'):assert (p.parent/l).exists(),l
print('single report compacted; links verified')
