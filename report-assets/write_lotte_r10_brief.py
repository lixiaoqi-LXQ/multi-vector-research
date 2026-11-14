from pathlib import Path
import json,statistics
ROOT=Path('/home/lxq/graduate/ai-workspace/research/multivector')
formal=json.loads((ROOT/'report/lotte-r10-full-query-records.json').read_text())
control=json.loads((ROOT/'chimera/fine-completion-v1/r10-controls-cal-v1/completion-summary.json').read_text())
rows=formal['records'];assert len(rows)==6 and control['complete']
lookup={r['configuration']:r for r in control['rows']}
gain=(lookup['cpu-4fast-chunk1']['calibration_qps']/lookup['cpu-20all-chunk1']['calibration_qps']-1)*100
text='''# LoTTE Recall@10 简报：端到端比较与瓶颈

## 实验平台与测试设置

| 项目 | 设置 |
|---|---|
| 实验平台 | NVIDIA DGX Spark，GB10 GPU（sm_121）、20个ARM物理CPU核，128 GB LPDDR5X统一内存 |
| 数据集 | LoTTE pooled dev；2,428,854篇文档、2,931条查询 |
| 搜索设置 | 4bit索引，B1逐条搜索，实际返回Top-10；目标Recall@10为0.90、0.95 |
| 正式测试 | 全部2,931条查询×3轮，每轮预热32条；主配置使用20线程 |
| 计时范围 | 预编码主机FP32查询→主机Top-10结果完成，包含查询上传；排除模型编码、索引加载、预热和写盘 |

| 对象 | 向量数量 | 每条向量维度 |
|---|---:|---:|
| 每条查询 | 固定 **32条** | 128维 |
| 每篇文档 | **3–180条**，平均 **109.6条** | 128维 |

全库文档合计 **266,205,513条向量**；2,931条查询合计 **93,792条向量**。

**结论：在已完成全量三轮的主配置中，R10≥0.90时，Chimera为811.8 QPS，PLAID最快放置为169.0 QPS，前者约4.80倍。R10≥0.95时，PLAID为54.2 QPS；Chimera冻结点的全量召回约0.9486，未达标。**

## 1. 全量端到端结果

每配置2,931条查询×3轮。QPS为三轮算术平均；表内R10亦为三轮平均。Recall@10和QPS均使用全部2,931条查询，是否达标按三轮平均全量R10判定。

| 系统 / 放置 | 目标R10 | 平均QPS | 全量R10 | 资格 |
|---|---:|---:|---:|---|
'''
for r in rows:
 text+=f"| {r['label']} | {r['target_recall10']:.2f} | {statistics.mean(x['qps'] for x in r['rounds']):.1f} | {r['full_recall10']:.6f} | {'达标' if r['qualification'] else '**未达标**'} |\n"
text+='''
![全量三轮QPS与尾延迟](../report-assets/lotte-r10-full-query-v2.png)

图中误差线为三轮最小—最大值，尾延迟为各轮p99的均值；斜线保留未达标点。主配置均为20线程。PLAID两种放置在每个目标下各8,793对结果ID/FP32评分逐位一致；Chimera未达标点未用于0.95目标的倍率比较。

## 2. 瓶颈与 breakdown

下面是0.90目标、相同32条校准查询的诊断追踪；与上面的全量正式QPS分开统计。

![R10代表配置的阶段分解](../report-assets/lotte-r10-breakdown-v1.png)

- **Chimera：CPU/GPU串行是明确问题。** 单chunk配置中，GPU活动占查询时间46.5%，CPU完整4bit评分占40.7%，两者重叠为0。GPU的CAGRA路由与1bit token评分共占kernel累计时间69.4%。FP32文档展开用于构建，CPU搜索评分读取的是4bit编码。
- **PLAID：主机路径是主要改进位置。** 原生`rank`有63.5%的时间没有GPU活动。gpu-all使32条查询的`rank`从306.673ms降至207.362ms，而GPU活动并集仅从111.809ms变为107.345ms；99.311ms差额中94.847ms来自非GPU活动区间缩短。大数组上GPU后，PID长度/偏移等CPU元数据准备仍保留。
- **compute / memory归因：当前未证实算力或LPDDR带宽饱和。** 本轮缺有效DRAM和CPU PMU计数，异常occupancy已剔除；能够确认的是阶段串行及主机整理、提交和同步开销。

分解按区间并集统计，包含kernel、copy和memset。Chimera范围为完整query；PLAID为`rank`，包含排序与主机列表转换，范围外是查询上传/half转换、最终Top-k切片和调用方同步。图中的灰色区间包括主机工作及等待，不等同纯CPU计算。

## 3. 补充短测：线程与chunk

保持Chimera参数`nprobe=32 / refine=1000 / full=20 / itopk=64`，相同100条校准查询，每配置一次；六点R10均为0.910。

![CPU线程与chunk短测](../report-assets/lotte-r10-controls-v1.png)

'''
text+=f'4个快核短测为1134.2 QPS，相比20核的847.6 QPS高{gain:.1f}%，显示线程组织仍有优化空间；这是单次cal100结果，未替换上面的全量正式比较。20线程下chunk从1增加到2/4，QPS反而降至790.8/544.1；本轮没有记录额外评分工作量Khat，此处只比较整体效果，不将变化归为纯重叠收益。\n\n'
text+='''## 证据与收尾

[完整报告与参数](lotte-r10-report.md) · [六组正式数据](lotte-r10-full-query-records.json) · [阶段投影及SQL](lotte-r10-nsys-stage-projection.json) · [硬件采样原始命令](lotte-r10-profile-commands.md) · [六点短测验收](../chimera/fine-completion-v1/r10-controls-cal-v1/completion-summary.json)

本轮性能测试已结束。交付本简报后按要求停止后续实验，R100、其他数据集和消融均不继续。
'''
(ROOT/'report/lotte-r10-brief.md').write_text(text)
print('Wrote report/lotte-r10-brief.md',len(text),'characters')
