from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/lxq/graduate/ai-workspace/research/multivector'); A=R/'report-assets'; D=R/'report'
d=json.loads((D/'lotte-r10-95-chunks-data.json').read_text()); xs=d['chunk_results']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
f,ax=plt.subplots(1,2,figsize=(10,3.7),layout='constrained'); xx=np.arange(3)
for aa,key,rang,title in [(ax[0],'mean_QPS','QPS_range','B1 throughput (queries/s)'),(ax[1],'mean_round_p99_ms','p99_range_ms','Mean of round p99 (ms)')]:
 y=np.array([x[key] for x in xs]); lo=np.array([x[rang][0] for x in xs]); hi=np.array([x[rang][1] for x in xs]); aa.bar(xx,y,color=['#287f8e','#8292ac','#b3bac5']); aa.errorbar(xx,y,yerr=[y-lo,hi-y],fmt='none',ecolor='#263746',capsize=4); aa.set_xticks(xx,['chunks=1','chunks=4','chunks=5\n(upstream default)']); aa.set_title(title); aa.set_ylim(0,max(hi)*1.24); aa.grid(axis='y',alpha=.18); aa.set_axisbelow(True)
 for i,v in enumerate(y):aa.text(i,hi[i]+max(hi)*.035,f'{v:.2f}',ha='center')
f.suptitle('LoTTE Recall@10 >= 0.95 | CPU20 | 2931 queries x 3 rounds',fontsize=12)
f.savefig(A/'lotte-r10-95-chunks.png',dpi=180);plt.close(f)
rows=[]
for x in xs:
 c=x['point']['num_chunks'];rows.append(f"| {c}{'（上游默认）' if c==5 else ''} | {x['mean_recall10']:.6f} | **{x['mean_QPS']:.2f}** | {x['QPS_range'][0]:.2f}–{x['QPS_range'][1]:.2f} | {x['mean_latency_ms']:.3f} | {x['mean_round_p99_ms']:.3f} |")
screen=[]
for x in d['selection_screening']:
 p=x['point'];screen.append(f"| {x['label'][0]} | {p['nprobe']} / {p['k_refine']} / {p['k_full_bit']} / {p['cagra_itopk_size']} | {x['mean_recall10']:.6f} | {x['mean_QPS']:.2f} | {'达标' if x['target_met'] else '未达标'} |")
confirm=[]
for x in d['selection_confirmation']:confirm.append(f"| {x['label'][0]} | {x['mean_recall10']:.6f} | {x['mean_QPS']:.2f} | "+' / '.join(f"{r['qps']:.2f}" for r in x['rounds'])+' |')
text='''# LoTTE：Recall@10≥0.95 选参与 chunks 对比

**已完成。保留 chunks=1：585.25 QPS，Recall@10=0.958649；默认 chunks=5 为339.35 QPS。** CPU固定20线程，本轮没有修改C++/CUDA算法或重建索引。

## 实验平台与测试设置

| 项目 | 设置 |
|---|---|
| 平台 | DGX Spark；GB10 GPU（sm_121）、20核ARM CPU、128 GB LPDDR5X统一内存 |
| 数据集 | LoTTE pooled dev，2,428,854篇文档、266,205,513条文档向量 |
| 查询 | 2,931条，每条32个128维向量，合计93,792条向量 |
| 文档 | 每篇3–180条128维向量，平均109.6条 |
| 表示 | 发布数据为FP16；已有完整4bit索引；CPU精排读取完整4bit编码 |
| 正式测试 | B1，Top-10；全部2,931条查询×3轮，每轮预热32条；CPU20线程、亲和性0–19 |
| 召回与计时 | 全部查询对照FP32 MaxSim真值；三轮Recall@10、QPS各取算术平均；没有留出集 |
| 计时边界 | 预编码主机查询输入至主机Top-10结果完成，包含查询上传；索引加载、编码、预热、写盘排除 |
| 设备协调 | 全程使用 ~/device.lock 互斥锁并监测外部计算干扰 |

每轮QPS = 2,931 / 该轮同步搜索延迟总和。正式结果使用主程序，观测程序只用于下文32条查询的轨迹分析。

## 选参结果

先以全量查询单轮筛查4个邻点，参数顺序为 nprobe / k_refine / k_full_bit / CAGRA itopk；此时均为chunks=1。

| 点 | 参数 | Recall@10 | 单轮QPS | 资格 |
|---|---|---:|---:|---|
'''+ '\n'.join(screen)+'''

B与E的单轮QPS接近，分别补齐全量三轮后选E；E的chunk1结果直接复用于后续比较。

| 点 | 三轮平均Recall@10 | 三轮平均QPS | 各轮QPS |
|---|---:|---:|---|
'''+ '\n'.join(confirm)+'''

**选中 nprobe=128、k_refine=1000、k_full_bit=50、cagra_itopk_size=160。** 选择依据是本轮三轮均值；表中保留轮间波动。这里仅chunks=5是上游默认值，整组参数经过选参。

## 同参比较 chunks=1 / 4 / 5

| chunks | 平均Recall@10 | 平均QPS | 三轮QPS范围 | 平均延迟ms | 各轮p99均值ms |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

![chunks对比](../report-assets/lotte-r10-95-chunks.png)

误差线表示三轮最小—最大值。chunks=1的QPS分别是4的1.58倍、5的1.72倍，因此保留1。三轮范围用于显示波动；p99列是各轮p99的平均。

chunks拆分的是**一个查询的候选文档**。后续chunk的GPU评分可以与前一chunk的CPU精排重叠，但每个chunk也增加事件、等待、筛选及OpenMP调度，实际收益见下方轨迹。

'''
(D/'lotte-r10-95-chunks.md').write_text(text)
print([(x['point']['num_chunks'],x['mean_latency_ms'],x['mean_round_p99_ms']) for x in xs])
