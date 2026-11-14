from pathlib import Path
import json,numpy as np,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/lxq/graduate/ai-workspace/research/multivector');P=R/'chimera/profile-v2/r10p95-overlap-v1';D=R/'report';A=R/'report-assets'
ds=[json.loads((P/f'chimera-E-c{c}/NSYS-overlap-Khat-breakdown.json').read_text()) for c in (1,4)]
f,ax=plt.subplots(figsize=(9,3.3),layout='constrained');left=np.zeros(2)
keys=['GPU only','CPU fullbit only','GPU + CPU overlap','Other'];colors=['#287f8e','#dfac54','#8b7ab0','#c2c7ce'];part=[]
for d in ds:
 s=d['summary'];T=s['query_NVTX_ns'];G=s['GPU_kernel_copy_set_union_ns'];C=s['CPU_fullbit_union_ns'];O=s['GPU_CPU_fullbit_overlap_ns'];v=[G-O,C-O,O,T-G-C+O];assert min(v)>=0;assert sum(v)==T;part.append(np.array(v)/32e6)
for j,(lab,col) in enumerate(zip(keys,colors)):
 vals=[v[j] for v in part];ax.barh(['chunks=1','chunks=4'],vals,left=left,label=lab,color=col);left+=vals
ax.set_xlabel('Observed mean query span (ms), 32 diagnostic queries');ax.set_title('Chimera 0.95: CPU/GPU overlap and exclusive time');ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,-.23));ax.invert_yaxis();ax.spines[['top','right']].set_visible(False);f.savefig(A/'lotte-r10-95-overlap.png',dpi=180);plt.close(f)
rows=[]
for c,d in zip((1,4),ds):
 s=d['summary'];rows.append('| '+str(c)+' | '+' | '.join(f'{s[k]/32e6:.3f}' for k in ['query_NVTX_ns','GPU_kernel_copy_set_union_ns','CPU_fullbit_union_ns','GPU_CPU_fullbit_overlap_ns','wait_chunk_union_ns'])+' |')
api=[]
for c,d in zip((1,4),ds):
 rt=d['runtime_API_totals'];launch=sum(v['count'] for k,v in rt.items() if 'cudaLaunchKernel' in k);sync=sum(v['count'] for k,v in rt.items() if 'cudaEventSynchronize' in k);api.append(f"| {c} | {d['summary']['CPU_fullbit_call_count']/32:g} | {'50' if c==1 else '12 / 13 / 12 / 13'} | 50 | {launch/32:g} | {sync/32:g} |")
text='''## 0.95配置的瓶颈与 breakdown

以下两组追踪使用相同32条固定诊断查询；正式召回和QPS仍使用全部2,931条查询。观测版与主程序在各自chunks设置下的320个结果ID及FP32评分逐位一致。Nsight软件追踪用于解释执行组织，正式性能以上表为准。

| chunks | 查询跨度ms/条 | GPU活动ms/条 | CPU精排ms/条 | 两者重叠ms/条 | wait_chunk跨度ms/条 |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

![CPU/GPU时间分解](../report-assets/lotte-r10-95-overlap.png)

GPU活动包含kernel、memcpy及memset并集；图中将重叠独立计入一次。wait_chunk与GPU活动重叠，不能再次加到端到端时间中。

| chunks | CPU精排调用/查询 | 各chunk文档数 | 实际精排文档合计/查询 | kernel launch/查询 | event synchronize/查询 |
|---|---:|---|---:|---:|---:|
'''+ '\n'.join(api)+'''

**多分块确实实现了重叠，但CPU精排碎片化的代价更大。** chunks=4重叠约0.206ms/条，GPU活动时间基本持平；CPU精排区域由0.616ms增至1.898ms，抵消重叠收益。每chunk只有12或13篇文档，却仍进入20线程的OpenMP区域；调用次数增至4倍，同时增加kernel启动及事件同步。这个结果支持优先优化小任务的组织与调度。

两种设置实际都精排50篇文档，**本次变慢没有伴随额外文档精排量**。源码中分块采用累计候选阈值，其他参数下实际精排量仍应通过payload测量；这里的50来自真实轨迹。

追踪包含“可能未采集全部事件”的诊断告警；两组均核对到32个完整查询范围，CPU精排调用分别为32和128，文档payload与分块配置一致。时间占比按已采集事件计算。本轮没有有效DRAM带宽/算术吞吐饱和度计数，因此compute-bound或memory-bound仍缺少关键证据；已确认的是CPU精排粒度及CPU/GPU协同效率问题。

## UMA上可优化的空间

| 优先方向 | 当前证据与具体改法 | 验证目标 |
|---|---|---|
| 1. 降低CPU小任务开销 | 每chunk 12–13篇文档触发20线程OpenMP区域；可试持久线程池、合并小任务、按候选量选择并行粒度 | 保持候选及评分一致，降低CPU精排区域时间，再看端到端QPS |
| 2. 小型控制/评分缓冲区零拷贝 | 查询、候选计数/偏移/ID、每chunk 1bit评分仍有显式拷贝及同步；可试CPU/GPU可共同访问的映射缓冲区，由GPU直接写入 | 检查指针属性、结果一致及生产者—消费者顺序，减少拷贝API与往返同步 |
| 3. 复用事件与工作区 | 每chunk创建/销毁事件，启动和同步次数随chunks增加；可在索引工作区复用事件 | 减少host调度开销，保留完成边界 |
| 4. 扩大有效重叠范围 | 当前chunk重叠只覆盖文档评分阶段，路由和候选生成/精炼先完成 | 后续单独研究跨查询流水线；按新的吞吐协议测量 |

CPU精排已经直接访问主机侧完整4bit文档编码，无需每查询搬运整库FP32文档。UMA上的首个零拷贝原型应聚焦小型共享缓冲区与同步往返；共享物理内存本身不会消除生产者—消费者等待。上述均为后续优化方向，本轮未更改算法、内存布局或线程策略。

## 制品与执行状态

- [全量结果及每轮明细](lotte-r10-95-chunks-data.json)，包含原始接纳路径及哈希。
- [chunks=1追踪汇总](../chimera/profile-v2/r10p95-overlap-v1/chimera-E-c1/NSYS-overlap-Khat-breakdown.json)、[chunks=4追踪汇总](../chimera/profile-v2/r10p95-overlap-v1/chimera-E-c4/NSYS-overlap-Khat-breakdown.json)。
- 正式二进制SHA256：`c54539b5143f63ad96ab0118f165dded7861d9c5af180767cd74c66d7bd4945d`。本轮仅更新Python实验入口的设备锁/接纳口径及报告；沿用此前披露的ARM与正确性适配。
- 范围：LoTTE、Recall@10、CPU20；没有新build、R100或其他数据集实验。
'''
p=D/'lotte-r10-95-chunks.md';s=p.read_text().split('## 0.95配置的瓶颈与 breakdown')[0];p.write_text(s+text)
print('report and overlap figure written')
