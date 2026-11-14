from pathlib import Path
import json,shlex,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent.parent;D=R/'report';A=R/'report-assets';d=json.loads((P/'flow-summary.json').read_text());stages=['GPU route','GPU candidate/LUT','GPU 1bit scoring','GPU 4bit scoring'];g=d['stages'];labels=['Routing','Candidate/LUT','1bit scoring','4bit scoring']
f,axes=plt.subplots(1,2,figsize=(10,3.4),layout='constrained');xx=np.arange(4);cols=['#315b8a','#8b7ab0','#dfac54','#378579']
for ax,key,scale,title,ylabel in [(axes[0],'FP32_FLOPs',1e9,'GPU arithmetic: entire query','GFLOPs per query'),(axes[1],'L2_requested_bytes',1e6,'L2 requests: entire query','MB per query')]:
 y=[g[k][key]/scale for k in stages];ax.bar(xx,y,color=cols);ax.set_xticks(xx,labels,rotation=12);ax.set_title(title);ax.set_ylabel(ylabel);ax.set_ylim(0,max(y)*1.2);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 for x,v in zip(xx,y):ax.annotate(f'{v:.3f}'if scale==1e9 else f'{v:.1f}',(x,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
f.suptitle('Chimera+ R100 target0.95 | All24 GPU kernels of one complete query',fontsize=12);f.supxlabel('End-to-end normalization includes CPU, transfers and synchronization. L2 requests are not DRAM traffic.',fontsize=8);f.savefig(A/'lotte-gpu-profile.png',dpi=180);plt.close(f)
s='''## Chimera+：R100 0.95全流程GPU利用与带宽

固定`nprobe256 / refine3000 / full300 / itopk288 / chunks1`、CPU20。复用32条查询的正常轨迹，再对其中第一条完整查询采全部24次GPU kernel，覆盖路由、候选筛选、1bit评分、4bit精排。

**统计整个搜索调用，分母包含CPU、传输和同步，不再只看精排kernel。**

| 全流程指标 | 结果 | 统计方式 |
|---|---:|---|
'''
s+=f"| 正常端到端耗时 | {d['normal_end_to_end_ms']:.3f} ms | 同一条查询的正常search计时，未用NCU replay耗时 |\n"
s+=f"| 全GPU kernel FP32运算量 | {d['all_kernels_FP32_FLOPs']/1e9:.3f} GFLOP/查询 | 硬件指令计数，2×FFMA+FADD+FMUL |\n"
s+=f"| **端到端有效FP32吞吐** | **约{d['estimated_end_to_end_FP32_TFLOPs']:.3f} TFLOP/s** | 全GPU运算量 ÷ 端到端时间，估算 |\n"
s+=f"| 全GPU kernel L2请求量 | {d['all_kernels_L2_requested_bytes']/1e6:.1f} MB/查询 | 所有24次kernel的请求字节计数 |\n"
s+=f"| **端到端L2请求带宽** | **约{d['estimated_end_to_end_L2_request_GB_s']:.1f} GB/s** | 全L2请求量 ÷ 端到端时间，估算 |\n"
s+=f"| **下层内存读取请求带宽** | **约{d['estimated_end_to_end_lower_memory_read_request_GB_s']:.1f} GB/s** | L2读取未命中扇区×32B ÷ 端到端时间，估算 |\n"
s+='''| LPDDR实际读写总带宽 | 未直接取得 | 本机工具缺少DRAM计数 |

32条正常查询区间内，GPU活跃平均94.6%、SM活跃85.7%、SM指令发射14.1%。这些是全流程采样；上表吞吐是单条完整查询的计数估算，正式QPS仍使用前面的全量三轮结果。

![全流程GPU运算与L2请求分布](../report-assets/lotte-gpu-profile.png)

**路由和候选阶段占据大量L2流量，只测精排会明显低估全流程。** GPU精排仅约6.3MB L2请求，而整条查询约969.6MB。

L2请求带宽不是LPDDR总线带宽。读取未命中估算未包含写回，且一致性缓存也可能响应请求，不能换算成DRAM利用率；[NVIDIA对32B扇区与L2未命中的说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)和[计数不等于DRAM流量的说明](https://docs.nvidia.com/nsight-compute/ComputeTriage/)支持这一口径。

算术和内存计数分别来自两次短采集；都核对了24个kernel顺序，并与正常轨迹同一查询的Top-100 ID/FP32评分逐位一致。GPU拷贝、memset和CPU阶段的时间保留在端到端分母中，CPU算术与整数/位操作未计入GPU FP32 FLOPs。

<details>
<summary>全流程采集命令与原始计数</summary>

'''
for phase in ['ncuflow','memflow']:
 cmd=json.loads((P/phase/'tree-config.json').read_text())['launch_command'];s+='```sh\n'+shlex.join(cmd)+'\n```\n\n'
s+='```csv\nGPU_stage,FP32_FLOPs,L2_requested_bytes,L2_read_miss_request_bytes_estimate\n'
for k in stages:s+=f"{k},{g[k]['FP32_FLOPs']:.0f},{g[k]['L2_requested_bytes']:.0f},{g[k]['estimated_lower_memory_read_request_bytes']:.0f}\n"
s+='''```

[全24次kernel算术原始CSV](../chimera-plus/profile-r100-p95/ncuflow/raw.csv) · [全24次kernel内存原始CSV](../chimera-plus/profile-r100-p95/memflow/raw.csv) · [全流程汇总与公式](../chimera-plus/profile-r100-p95/flow-summary.json)

</details>

'''
p=D/'lotte-r10-brief.md';t=p.read_text();a=t.index('## Chimera+：R100 0.95的GPU利用与带宽');b=t.index('## Chimera+：只替换完整4bit精排',a);t=t[:a]+s+t[b:];p.write_text(t);print('Replaced kernel-only headline analysis with full-flow counts and end-to-end normalization.')
