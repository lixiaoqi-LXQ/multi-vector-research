from pathlib import Path
import json,shlex,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent.parent;D=R/'report';A=R/'report-assets';x=json.loads((P/'summary.json').read_text());agg=x['NCU_aggregate_time_weighted'];ms=x['GPU_activity_sampling']
# Keep visualization on the main new GPU-stage question, with explicit distinct units.
f,axes=plt.subplots(1,2,figsize=(9,3.1),layout='constrained');xx=np.arange(3);rr=x['NCU_launches']
axes[0].bar(xx,[r['FP32_TFLOPs_per_s'] for r in rr],color='#378579');axes[0].set_ylabel('Counted FP32 TFLOP/s');axes[0].set_title('GPU full-bit scoring: arithmetic')
w=.35;axes[1].bar(xx-w/2,[r['L2_request_GB_per_s'] for r in rr],w,label='Measured L2 requests',color='#315b8a');axes[1].bar(xx+w/2,[r['payload_effective_GB_per_s_estimate'] for r in rr],w,label='Estimated useful payload',color='#dfac54');axes[1].set_ylabel('GB/s');axes[1].set_title('Two distinct bandwidth measures');axes[1].legend(fontsize=8,frameon=False);axes[1].set_ylim(0,25)
for ax in axes:ax.set_xticks(xx,['Launch 1','Launch 2','Launch 3']);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
f.suptitle('Chimera+ | R100 target 0.95 | NCU: 3 full-bit kernel launches',fontsize=11);f.supxlabel('L2 requests and useful payload are not physical LPDDR traffic.',fontsize=9);f.savefig(A/'lotte-gpu-profile.png',dpi=180);plt.close(f)
sec='''## Chimera+：R100 0.95的GPU利用与带宽

参数仍为`nprobe256 / refine3000 / full300 / itopk288 / chunks1`。仅32条查询短追踪，NCU采3次精排kernel，无全量重测。

| 指标 | 结果 | 范围与口径 |
|---|---:|---|
'''
for label,name in [('GPU活跃','GR Active [Throughput %]'),('SM活跃','SMs Active [Throughput %]'),('SM指令发射','SM Issue [Throughput %]')]:sec+=f"| {label} | {ms[name]['mean_percent']:.1f}% | 整条查询区间内硬件采样均值 |\n"
sec+=f"| FP32算术吞吐 | **{agg['FP32_TFLOPs_per_s']:.2f} TFLOP/s** | GPU精排，按2×FFMA+FADD+FMUL计数除以kernel时间 |\n"
sec+=f"| L2请求带宽 | **{agg['L2_request_GB_per_s']:.2f} GB/s** | GPU精排，L2请求字节/时间，实测 |\n"
sec+=f"| 代码/因子有效带宽 | **约{agg['payload_effective_GB_per_s_estimate']:.2f} GB/s** | GPU精排，按唯一数据量68B/token估算 |\n"
sec+='''| LPDDR实际带宽 | 未取得 | 本机NCU与NSYS未暴露该计数 |

**GPU忙碌不等于FP32算力吃满。** 整体GPU活跃约95%，但指令发射约14%；精排本身有位解包、共享内存访问和归约，FP32吞吐不能代表全部GPU工作。

三次精排的FP32吞吐为0.69–1.31 TFLOP/s，L2请求带宽11.24–18.82 GB/s；表中均按总工作量/总kernel时间汇总。有效带宽按FFMA计数反推的实际token量估算，每次约3.4–3.6万token，没有用全库平均文档长度代替。

![GPU精排计算与带宽短采样](../report-assets/lotte-gpu-profile.png)

**新增GPU精排已不是主要耗时。** 32条轨迹中，精排kernel平均0.213ms/查询，整条查询平均2.822ms，约占7.6%；1bit token评分、CAGRA路由、1bit文档聚合分别约0.653、0.573、0.415ms/查询。

L2请求与有效数据量都不等于LPDDR实际流量，不能据此计算LPDDR带宽利用率。本机官方LPDDR峰值为[273 GB/s](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)。诊断追踪和NCU replay不替代前文正式三轮性能。

<details>
<summary>采集命令与原始计数</summary>

'''
for phase in ['nsys','ncu']:
 cmd=json.loads((P/phase/'tree-config.json').read_text())['launch_command'];sec+='```sh\n'+shlex.join(cmd)+'\n```\n\n'
sec+='''NCU原始CSV中GPU精排三次计数：

```csv
ID,Duration_us,FFMA,FADD,FMUL,L2_requested_Mbyte
'''
for r in rr:sec+=f"{r['launch']},{r['kernel_duration_us']:.6f},{r['FFMA']:.0f},{r['FADD']:.0f},{r['FMUL']:.0f},{r['L2_requested_bytes']/1e6:.6f}\n"
sec+='''```

[原始NCU CSV](../chimera-plus/profile-r100-p95/ncu/raw.csv) · [汇总与计算公式](../chimera-plus/profile-r100-p95/summary.json)

</details>

'''
p=D/'lotte-r10-brief.md';s=p.read_text();heading='## Chimera+：R100 0.95的GPU利用与带宽'
if heading in s:
 a=s.index(heading);b=s.index('## Chimera+：只替换完整4bit精排',a);s=s[:a]+sec+s[b:]
else:s=s.replace('## Chimera+：只替换完整4bit精排',sec+'## Chimera+：只替换完整4bit精排',1)
p.write_text(s);print('Added small GPU-profile section to the existing single report.')
