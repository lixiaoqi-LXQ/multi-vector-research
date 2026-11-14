from pathlib import Path
import sqlite3,json,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Patch
from matplotlib.font_manager import FontProperties
P=Path('/home/lxq/graduate/ai-workspace/research/multivector');A=P/'report-assets';D=P/'report';db=P/'chimera-plus/profile-r100-p95/nsys/trace.sqlite';c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
cn=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');mono=FontProperties(family='DejaVu Sans Mono');plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'path'})
qstart,qend=c.execute("select start,end from NVTX_EVENTS where text='Chimera/query_local_row' order by start limit 1").fetchone()
ks=c.execute('select s.value,k.start,k.end,k.gridX,k.blockX,k.correlationId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id where k.start>=? and k.end<=? order by k.start',(qstart,qend)).fetchall();assert len(ks)==24
palette={'cpu':'#e5eef9','gpu':'#e2f1ec','special':'#c5ebdc','transfer':'#b97620','gpu_line':'#35775d','cpu_line':'#3f638e','gray':'#657180'}
f,ax=plt.subplots(figsize=(13,22),layout='constrained');ax.set_xlim(0,15.2);ax.set_ylim(0,27.5);ax.axis('off')
ax.axvspan(.15,5.5,color='#f4f7fb',zorder=0);ax.axvspan(6.0,15.05,color='#f3f8f5',zorder=0)
ax.text(7.6,27.05,'Chimera+：一个 query 的完整路径',fontproperties=cn,fontsize=21,ha='center',weight='bold')
ax.text(7.6,26.48,'R100 ≥ 0.95 · chunks=1 · 32×128 FP32 输入 → Top-100',fontproperties=cn,fontsize=12,ha='center',color='#435262')
ax.text(2.8,25.9,'CPU',fontsize=17,ha='center',weight='bold',color=palette['cpu_line']);ax.text(10.5,25.9,'GPU',fontsize=17,ha='center',weight='bold',color=palette['gpu_line'])
def box(x,y,w,h,title,body,kind='gpu',code=False):
 edge=palette['cpu_line'] if kind=='cpu' else palette['gpu_line'];col=palette.get(kind,palette['gpu']);ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.12,rounding_size=0.13',facecolor=col,edgecolor=edge,lw=1.3,zorder=3));ax.text(x-w/2+.18,y+h/2-.22,title,fontproperties=cn,fontsize=12.5,color=edge,va='top',weight='bold',zorder=4)
 lines=body.split('\n');base=y+h/2-.78;step=.34 if code else .39
 for i,line in enumerate(lines):ax.text(x-w/2+.18,base-i*step,line,fontproperties=cn if any('\u4e00'<=ch<='\u9fff' for ch in line) else (mono if code else cn),fontsize=9.5 if code else 11.5,va='top',zorder=4,color='#20303b')
def arrow(a,b,label='',color=None,labelpos=None):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,lw=1.55,color=color or '#546c60',connectionstyle='arc3,rad=0',zorder=2))
 if label:
  xy=labelpos or((a[0]+b[0])/2,(a[1]+b[1])/2);ax.text(*xy,label,fontproperties=cn,fontsize=9.5,ha='center',va='center',color=color or '#435965',bbox=dict(facecolor='white',edgecolor='none',pad=2),zorder=5)
CX,GX=2.8,10.5;CW,GW=4.7,8.2;CL,CR=CX-CW/2,CX+CW/2;GL,GR=GX-GW/2,GX+GW/2
box(CX,24.7,CW,1.55,'① 查询准备','32条128维查询向量\n旋转 + cb1 / cbex 修正项','cpu')
box(GX,22.6,GW,2.55,'② 建LUT + CAGRA质心路由（4次kernel）','01  precompute_one_bit_lut_kernel\n02  set_value_batch_kernel\n03  search_kernel  [CAGRA]\n04  kern_topk_cta_11',code=True)
arrow((CR,24.65),(GX,23.995),'H2D：查询16KB + cb1',palette['transfer'],(7.45,24.45))
box(GX,19.25,GW,2.65,'③ 候选文档去重、计数（4次kernel）','05  mark_candidate_documents_kernel\n06  count_candidate_bitmap_kernel\n07  DeviceScanInitKernel\n08  DeviceScanKernel  [bitmap前缀和]',code=True)
arrow((GX,21.2),(GX,20.68),'每条查询向量选择256个质心',labelpos=(11.6,20.94))
box(CX,17.5,CW,1.6,'W1 · 取候选数 / 等待','接收4B候选数量\n检查候选缓冲区容量','cpu')
arrow((GL,19.25),(CR,17.7),'D2H：候选数',palette['transfer'],(5.7,18.55))
box(GX,15.75,GW,3.0,'④ GPU粗排 → 保留3000篇（9次kernel）','09  materialize_candidate_documents_kernel\n10  partial_score_kernel\n11  sum_partial_scores_kernel\n12  DeviceRadixSortHistogramKernel\n13  DeviceRadixSortExclusiveSumKernel\n14–17  DeviceRadixSortOnesweepKernel ×4',code=True)
arrow((CR,17.2),(GL,16.2))
box(GX,12.25,GW,2.65,'⑤ 准备3000篇文档的token位置（4次kernel）','18  gather_refined_doc_lengths_kernel\n19  DeviceScanInitKernel\n20  DeviceScanKernel  [文档长度前缀和]\n21  gather_refined_token_positions_kernel',code=True)
arrow((GX,14.12),(GX,13.69),'D2D：Top-3000候选ID',palette['transfer'],(12.3,13.95))
box(CX,10.6,CW,1.6,'W2 · 接收元数据 / 等待','候选ID、token偏移、总数\n用于后续chunk与CPU选择','cpu')
arrow((GL,12.25),(CR,10.85),'D2H：元数据',palette['transfer'],(5.7,11.7))
box(GX,8.6,GW,1.9,'⑥ GPU 1bit文档评分（2次kernel）','22  one_bit_token_score_kernel\n23  one_bit_document_score_kernel',code=True)
arrow((CR,10.35),(GL,8.95))
box(CX,6.7,CW,1.8,'W3 · 选300篇 / 等待','接收3000个1bit分数\nnth_element / sort\n得到精排文档ID','cpu')
arrow((GL,8.6),(CR,7.0),'D2H：1bit分数',palette['transfer'],(5.7,7.85))
box(GX,4.65,GW,2.05,'⑦ Chimera+：GPU完整4bit精排（1次kernel）','24  full_bit_maxsim_kernel\n300 blocks × 512 threads\n每个block对一篇文档计算FP32 MaxSim',kind='special',code=True)
arrow((CR,6.35),(GL,4.98),'H2D：300个ID + cbex',palette['transfer'],(5.75,5.78))
box(CX,2.5,CW,1.8,'W4 · 最终Top-100 / 等待','接收300个完整4bit分数\nCPU heap选择Top-100\n返回文档ID与分数','cpu')
arrow((GL,4.65),(CR,2.8),'D2H：300个分数',palette['transfer'],(5.7,3.9))
box(GX,1.85,GW,1.3,'已加载的UMA数据（不随query搬运整库）','ATS: 4bit codes + full-bit factors\nCUDA buffer: doc_ptrs','gpu',code=True)
arrow((GX,2.62),(GX,3.50),'ATS直接读取',palette['gpu_line'],(11.95,3.05))
ax.text(7.6,.52,'实测：24次GPU kernel + 11次cudaMemcpy + 10次cudaMemset；W1–W4为4个显式等待点。',fontproperties=cn,fontsize=10,ha='center',color='#465666')
for ext in ['png','svg']:f.savefig(A/f'chimera-plus-query-flow.{ext}',dpi=170,bbox_inches='tight')
plt.close(f)
# Actual CUDA timeline: kernel numbers map exactly to the flow diagram.
f,ax=plt.subplots(figsize=(13,4.8),layout='constrained');ms=lambda t:(t-qstart)/1e6
colors={'Routing':'#315b8a','Candidate':'#8b7ab0','1bit scoring':'#dfac54','4bit scoring':'#378579'}
def stage(i):return 'Routing' if i<=4 else 'Candidate' if i<=21 else '1bit scoring' if i<=23 else '4bit scoring'
for i,(name,start,end,g,b,corr) in enumerate(ks,1):
 ax.barh(3.35,ms(end)-ms(start),left=ms(start),height=.55,color=colors[stage(i)],edgecolor='white',linewidth=.4)
 if end-start>80000:ax.text((ms(start)+ms(end))/2,3.35,f'#{i}',ha='center',va='center',fontsize=8,color='white' if stage(i)!='1bit scoring' else '#202020')
for st,en,size,kind in c.execute('select start,end,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=? order by start',(qstart,qend)):
 y=1.65 if kind==1 else .8 if kind==2 else 2.5;ax.barh(y,ms(en)-ms(st),left=ms(st),height=.35,color='#c27630')
for st,en in c.execute('select start,end from CUPTI_ACTIVITY_KIND_MEMSET where start>=? and end<=?',(qstart,qend)):
 ax.barh(2.5,ms(en)-ms(st),left=ms(st),height=.35,color='#aab6c0')
waits=c.execute("select r.start,r.end,s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where r.start>=? and r.end<=? and (s.value like 'cudaStreamSynchronize%' or s.value like 'cudaEventSynchronize%') order by r.start",(qstart,qend)).fetchall();assert len(waits)==4
for i,(st,en,name) in enumerate(waits,1):
 ax.barh(-.05,ms(en)-ms(st),left=ms(st),height=.35,color='#a95555',alpha=.85);ax.text((ms(st)+ms(en))/2,.18,f'W{i}',ha='center',va='bottom',fontsize=8)
ax.set_yticks([3.35,2.5,1.65,.8,-.05],['GPU compute kernels','GPU memset / D2D','H2D transfers','D2H transfers','CPU explicit sync APIs']);ax.set_ylim(-.55,4.05);ax.set_xlim(0,(qend-qstart)/1e6);ax.set_xlabel('Time since query start (ms)');ax.set_title('One real query: kernel order, transfers and explicit CPU waits');ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);f.legend(handles=[Patch(facecolor=v,label=k)for k,v in colors.items()],loc='outside upper center',ncol=4,frameon=False,fontsize=9);f.supxlabel('Kernel numbers match the flowchart. Explicit synchronization can overlap GPU work; API spans are not added to kernel time.',fontsize=8)
for ext in ['png','svg']:f.savefig(A/f'chimera-plus-query-timeline.{ext}',dpi=180,bbox_inches='tight')
plt.close(f)
# Standalone requested research document, no performance re-runs.
text='''# Chimera+：一个query会触发什么？

**主干：CPU准备查询 → GPU路由/候选粗排 → GPU 1bit评分 → CPU选300篇 → GPU 4bit精排 → CPU返回Top-100。**

以当前LoTTE、R100≥0.95配置的一条真实查询为例：`nprobe256 / refine3000 / full300 / itopk288 / chunks1`，CPU预算20，B1。

## 完整流程图

蓝色为CPU步骤，绿色为GPU，橙色箭头为数据传输；**W1–W4表示CPU必须确认GPU结果可用后才能继续。** 图中编号对应实际kernel调用顺序。

![Chimera+单query流程](../report-assets/chimera-plus-query-flow.png)

[放大矢量流程图](../report-assets/chimera-plus-query-flow.svg)

## 候选规模怎么收缩？

| 步骤 | 输入 → 输出 |
|---|---|
| CPU查询准备 | 32×128 FP32 → 旋转后的查询与修正项 |
| CAGRA路由 | 每条查询向量 → 256个聚类中心 |
| 候选生成 | 命中的倒排文档 → 去重候选集合，数量随query变化 |
| GPU候选粗排 | 全部候选 → 3000篇文档 |
| GPU 1bit评分 | 3000篇 → 3000个近似MaxSim分数 |
| CPU选择 | 按1bit分数 → 300篇精排文档 |
| GPU完整4bit评分 | 300篇 → 300个完整4bit MaxSim分数 |
| CPU最终排序 | 300个分数 → Top-100文档 |

**Chimera+只替换了最后的完整4bit评分。** 候选选择与最终Top-k仍在CPU，查询也仍在CPU旋转；当前还不是整个流程都留在GPU。

## 实际执行时间轴

下面是同一条查询的实测轨迹，横轴为毫秒。GPU行上的编号对应流程图；短kernel很多，因此只在较宽的条上标号。

![实际query时间轴](../report-assets/chimera-plus-query-timeline.png)

**当前GPU计算stream按序执行，查询之间无流水线。** H2D、D2H和部分清零可以在独立stream中异步执行。W1–W4共对应3次`cudaStreamSynchronize`和1次`cudaEventSynchronize`；部分传输API本身也可能阻塞CPU。

## 24次kernel，一次看全

按短名称计，这条查询包含19类kernel、共24次调用。两组前缀和各启动初始化与扫描kernel；CUB排序的Onesweep实际调用4次。次数来自这条轨迹，内部库调用次数会随规模与版本变化。

| 编号 | Kernel | 作用 | GPU耗时µs |
|---:|---|---|---:|
'''
roles=['建立1bit查找表LUT','CAGRA搜索工作区置值','CAGRA搜索质心图','合并质心Top-k','标记候选文档位图/去重','统计候选位图数量','候选前缀和初始化','候选前缀和扫描','生成候选文档ID数组','计算候选部分分数','合并部分分数','基数排序直方图','基数排序前缀和','基数排序pass1','基数排序pass2','基数排序pass3','基数排序pass4','收集3000篇文档长度','文档长度前缀和初始化','文档长度前缀和扫描','收集候选文档token位置','计算1bit token分数','聚合1bit文档MaxSim','完整4bit文档MaxSim（Chimera+新增）']
for i,((name,st,en,g,b,corr),role) in enumerate(zip(ks,roles),1):text+=f'| {i} | `{name}` | {role} | {(en-st)/1000:.3f} |\n'
text+='''
## 几个容易混淆的地方

- **1bit和4bit是两次评分。** 先便宜地给3000篇打分，再用完整4bit编码给选中的300篇精排；精排算术仍为FP32。
- **4bit编码无需逐query整库上传。** GPU通过ATS直接读UMA中的现有codes和factor；每query仍有小型ID、修正项与分数传输。
- **cudaMemcpy、cudaMemset不是上表中的独立命名kernel。** 这条query另有11次拷贝、10次清零，已在流程与时间轴中区分。
- **kernel耗时不能直接相加当端到端延迟。** 传输与计算可能重叠，CPU准备、选择和等待也在搜索调用内。

## 对应实现与证据

- [查询调度与候选/1bit评分源码](/home/lxq/graduate/new-idea/multi-vector/Chimera/cpp/src/search_impl.cu)
- [GPU完整4bit精排源码](/home/lxq/graduate/new-idea/multi-vector/Chimera/cpp/src/gpu_fullbit_scoring.cu)
- [实测全流程汇总与24次kernel记录](../chimera-plus/profile-r100-p95/flow-summary.json)
- [两系统及Chimera+的性能报告](lotte-r10-brief.md)

源码分支`Chimera+`，提交`c97f24c`；图来自已采集的NSYS轨迹，本次没有新增GPU测试。
'''
(D/'chimera-plus.md').write_text(text)
main=D/'lotte-r10-brief.md';s=main.read_text();link='[Chimera+单query流程图与kernel明细](chimera-plus.md)'
if link not in s:s=s.replace('## Chimera+：只替换完整4bit精排','## Chimera+：只替换完整4bit精排\n\n'+link,1);main.write_text(s)
(A/'chimera-plus-query-kernels.json').write_text(json.dumps({'query_NVTX_start_ns':qstart,'query_NVTX_end_ns':qend,'kernels':[{'sequence':i,'name':r[0],'GPU_us':(r[2]-r[1])/1000,'role':roles[i-1]}for i,r in enumerate(ks,1)]},ensure_ascii=False,indent=2)+'\n')
print('Created chimera-plus.md, flow diagram, timeline and vector sources; no GPU run.')
