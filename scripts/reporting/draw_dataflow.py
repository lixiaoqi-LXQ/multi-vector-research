"""Static source-supported CPU/GPU search dataflows; no timing proportions."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib import rcParams
rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','savefig.facecolor':'white'})
OUT=Path(__file__).resolve().parent
CPU='#e9f1fa';GPU='#e8f4ee';INK='#203447';EDGE='#4b6379';ALT='#886539'
fig,axs=plt.subplots(1,2,figsize=(14,8.5));fig.subplots_adjust(left=.035,right=.98,top=.90,bottom=.10,wspace=.08)
def box(ax,x,y,text,w=2.37,h=.64,kind='cpu',dashed=False,fontsize=10):
 p=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.07,rounding_size=.10',facecolor=CPU if kind=='cpu' else GPU,edgecolor=ALT if dashed else EDGE,linewidth=1.2,linestyle=(0,(4,3)) if dashed else 'solid',zorder=3)
 ax.add_patch(p);ax.text(x,y,text,ha='center',va='center',fontsize=fontsize,color=INK,zorder=4)
 return(x,y)
def arrow(ax,a,b,label=None,dashed=False,rad=0):
 p=FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color=ALT if dashed else EDGE,linewidth=1.25,linestyle=(0,(4,3)) if dashed else 'solid',connectionstyle=f'arc3,rad={rad}',shrinkA=4,shrinkB=4,zorder=2);ax.add_patch(p)
 if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.11,label,ha='center',va='bottom',fontsize=8.5,color=EDGE,bbox=dict(facecolor='white',edgecolor='none',pad=1),zorder=5)
for ax,title in zip(axs,['(a) Chimera','(b) PLAID']):
 ax.set_xlim(0,6.4);ax.set_ylim(-.6,7.6);ax.axis('off')
 ax.text(3.2,7.35,title,ha='center',fontsize=16,fontweight='bold',color=INK)
 for x,name,color in [(1.55,'CPU',CPU),(4.85,'GPU',GPU)]:
  ax.add_patch(FancyBboxPatch((x-1.43,.05),2.86,6.8,boxstyle='round,pad=0.02,rounding_size=.12',facecolor=color,alpha=.30,edgecolor='#d2dbe2',linewidth=.9,zorder=0))
  ax.text(x,6.98,name,ha='center',fontsize=12,fontweight='bold',color=INK)
 ax.add_patch(FancyBboxPatch((.15,-.48),6.05,.4,boxstyle='round,pad=.04',facecolor='#f2f0e8',edgecolor='#c6bd9e',linewidth=1))
 ax.text(3.18,-.28,'One physical LPDDR5X UMA pool · CPU / GPU allocations',ha='center',va='center',fontsize=9.5,color=INK)
a=axs[0]
box(a,1.55,6.35,'Query → FHT rotation')
box(a,4.85,5.30,'Centroid route\nCAGRA on rotated query',kind='gpu')
box(a,4.85,4.20,'1-bit coarse scoring\nDocument candidate selection',kind='gpu')
box(a,4.85,2.95,'1-bit document scoring\nCandidate chunks',kind='gpu')
box(a,1.55,2.0,'Full-bit MaxSim\nCPU native scoring')
box(a,1.55,.87,'CPU Top-K → host results')
arrow(a,(2.78,6.30),(3.62,5.40),'rotated query')
arrow(a,(4.85,4.94),(4.85,4.57))
arrow(a,(4.85,3.84),(4.85,3.32))
arrow(a,(3.62,2.90),(2.78,2.08),'scores / candidate metadata')
arrow(a,(1.55,1.62),(1.55,1.24))
a.add_patch(FancyBboxPatch((.23,1.49),5.98,1.98,boxstyle='round,pad=.04',facecolor='none',edgecolor='#748c7c',linestyle=(0,(3,3)),linewidth=1,zorder=1))
a.text(3.20,3.57,'Chunk pipeline: GPU chunk i+1 ∥ CPU chunk i',ha='center',fontsize=9,color='#426450')
a.text(4.85,1.17,'CPU reads full-bit codes;\nGPU reads clustered 1-bit codes',ha='center',va='center',fontsize=9,color=EDGE)
a.text(3.2,-.78,'Arrows: search dependencies and logical handoffs',ha='center',fontsize=9,color=EDGE)
b=axs[1]
box(b,1.55,6.35,'Pre-encoded query (host)')
box(b,4.85,5.30,'Centroid scores\nCandidate pruning / scoring',kind='gpu')
box(b,1.55,3.92,'IDs / lengths / offsets\nIVF + native payload gather',h=.80)
box(b,4.85,2.35,'Residual decode → FP16 MaxSim\nGPU score sort',h=.8,kind='gpu',fontsize=9.5)
box(b,1.55,.87,'Host result IDs + scores')
arrow(b,(2.78,6.30),(3.62,5.42),'query tensor')
arrow(b,(3.62,5.10),(2.78,4.13),'candidate IDs')
arrow(b,(2.78,4.29),(3.62,5.34),'IVF / code lookup',rad=-.28)
arrow(b,(2.78,3.71),(3.62,2.55),'native: gathered payload .cuda()')
arrow(b,(3.62,2.12),(2.78,1.02),'results .tolist()')
box(b,4.85,3.92,'Placement controls\nGPU code / all-bulk payload',h=.80,kind='gpu',dashed=True,fontsize=9.3)
arrow(b,(2.78,3.97),(3.62,3.97),dashed=True)
arrow(b,(4.85,3.47),(4.85,2.80),dashed=True)
b.text(3.2,1.50,'Dashed: payload placement changes; CPU metadata remains',ha='center',fontsize=8.8,color=ALT)
b.text(4.85,.73,'code: codes + residuals\nall-bulk: also IVF payload',ha='center',va='center',fontsize=8.7,color=ALT)
b.text(3.2,-.78,'Native lookup paths can revisit CPU metadata / gathers',ha='center',fontsize=9,color=EDGE)
fig.text(.5,.957,'Source-level search dataflow on DGX Spark',ha='center',fontsize=18,color=INK)
fig.text(.5,.925,'CPU and GPU share physical LPDDR5X memory · Arrows show logical search dependencies',ha='center',fontsize=10,color=EDGE)
for ext in ['svg','png']:fig.savefig(OUT/f'system-dataflow.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
