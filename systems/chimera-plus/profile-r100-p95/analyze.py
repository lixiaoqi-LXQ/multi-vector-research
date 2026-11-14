from pathlib import Path
import sqlite3,json,csv,statistics,hashlib,sys,fcntl
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent.parent;sys.path.insert(0,str(R/'chimera'));from run_search_grid_device_lock import check
c=sqlite3.connect('file:'+str(P/'nsys/trace.sqlite')+'?mode=ro',uri=True)
qs=c.execute("select start,end from NVTX_EVENTS where text='Chimera/query_local_row' order by start").fetchall();assert len(qs)==32 and all(b>a for a,b in qs)
k=c.execute("select k.start,k.end,k.gridX,k.blockX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where s.value='full_bit_maxsim_kernel' order by k.start").fetchall();assert len(k)==32 and all((g,b)==(300,512) for _,__,g,b in k)
metrics={}
for mid,name in c.execute('select metricId,metricName from TARGET_INFO_GPU_METRICS'):
 if mid not in [6,7,8,9]:continue
 vals=[v for t,v in c.execute('select timestamp,value from GPU_METRICS where metricId=?',(mid,)) if any(a<=t<b for a,b in qs)]
 assert len(vals)>100 and all(0<=v<=100 for v in vals)
 metrics[name]={'mean_percent':statistics.mean(vals),'sample_count':len(vals),'min_percent':min(vals),'max_percent':max(vals)}
raw=list(csv.DictReader((P/'ncu/raw.csv').open()));units=raw[0];assert units['gpu__time_duration.sum']=='us' and units['lts__t_bytes.sum']=='Mbyte';rr=[x for x in raw if x['ID']];assert len(rr)==3
rows=[]
for x in rr:
 get=lambda k:float(x[k]);ff=get('smsp__sass_thread_inst_executed_op_ffma_pred_on.sum');fa=get('smsp__sass_thread_inst_executed_op_fadd_pred_on.sum');fm=get('smsp__sass_thread_inst_executed_op_fmul_pred_on.sum');sec=get('gpu__time_duration.sum')*1e-6;b=get('lts__t_bytes.sum')*1e6;t=ff/4096
 assert int(t)==t and 300*3<=t<=300*180
 rows.append({'launch':int(x['ID']),'kernel_duration_us':sec*1e6,'FFMA':ff,'FADD':fa,'FMUL':fm,'FP32_FLOPs':2*ff+fa+fm,'FP32_TFLOPs_per_s':(2*ff+fa+fm)/sec/1e12,'L2_requested_bytes':b,'L2_request_GB_per_s':b/sec/1e9,'tokens_inferred_from_FFMA':int(t),'minimum_unique_codes_factors_bytes':68*t,'payload_effective_GB_per_s_estimate':68*t/sec/1e9,'SM_throughput_pct':get('sm__throughput.avg.pct_of_peak_sustained_elapsed')})
total_s=sum(x['kernel_duration_us'] for x in rows)*1e-6
aggregate={'FP32_TFLOPs_per_s':sum(x['FP32_FLOPs'] for x in rows)/total_s/1e12,'FP32_TFLOPs_range':[min(x['FP32_TFLOPs_per_s'] for x in rows),max(x['FP32_TFLOPs_per_s'] for x in rows)],'L2_request_GB_per_s':sum(x['L2_requested_bytes'] for x in rows)/total_s/1e9,'L2_request_GB_per_s_range':[min(x['L2_request_GB_per_s'] for x in rows),max(x['L2_request_GB_per_s'] for x in rows)],'payload_effective_GB_per_s_estimate':sum(x['minimum_unique_codes_factors_bytes'] for x in rows)/total_s/1e9,'payload_effective_GB_per_s_range':[min(x['payload_effective_GB_per_s_estimate'] for x in rows),max(x['payload_effective_GB_per_s_estimate'] for x in rows)]}
validations={}
for phase in ['nsys','ncu']:
 a=json.loads((P/phase/'capture-completion.json').read_text());assert a['returncode']==0 and a['all_descendants_gone'] and a['lease_released'];v=check(P/phase/'rows.tsv',R/'chimera-plus/qa100-ids.npy',32,2428854,100);assert v['returned']==3200;validations[phase]={'returned':v['returned'],'recall100_sample_only':v['recall100'],'output_sha256':v['output_sha256']}
 with open(P/phase/'tree-supervision.json') as f:tree=json.load(f)
 assert tree['all_owned_descendants_gone'] and not tree['timed_out']
# Report only physically valid quantities; raw clock/warp fields are not used.
result={'scope':'Chimera+ R100 p95 np256/ref3000/full300/itopk288/chunks1,CPU20;32-query one-shot diagnostic;NCU3 launches','queries':32,'GPU_fullbit_kernel':{'count':32,'mean_us':statistics.mean(b-a for a,b,_,__ in k)/1000,'range_us':[min(b-a for a,b,_,__ in k)/1000,max(b-a for a,b,_,__ in k)/1000]},'query_NVTX_mean_ms':sum(b-a for a,b in qs)/32e6,'GPU_activity_sampling':metrics,'NCU_launches':rows,'NCU_aggregate_time_weighted':aggregate,'DRAM_bandwidth_available':False,'DRAM_absence_evidence':{'NCU_actual_device':'GB20B','dram_base_metrics':[l for l in (P/'metrics-actual.txt').read_text().splitlines() if l.startswith('dram__')],'NSYS_metrics':[x[0] for x in c.execute('select metricName from TARGET_INFO_GPU_METRICS') if any(y in x[0].lower() for y in ['dram','bandwidth','lpddr'])]},'theoretical_LPDDR_bandwidth_GB_s':273,'hardware_source':'https://docs.nvidia.com/dgx/dgx-spark/hardware.html','method':'NCU counted FP32 = 2*FFMA+FADD+FMUL; bandwidth=L2 bytes/kernel seconds. 68*T/kernel seconds is an estimated useful-payload rate, not physical DRAM traffic. T inferred from FFMA/(32*128); no whole-corpus mean doc length used.','formal_performance_unchanged':True,'validations':validations,'trace_warnings':[r[0] for r in c.execute('select text from DIAGNOSTIC_EVENT where severity>=2')],'unused_metadata':'clock-rate and warp-normalization values not used; report uses bounded activity metrics and per-kernel instruction/byte counters.'}
result['evidence_sha256']={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'ncu/raw.csv',P/'ncu/counters.ncu-rep',P/'nsys/trace.sqlite',P/'nsys/trace.nsys-rep']};(P/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(aggregate,indent=2))
