from pathlib import Path
import csv,json,sqlite3,numpy as np,statistics,sys,hashlib
P=Path(__file__).resolve().parent;R=P.parent.parent;sys.path.insert(0,str(R/'chimera'));from run_search_grid_device_lock import check
c=sqlite3.connect('file:'+str(P/'nsys/trace.sqlite')+'?mode=ro',uri=True)
a,b=c.execute("select start,end from NVTX_EVENTS where text='Chimera/query_local_row' order by start limit 1").fetchone()
ks=c.execute('select s.value,k.start,k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.start>=? and k.end<=? order by k.start',(a,b)).fetchall();assert len(ks)==24
nsysrows=list(csv.DictReader((P/'nsys/rows.tsv').open(),delimiter='\t'));ref=[r for r in nsysrows if r['query_id']=='0'];e2e_s=float(ref[0]['latency_ms'])*1e-3
ref_ids=[int(x['document_id']) for x in ref];ref_scores=np.array([float(x['quantized_score']) for x in ref],dtype=np.float32).view('u4')
def read(phase):
 rows=list(csv.DictReader((P/phase/'raw.csv').open()));units=rows[0];rr=[x for x in rows if x['ID']];assert len(rr)==24
 for (name,_,__),r in zip(ks,rr):assert name in r['Kernel Name'],(name,r['Kernel Name'])
 vals=list(csv.DictReader((P/phase/'rows.tsv').open(),delimiter='\t'));v=[r for r in vals if r['query_id']=='0'];assert [int(x['document_id']) for x in v]==ref_ids;assert np.array_equal(np.array([float(x['quantized_score']) for x in v],dtype=np.float32).view('u4'),ref_scores)
 check(P/phase/'rows.tsv',R/'chimera-plus/qa100-ids.npy',32,2428854,100)
 assert units['gpu__time_duration.sum']=='ms' and units['lts__t_bytes.sum']=='Mbyte'
 return rr
math=read('ncuflow');mem=read('memflow')
def category(n):
 if n in ['search_kernel','set_value_batch_kernel','kern_topk_cta_11']:return 'GPU route'
 if n=='full_bit_maxsim_kernel':return 'GPU 4bit scoring'
 if n in ['one_bit_token_score_kernel','one_bit_document_score_kernel']:return 'GPU 1bit scoring'
 return 'GPU candidate/LUT'
rows=[];groups={}
for idx,((name,start,end),r,m) in enumerate(zip(ks,math,mem)):
 ff=float(r['smsp__sass_thread_inst_executed_op_ffma_pred_on.sum']);fa=float(r['smsp__sass_thread_inst_executed_op_fadd_pred_on.sum']);fm=float(r['smsp__sass_thread_inst_executed_op_fmul_pred_on.sum']);flops=2*ff+fa+fm;byt=float(m['lts__t_bytes.sum'])*1e6;miss=float(m['lts__t_sectors_op_read_lookup_miss.sum']);sysmiss=float(m['lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum'])
 row={'kernel_id':idx,'kernel':name,'stage':category(name),'FP32_FLOPs':flops,'L2_requested_bytes':byt,'L2_read_miss_sectors':miss,'system_memory_read_miss_sectors':sysmiss,'estimated_lower_memory_read_request_bytes':32*miss,'NSYS_GPU_duration_us':(end-start)/1e3,'FFMA':ff,'FADD':fa,'FMUL':fm};rows.append(row)
 g=groups.setdefault(category(name),{'FP32_FLOPs':0,'L2_requested_bytes':0,'estimated_lower_memory_read_request_bytes':0,'NSYS_GPU_duration_us':0})
 for key in g:g[key]+=row[key]
totalflops=sum(r['FP32_FLOPs'] for r in rows);totalbytes=sum(r['L2_requested_bytes']for r in rows);missbytes=sum(r['estimated_lower_memory_read_request_bytes']for r in rows)
def union(intervals):
 iv=sorted(intervals);end=-1;total=0
 for l,r in iv:
  if l>end:total+=r-l
  elif r>end:total+=r-end
  end=max(end,r)
 return total
intervals=[(l,r)for _,l,r in ks];counts={};copies=[]
for table in ['CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']:
 iv=c.execute('select start,end from '+table+' where start>=? and end<=?',(a,b)).fetchall();counts[table]={'count':len(iv),'sum_GPU_time_us':sum(y-x for x,y in iv)/1e3};intervals+=iv
 if table.endswith('MEMCPY'):copies=c.execute('select copyKind,sum(bytes),count(*) from '+table+' where start>=? and end<=? group by copyKind',(a,b)).fetchall()
GPU=union(intervals)
result={'scope':'Chimera+ R100 p95, full first query across all24 GPU kernels; original preencoded-host-input to hostTop100 search completion boundary','parameters':{'nprobe':256,'k_refine':3000,'k_full_bit':300,'itopk':288,'chunks':1,'CPU_budget':20,'batch':1},'normal_timing_source':'existing32-query NSYS, first query search timer (not NCU replay time)','count_source':'NCU whole-query first24 kernels, arithmetic and memory short counter captures; normal/NCU/memory first-query Top100IDs and FP32score bits equal','normal_end_to_end_ms':e2e_s*1e3,'GPU_kernel_count':24,'GPU_kernel_sequence_matches_NSYS':True,'all_kernels_FP32_FLOPs':totalflops,'all_kernels_L2_requested_bytes':totalbytes,'L2_read_miss_bytes_estimate':missbytes,'system_memory_read_miss_sectors':sum(r['system_memory_read_miss_sectors'] for r in rows),'estimated_end_to_end_FP32_TFLOPs':totalflops/e2e_s/1e12,'estimated_end_to_end_L2_request_GB_s':totalbytes/e2e_s/1e9,'estimated_end_to_end_lower_memory_read_request_GB_s':missbytes/e2e_s/1e9,'DRAM_total_read_write_bandwidth_measured':False,'memory_proxy_note':'32B*L2 read-miss sectors counts lower-level/system-memory read requests; not physical LPDDR read+write traffic, excludes writebacks and may be served by coherent caches. No DRAM utilization percentage inferred.','arithmetic_scope':'FP32 counts only GPU; CPU work and integer/bit/shuffle/max instructions not FP32 FLOPs, but CPU/transfer/sync time remains in end-to-end denominator','normal_firstquery_GPU_activity_union_ms':GPU/1e6,'normal_firstquery_outside_GPU_union_ms':(b-a-GPU)/1e6,'normal_firstquery_gpu_copies':counts,'memcpy_logical_bytes_by_kind':[{'copy_kind':k,'bytes':byt,'calls':num}for k,byt,num in copies],'stages':groups,'kernels':rows,'original_32query_activity_sampling':json.loads((P/'summary.json').read_text())['GPU_activity_sampling'],'formal_QPS_not_changed':True,'source_hashes':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P/'ncuflow/raw.csv',P/'memflow/raw.csv',P/'nsys/rows.tsv',P/'nsys/trace.sqlite']}}
(P/'flow-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['normal_end_to_end_ms','all_kernels_FP32_FLOPs','all_kernels_L2_requested_bytes','L2_read_miss_bytes_estimate','estimated_end_to_end_FP32_TFLOPs','estimated_end_to_end_L2_request_GB_s','estimated_end_to_end_lower_memory_read_request_GB_s','normal_firstquery_GPU_activity_union_ms','normal_firstquery_outside_GPU_union_ms']},indent=2));print(groups)
