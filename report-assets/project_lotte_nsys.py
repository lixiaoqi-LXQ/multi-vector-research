"""CPU-only projection of existing three NSYS SQLite traces; no new measurement."""
from pathlib import Path
import hashlib,json,sqlite3

ROOT=Path('/home/lxq/graduate/ai-workspace/research/multivector')
BASE=ROOT/'chimera/profile-v2/firstdelivery-R10-profile-plans'
SQL={
 'nvtx':'SELECT COALESCE(n.text,s.value),n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId WHERE n.end IS NOT NULL',
 'runtime':'SELECT s.value,COUNT(*),SUM(r.end-r.start) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId GROUP BY s.value',
 'copy_bytes':'SELECT SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY',
}
def merged(a):
 out=[]
 for l,h in sorted(a):
  if out and l<=out[-1][1]:out[-1]=(out[-1][0],max(h,out[-1][1]))
  else:out.append((l,h))
 return out
def total(a):return sum(h-l for l,h in merged(a))
def intersect(a,b):return sum(max(0,min(h,k)-max(l,j)) for l,h in merged(a) for j,k in merged(b))
records=[]
for name,outer in [('chimera-R10p90','Chimera/query_local_row'),('plaid-native-R10p90','PLAID.rank'),('plaid-gpuall-R10p90','PLAID.rank')]:
 p=BASE/name/'nsys/trace.sqlite';c=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True)
 nv={}
 for label,l,h in c.execute(SQL['nvtx']):nv.setdefault(label,[]).append((l,h))
 scope=nv[outer];assert len(scope)==32
 gpu=[];tables={}
 for tab in ['CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']:
  a=c.execute('SELECT start,end FROM '+tab).fetchall();gpu+=a;tables[tab]={'count_capture':len(a),'capture_union_ns':total(a),'scope_intersection_ns':intersect(scope,a)}
 api=[{'name':label,'count_capture':n,'duration_capture_ns':ns} for label,n,ns in c.execute(SQL['runtime'])]
 record={'label':name,'source_sqlite':str(p),'source_sqlite_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'outer_NVTX':outer,'outer_count':32,'outer_union_ns':total(scope),'GPU_activity_in_outer_ns':intersect(scope,gpu),'outer_without_GPU_activity_ns':total(scope)-intersect(scope,gpu),'GPU_activity_tables':tables,'logical_memcpy_bytes_capture':c.execute(SQL['copy_bytes']).fetchone()[0],'NVTX_stages':[{'name':label,'count':len(a),'sum_ns':sum(h-l for l,h in a),'union_ns':total(a)} for label,a in sorted(nv.items())],'CUDA_API_capture':api}
 if 'Chimera/cpu_fullbit' in nv:record.update(CPU_fullbit_union_ns=total(nv['Chimera/cpu_fullbit']),CPU_GPU_overlap_ns=intersect(nv['Chimera/cpu_fullbit'],gpu))
 records.append(record);c.close()
out={'scope':'existing .90 frozen-point 32-query diagnostic traces, CPU projection only','scope_rule':'Interval percentages are within named outer NVTX. Event counts, logical bytes and API durations use entire same 32-query capture, including query conversion outside rank. NVTX spans can nest; CUDA API durations can overlap GPU activity.','sql':SQL,'event_sql':'SELECT start,end FROM <one of three named GPU activity tables>','records':records}
p=ROOT/'report/lotte-r10-nsys-stage-projection.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(p)
