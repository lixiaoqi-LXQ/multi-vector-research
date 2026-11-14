"""Project existing accepted full-query runs; never launches search."""
from pathlib import Path
import json,hashlib,statistics
R=Path('/home/lxq/graduate/ai-workspace/research/multivector')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def extract(rel,label):
 p=R/rel;a=json.loads((p/'accepted-summary.json').read_text());q=json.loads((p/'prepared.json').read_text());g=a['groups'][0];rows=g['rounds']
 assert a['output_contract_passed'] and q['queries']==2931 and q['spec']['k']==10 and q['OMP_NUM_THREADS']==20
 assert q['binary_sha256']=='c54539b5143f63ad96ab0118f165dded7861d9c5af180767cd74c66d7bd4945d'
 assert not q['spec'].get('heldout_rows') and not q['spec'].get('query_rows')
 assert all(x['queries']==2931 and x['returned']==29310 for x in rows)
 return {'label':label,'point':g['point'],'round_count':len(rows),'queries_per_round':2931,'rounds':rows,'mean_recall10':statistics.mean(x['recall10'] for x in rows),'mean_QPS':statistics.mean(x['qps'] for x in rows),'QPS_range':[min(x['qps'] for x in rows),max(x['qps'] for x in rows)],'mean_latency_ms':statistics.mean(x['mean_ms'] for x in rows),'mean_round_p99_ms':statistics.mean(x['p99_ms'] for x in rows),'p99_range_ms':[min(x['p99_ms'] for x in rows),max(x['p99_ms'] for x in rows)],'source_run':str(p),'source_sha256':{n:sha(p/n) for n in ['prepared.json','summary.json','accepted-summary.json','release.json']}}
base='chimera/runs/'
sel=[extract(base+'lotte-r10-95-neighbors-v1/r10-0.95-qualify-ref1000-full100-chunk1-full1','A:64/1000/100, screening'),extract(base+'lotte-r10-95-neighbors-BC-v1/r10-0.95-qualify-np128-ref1000-full100-chunk1-full1','B:128/1000/100, screening'),extract(base+'lotte-r10-95-neighbors-BC-v1/r10-0.95-qualify-ref3000-full100-chunk1-full1','C:64/3000/100, screening'),extract(base+'lotte-r10-95-neighbor-E-v1/r10-0.95-qualify-np128-ref1000-full50-chunk1-full1','E:128/1000/50, screening')]
confirm=[extract(base+'lotte-r10-95-BE-full3-confirm-v1/r10-0.95-selected-chunk1-full3','B full3'),extract(base+'lotte-r10-95-BE-full3-confirm-v1/r10-0.95-E-full50-chunk1-full3','E full3')]
chunks=[dict(confirm[1],label='chunks=1'),extract(base+'lotte-r10-95-winner-chunks45-full3-v1/r10-0.95-winner-chunk4-full3','chunks=4'),extract(base+'lotte-r10-95-winner-chunks45-full3-v1/r10-0.95-winner-chunk5-full3','chunks=5 (upstream default)')]
for x in sel+confirm+chunks:x['target_met']=x['mean_recall10']>=.95
assert all(x['round_count']==3 for x in confirm+chunks)
assert max((x for x in confirm if x['target_met']),key=lambda x:x['mean_QPS'])['point']['k_full_bit']==50
assert all({k:v for k,v in x['point'].items() if k!='num_chunks'}=={k:v for k,v in chunks[0]['point'].items() if k!='num_chunks'} for x in chunks)
best=max((x for x in chunks if x['target_met']),key=lambda x:x['mean_QPS'])
out={'scope':'LoTTE full4bit Recall@10>=.95; CPU20; full2931 queries; no heldout','quality_rule':'mean Recall@10 of three full-query rounds','QPS_rule':'arithmetic mean of three per-round QPS (N/sum synchronous search latencies)','selection_screening':sel,'selection_confirmation':confirm,'chunk_results':chunks,'selected_parameters':best['point'],'recommended_chunks':best['point']['num_chunks'],'upstream_default_chunks':5,'algorithm_binary_unchanged':True,'device_lock':'/home/lxq/device.lock','profile_timing_not_formal_QPS':True}
(R/'report/lotte-r10-95-chunks-data.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'recommended_chunks':out['recommended_chunks'],'formal':[(x['point']['num_chunks'],x['mean_recall10'],x['mean_QPS']) for x in chunks]}))
