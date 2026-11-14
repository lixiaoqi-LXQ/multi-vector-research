import sys,json,math,hashlib,subprocess
from pathlib import Path
import numpy as np
r=Path(__file__).parent;data=r/'data';data.mkdir(exist_ok=True)
np.save(data/'queries_ids.npy',np.array(['qA','qB','qC']))
np.save(data/'doc_ids.npy',np.array(['dA','dB','dC','dD']))
qrels=r/'qrels.tsv';qrels.write_text('qA\tQ0\tdA\t3\nqA\tQ0\tdB\t1\nqB\tQ0\tdC\t1\n')
ranks=[[1,0,3,2],[3,2,0,1],[0,1,2,3]]
rows=[]
for round_ in range(3):
 for row,ids in enumerate(ranks):rows.append(dict(round=round_,query_row=row,query_id=['qA','qB','qC'][row],ids=ids,scores=[4,3,2,1],elapsed_ms=1))
plaid=r/'plaid.jsonl';plaid.write_text(''.join(json.dumps(x)+'\n' for x in rows))
tsvs=[]
for round_ in range(3):
 p=r/f'chimera-round{round_}.tsv';tsvs.append(p)
 p.write_text('query_id\trank\tdocument_id\tquantized_score\tlatency_ms\n'+''.join(f'{q}\t{rank}\t{doc}\t{5-rank}\t1\n' for q,docs in enumerate(ranks) for rank,doc in enumerate(docs,1)))
tool=Path('/home/lxq/graduate/ai-workspace/research/multivector/scripts/qrels_metrics.py')
base=[sys.executable,str(tool),'--data',str(data),'--qrels',str(qrels)]
qa=(1+7/math.log2(3))/(7+1/math.log2(3));expected={'MRR@10':.5,'NDCG@10':(qa+1/math.log2(3))/3,'Success@5':2/3};checks=[]
def run(name,files,extra=(),success=True,match=None):
 out=r/(name+'.json');p=subprocess.run(base+list(extra)+['--output',str(out)]+[str(f) for f in files],capture_output=True,text=True)
 assert (p.returncode==0)==success,(name,p.stderr)
 if not success:
  assert match in p.stderr,p.stderr
  return
 return json.loads(out.read_text())
a=run('plaid-metrics',[plaid],['--expected-rounds','3']);b=run('chimera-metrics',tsvs,['--expected-rounds','3'])
for result in [a,b]:
 assert len(result['rounds'])==3
 for key,want in expected.items():assert abs(result['mean_round_metrics'][key]-want)<1e-12
 assert result['label_statistics']['selected_queries_without_positive_judgment']==1
 assert result['identity']['qrels']['sha256']==hashlib.sha256(qrels.read_bytes()).hexdigest()
 assert result['all_result_queries_complete']
checks.append('PLAID3round and Chimera3TSV round mapping match hand-computed graded metrics with all3query denominator')
partial=r/'partial.jsonl';partial.write_text(json.dumps(rows[0])+'\n')
run('partial-rejected',[partial],success=False,match='Incomplete query coverage')
p=run('partial-counted-zero',[partial],['--allow-missing-results']);assert p['rounds'][0]['missing_queries']==2 and abs(p['mean_round_metrics']['MRR@10']-1/3)<1e-12
checks.append('missing result queries rejected by default; explicit partial diagnostic counts missing2queries as0 in full3query denominator')
run('round-count-rejected',[plaid],['--expected-rounds','2'],False,'round count')
bad=rows[0]|{'query_id':'qB'};wrong=r/'wrong-qid.jsonl';wrong.write_text(json.dumps(bad)+'\n');run('qid-rejected',[wrong],success=False,match='qid disagrees')
bad=rows[0]|{'ids':[1,1,3,2]};wrong=r/'wrong-docids.jsonl';wrong.write_text(json.dumps(bad)+'\n');run('duplicate-docs-rejected',[wrong],success=False,match='Duplicate or invalid document')
checks.append('wrong round count, external-qid mapping and duplicate returned doc IDs rejected')
subset=r/'subset-rows.npy';np.save(subset,np.array([2,0],dtype=np.int64));tsv=r/'subset.tsv';tsv.write_text('query_id\trank\tdocument_id\n'+''.join(f'{local}\t{rank}\t{doc}\n' for local,global_ in enumerate([2,0]) for rank,doc in enumerate(ranks[global_],1)))
s=run('subset-metrics',[tsv],['--query-rows',str(subset)]);assert s['rounds'][0]['expected_queries']==2 and s['mean_round_metrics']['MRR@10']==.5 and abs(s['mean_round_metrics']['NDCG@10']-qa/2)<1e-12
checks.append('Chimera local rows correctly map through selected original rows[2,0] to external qids')
report={'CPU_only':True,'hand_example':{'qA':'grades3for dA and1for dB; rank dB,dA,dD,dC -> RR1; NDCG=(1+7/log2(3))/(7+1/log2(3))','qB':'grade1for dC; rank dD,dC,dA,dB -> RR1/2; NDCG=1/log2(3)','qC':'no positive judgment -> all metrics0; included in denominator3'},'expected_mean':expected,'checks':checks}
(r/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
