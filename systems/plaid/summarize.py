"""Strict completion/identity validation; quality reporting never selects parameters."""
import argparse,json,math
from pathlib import Path
import numpy as np
from identity import gt_identity,index_identity,file_sha,require

def validate_result(result,groundtruth,*,expected_rounds,expected_query_rows,topk,data,index,heldout_rows=None,index_placement='native',output_summary=None):
    result=Path(result);gt_info=gt_identity(str(groundtruth),str(data),json.loads((Path(index)/'metadata.json').read_text())['num_documents'])
    ix=index_identity(str(index));binding=ix['input']
    for key in ['corpus_source_id','query_sha256','query_ids_sha256','document_ids_sha256','doclens_sha256','docs','num_tokens']:
        require(binding[key]==gt_info[key],'Index/GT input mismatch: '+key)
    require(binding['scanned_prefix_sha256']==gt_info['gt_prefix_sha256'],'Index/GT corpus payload mismatch')
    contract=json.loads((result/'contract.json').read_text())
    require(contract['index_artifact_manifest_sha256']==ix['artifact_manifest_sha256'],'Result/index identity mismatch')
    require(contract['gt_manifest_sha256']==gt_info['gt_manifest_sha256'],'Result/GT manifest mismatch')
    require(contract.get('profile_range') is False,'Profile results cannot be accepted as QPS evidence')
    require(contract.get('query_placement')=='host','Formal timing must start with host FP32 query')
    require(contract.get('index_placement')==index_placement,'Unexpected index placement')
    run=json.loads((result/'run-identity.json').read_text())
    require(run.get('query_placement')=='host' and run.get('profile_range') is False,'Run timing/profile identity mismatch')
    require(run.get('index_placement')==index_placement,'Run index placement mismatch')
    devices=run['payload_devices']
    expected={'codes':'cpu','residuals':'cpu','ivf':'cpu'} if index_placement=='native' else {'codes':'cuda','residuals':'cuda','ivf':'cuda' if index_placement=='gpu-all-payload' else 'cpu'}
    require(all(devices[k].split(':')[0]==v for k,v in expected.items()),'Actual index payload devices disagree with placement label')
    rows=np.load(expected_query_rows,allow_pickle=False).astype(np.int64);require(rows.ndim==1 and len(np.unique(rows))==len(rows),'Invalid expected query rows')
    require(rows.tolist()==contract['expected_query_rows'],'Query contract mismatch')
    require(expected_rounds==contract['expected_rounds'] and topk==contract['topk'],'Round/topk contract mismatch')
    require(expected_rounds>0 and topk>0 and topk<=100,'Invalid expected rounds/topk')
    require(np.all((rows>=0)&(rows<gt_info['nq'])),'Expected query row outside source')
    qids=np.load(Path(data)/'queries_ids.npy',allow_pickle=False)
    gt=np.load(Path(groundtruth)/'ids.npy',allow_pickle=False)
    records=[]
    with open(result/'per-query.jsonl') as f:
        for line in f:
            require(bool(line.strip()),'Blank result row');records.append(json.loads(line))
    require(len(records)==expected_rounds*len(rows),'Result count mismatch: missing/truncated/extra round')
    heldout=set(np.load(heldout_rows,allow_pickle=False).tolist()) if heldout_rows else None
    out={'complete':True,'completed':len(records),'expected_rounds':expected_rounds,'queries_per_round':len(rows),'topk':topk,'index_placement':index_placement,'query_placement':'host','input_identity':gt_info,'index_identity':ix,'result_sha256':file_sha(result/'per-query.jsonl'),'rounds':[]}
    for r in range(expected_rounds):
        rr=records[r*len(rows):(r+1)*len(rows)];seen=set()
        for expected_row,x in zip(rows,rr):
            require(type(x['round']) is int and x['round']==r,'Missing/reordered/duplicate round')
            require(type(x['query_row']) is int and x['query_row']==int(expected_row),'Missing/reordered query row')
            require(type(x['query_id']) is int and x['query_id']==int(qids[expected_row]),'Query external ID mismatch')
            require(x['query_id'] not in seen,'Duplicate query ID');seen.add(x['query_id'])
            ids=x['ids'];scores=x['scores']
            require(len(ids)==len(scores)==topk,'Missing top-k IDs/scores')
            require(all(type(v) is int and 0<=v<binding['docs'] for v in ids),'ID type/bounds mismatch')
            require(len(set(ids))==topk,'Duplicate result ID')
            require(all(type(s) in (int,float) and math.isfinite(s) for s in scores),'Non-finite score')
            require(all(scores[j]>=scores[j+1] for j in range(topk-1)),'Unsorted result scores')
            require(type(x['elapsed_ms']) in (int,float) and math.isfinite(x['elapsed_ms']) and x['elapsed_ms']>0,'Invalid elapsed time')
        lat=np.asarray([x['elapsed_ms'] for x in rr]);item={'round':r,'queries':len(rr),'mean_ms':float(lat.mean()),'p99_ms':float(np.quantile(lat,.99)),'qps':float(1000/lat.mean())}
        for k in [k for k in (10,100) if k<=topk]:item[f'recall@{k}']=float(np.mean([len(set(x['ids'][:k])&set(gt[x['query_row'],:k]))/k for x in rr]))
        if heldout is not None:
            hh=[x for x in rr if x['query_row'] in heldout];require(len(hh)==len(heldout),'Heldout completion mismatch');item['heldout_queries']=len(hh)
            for k in [k for k in (10,100) if k<=topk]:item[f'heldout_recall@{k}']=float(np.mean([len(set(x['ids'][:k])&set(gt[x['query_row'],:k]))/k for x in hh]))
        out['rounds'].append(item)
    Path(output_summary or result/'summary.json').write_text(json.dumps(out,indent=2));return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('result');p.add_argument('--groundtruth',required=True);p.add_argument('--data',required=True);p.add_argument('--index',required=True);p.add_argument('--heldout-rows');p.add_argument('--expected-rounds',type=int,required=True);p.add_argument('--expected-query-rows',required=True);p.add_argument('--topk',type=int,required=True);p.add_argument('--index-placement',choices=['native','gpu-code-payload','gpu-all-payload','gpu-resident'],default='native');a=p.parse_args()
    print(json.dumps(validate_result(a.result,a.groundtruth,expected_rounds=a.expected_rounds,expected_query_rows=a.expected_query_rows,topk=a.topk,data=a.data,index=a.index,heldout_rows=a.heldout_rows,index_placement=a.index_placement),indent=2))
