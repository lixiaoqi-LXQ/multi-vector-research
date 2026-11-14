"""Post-window complete controlled outputs/heldout audit; no performance timings."""
import hashlib,json,itertools
from pathlib import Path
import numpy as np
root=Path(__file__).parent;p=root/'controls/lotte-prefix100000-code-payload';gt=np.load('/home/lxq/graduate/data/multivector-colbertv2/groundtruth/lotte-prefix-100000/ids.npy');held=set(np.load('/home/lxq/graduate/data/multivector-colbertv2/splits/lotte-calibration-seed42/heldout-rows.npy').tolist())
paths={mode:p/f'paired3-{mode}'/'per-query.jsonl' for mode in ['native','gpu-resident']};same_ids=True;maxerr=0;count=0;held_hits={mode:{r:{k:0 for k in [10,100]} for r in range(3)} for mode in paths};round_hashes={mode:{r:hashlib.sha256() for r in range(3)} for mode in paths}
with open(paths['native']) as fa,open(paths['gpu-resident']) as fb:
    for la,lb in itertools.zip_longest(fa,fb):
        assert la is not None and lb is not None
        aa,bb=json.loads(la),json.loads(lb);assert (aa['round'],aa['query_row'],aa['query_id'])==(bb['round'],bb['query_row'],bb['query_id'])
        same_ids &= aa['ids']==bb['ids'];maxerr=max(maxerr,max(abs(s-t) for s,t in zip(aa['scores'],bb['scores'])));count+=1
        for mode,x in [('native',aa),('gpu-resident',bb)]:
            round_hashes[mode][x['round']].update(json.dumps([x['query_row'],x['query_id'],x['ids'],x['scores']]).encode())
            if x['query_row'] in held:
                for k in [10,100]:held_hits[mode][x['round']][k]+=len(set(x['ids'][:k])&set(gt[x['query_row'],:k]))
assert count==8793
out={'queries_per_round':2931,'rounds':3,'records_per_variant':count,'same_all_returned_ids':same_ids,'max_abs_score_error_all_queries':maxerr,'controlled_change':'Only codes+residual payload placement; IVF/doclens/offsets remain CPU','native_label':'native','resident_label':'gpu-code-payload (legacy selector gpu-resident)','variants':{}}
for mode in paths:
    summary=json.loads((paths[mode].parent/'summary.json').read_text());out['variants'][mode]={'summary':str(paths[mode].parent/'summary.json'),'rounds':[],'round_payload_sha256':{r:h.hexdigest() for r,h in round_hashes[mode].items()}}
    for r,record in enumerate(summary['rounds']):out['variants'][mode]['rounds'].append(record|{f'heldout_recall@{k}':held_hits[mode][r][k]/(len(held)*k) for k in [10,100]})
qn=np.mean([r['qps'] for r in out['variants']['native']['rounds']]);qg=np.mean([r['qps'] for r in out['variants']['gpu-resident']['rounds']]);out['ratio_resident_over_native_mean_qps']=float(qg/qn)
(p/'controlled-comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='variants'},indent=2))
