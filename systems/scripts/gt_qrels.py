"""CPU relevance metrics for the accepted exhaustive FP32 MaxSim reference ranking."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from gt_state import array_sha, atomic_json, file_sha, require, validate_topk
import qrels_metrics as qm


def run(data, gt, qrels, output):
    data, gt, qrels, output = map(Path, (data, gt, qrels, output))
    m = json.loads((gt/'manifest.json').read_text())
    require(m.get('complete') is True and m['schema_version'] == 2, 'Require completed canonical GT')
    for name, digest in m['artifact_sha256'].items():
        require(file_sha(gt/name) == digest, 'GT artifact SHA mismatch: '+name)
    ids, scores = [np.load(gt/name, allow_pickle=False) for name in ['ids.npy','scores.npy']]
    validate_topk(ids, scores, m['nq'], m['k'], m['docs'])
    require(m['k'] >= 10, 'Reference has fewer than ten ranks')
    require(array_sha(ids) == m['ids_payload_sha256'] and array_sha(scores) == m['scores_payload_sha256'], 'GT payload identity mismatch')
    qids = np.load(data/'queries_ids.npy', allow_pickle=False)
    require(len(qids) == m['nq'] and array_sha(qids) == m['query_ids_sha256'], 'GT/source query ID mismatch')
    require(file_sha(data/'doc_ids.npy') == m['document_ids_sha256'], 'GT/source document ID mismatch')
    require(not output.exists(), 'Refusing existing reference metrics directory')
    output.mkdir(parents=True)
    exported = output/'reference-results.jsonl'
    with open(exported, 'x') as f:
        for row, values in enumerate(ids):
            f.write(json.dumps(dict(round=0, query_row=row, query_id=qm.external(qids[row]),
                                   ids=values.astype(np.int64).tolist()))+'\n')
    args = SimpleNamespace(data=str(data), qrels=str(qrels), format='plaid-jsonl', query_rows=None,
                           corpus_docs=m['docs'], expected_rounds=1, allow_missing_results=False,
                           results=[str(exported)])
    result = qm.evaluate(args)
    result['reference_origin'] = dict(kind='exhaustive FP32 MaxSim on released vectors',
                                    GT_manifest=str(gt/'manifest.json'), GT_manifest_sha256=file_sha(gt/'manifest.json'),
                                    export_and_evaluation_script_sha256=file_sha(__file__),
                                    performance_claim=False)
    atomic_json(output/'qrels-metrics.json', result)
    return dict(output=str(output/'qrels-metrics.json'), metrics=result['mean_round_metrics'],
                complete=result['all_result_queries_complete'], queries=m['nq'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data','gt','qrels','output']:
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.data, a.gt, a.qrels, a.output), indent=2))
