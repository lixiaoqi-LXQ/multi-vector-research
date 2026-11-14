"""CPU acceptance and calibration projection of the completed canonical MS GT."""
import argparse
import json
import struct
from pathlib import Path
import numpy as np
from gt_state import array_sha, atomic_json, atomic_write, file_sha, require, validate_topk

BASE = Path('/home/lxq/graduate/data/multivector-colbertv2')
RAW_RECEIPT = Path('/home/lxq/graduate/data/memory/multivector-msmarco-full-array/documents-full-sha256-2895138.json')


def accept(output):
    data = BASE/'msmarco-v1-passage'
    gt = BASE/'groundtruth/msmarco-v1-passage-full'
    split = BASE/'splits/msmarco-calibration-seed42'
    m = json.loads((gt/'manifest.json').read_text())
    require(m.get('complete') is True, 'Full GT is still incomplete; no split output created')
    require(m['schema_version'] == 2 and m['docs'] == m['docs_done'] == 8841823,
            'GT completion count mismatch')
    require((m['num_tokens'], m['nq'], m['k']) == (597909919, 6980, 100), 'GT dimensions mismatch')
    require(m['corpus_source']['repository'] == 'tuskanny/ms_marco_colbertv2' and
            m['corpus_source']['revision'] == 'f8540899613ccc13049ac87db1cc4041a1dc4e31', 'GT source mismatch')
    require(m['arithmetic'] == 'FP32 GEMM (TF32 disabled), segmented max then FP32 sum', 'GT arithmetic mismatch')
    for name, digest in m['artifact_sha256'].items():
        require(file_sha(gt/name) == digest, 'GT artifact SHA mismatch: '+name)
    ids, scores = [np.load(gt/name, allow_pickle=False) for name in ['ids.npy', 'scores.npy']]
    validate_topk(ids, scores, 6980, 100, 8841823)
    require(array_sha(ids) == m['ids_payload_sha256'] and array_sha(scores) == m['scores_payload_sha256'],
            'GT payload digest mismatch')
    lens = np.load(data/'doclens.npy', allow_pickle=False).astype(np.int64)
    queries = np.load(data/'queries.npy', mmap_mode='r', allow_pickle=False)
    qids = np.load(data/'queries_ids.npy', allow_pickle=False)
    require(array_sha(lens) == m['doclens_sha256'] and array_sha(queries) == m['query_sha256'], 'GT input order mismatch')
    require(file_sha(data/'doc_ids.npy') == m['document_ids_sha256'], 'GT document IDs mismatch')
    require(array_sha(qids) == m['query_ids_sha256'], 'GT external query IDs mismatch')
    require(np.array_equal(np.load(gt/'query_ids.npy'), np.arange(6980)) and
            np.array_equal(np.load(gt/'query_external_ids.npy'), qids), 'Published GT query IDs mismatch')
    count = 0
    with open(gt/'groundtruth.tsv') as f:
        for count, line in enumerate(f, 1):
            q, d, rank = map(int, line.split())
            expected_q, expected_rank = divmod(count-1, 100)
            require(q == expected_q and rank == expected_rank+1 and d == int(ids[q, expected_rank]),
                    'GT TSV/array ordering mismatch')
    require(count == 698000, 'GT TSV completion count mismatch')
    receipt = json.loads(RAW_RECEIPT.read_text())
    require(receipt['status'] == 'VERIFIED_COMPLETE' and receipt['actual_sha256'] == receipt['expected_sha256'],
            'Source full-file SHA not accepted')
    st = (data/'documents.npy').stat()
    measured = dict(device=st.st_dev, inode=st.st_ino, bytes=st.st_size, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)
    require(measured == receipt['file_after'], 'Source changed after complete SHA verification')
    rows = np.load(split/'calibration-rows.npy')
    heldout = np.load(split/'heldout-rows.npy')
    expected = np.sort(np.random.default_rng(42).choice(6980, 100, replace=False))
    require(np.array_equal(rows, expected) and np.array_equal(heldout, np.setdiff1d(np.arange(6980), expected)),
            'Frozen split differs from declared seed42 selection')
    sm = json.loads((split/'manifest.json').read_text())
    require(array_sha(rows) == sm['calibration_rows_sha256'] and array_sha(heldout) == sm['heldout_rows_sha256'],
            'Split manifest row digest mismatch')
    with open(split/'queries.bin', 'rb') as f:
        require(struct.unpack('<3i', f.read(12)) == (100, 32, 128), 'Calibration query header mismatch')
        payload = np.frombuffer(f.read(), dtype='<f4').reshape(100, 32, 128)
    require(np.array_equal(payload, queries[rows]), 'Calibration query value/order mismatch')
    out = Path(output)
    require(not out.exists(), 'Refusing existing projection output')
    out.mkdir(parents=True)
    for name, value in [('ids.npy', ids[rows]), ('scores.npy', scores[rows]),
                        ('query_rows.npy', rows), ('query_external_ids.npy', qids[rows])]:
        atomic_write(out/name, lambda f, value=value: np.save(f, value, allow_pickle=False))
    with open(out/'groundtruth.tsv', 'x') as f:
        for q, row in enumerate(ids[rows]):
            for rank, did in enumerate(row, 1):
                f.write(f'{q}\t{did}\t{rank}\n')
    evidence = {'status':'PASS_FULL_MS_GT_AND_CALIBRATION_PROJECTION', 'complete':True,
                'docs':8841823, 'tokens':597909919, 'full_queries':6980, 'full_returned_ids':698000,
                'calibration_queries':100, 'heldout_queries':6880, 'k':100,
                'parent_manifest':str(gt/'manifest.json'), 'parent_manifest_sha256':file_sha(gt/'manifest.json'),
                'split_manifest':str(split/'manifest.json'), 'split_manifest_sha256':file_sha(split/'manifest.json'),
                'source_full_sha_receipt':str(RAW_RECEIPT), 'source_full_sha_receipt_sha256':file_sha(RAW_RECEIPT),
                'source_stat_unchanged':measured, 'validator_sha256':file_sha(__file__),
                'projection_policy':'calibration rows in frozen sorted order; TSV localquery0..99, document corpus rows',
                'artifact_sha256':{p.name:file_sha(p) for p in out.iterdir() if p.is_file()}}
    atomic_json(out/'projection-receipt.json', evidence)
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(accept(args.output), indent=2))
