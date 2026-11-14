"""CPU-only ranking quality against relevance judgments, distinct from ANN recall GT."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np


def require(condition, message):
    if not condition: raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def external(value):
    return str(value.item() if isinstance(value, np.generic) else value)


def load_qrels(path):
    qrels = defaultdict(dict)
    duplicates = conflicts = nonpositive = lines = 0
    with open(path, encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            if not line.strip(): continue
            cells = line.split()
            require(len(cells) == 4, f'qrels line {number}: expected qid Q0 docid grade')
            qid, _, did, grade = cells
            grade = float(grade)
            require(math.isfinite(grade), f'qrels line {number}: nonfinite grade')
            lines += 1
            if grade <= 0: nonpositive += 1
            if did in qrels[qid]:
                duplicates += 1
                conflicts += int(qrels[qid][did] != grade)
                grade = max(grade, qrels[qid][did])
            qrels[qid][did] = grade
    return dict(qrels), {'lines': lines, 'queries': len(qrels), 'duplicate_pairs': duplicates,
                         'conflicting_duplicate_pairs': conflicts, 'nonpositive_judgments': nonpositive,
                         'duplicate_policy': 'maximum grade per query-document pair'}


def quality(ranked_external_docids, judgments):
    rel = [judgments.get(did, 0.0) for did in ranked_external_docids[:10]]
    first = next((i + 1 for i, grade in enumerate(rel) if grade > 0), None)
    mrr = 0.0 if first is None else 1.0 / first
    gains = lambda grades: sum((2.0 ** grade - 1.0) / math.log2(i + 2)
                               for i, grade in enumerate(grades) if grade > 0)
    dcg = gains(rel)
    ideal = gains(sorted((v for v in judgments.values() if v > 0), reverse=True)[:10])
    require(math.isfinite(dcg) and math.isfinite(ideal), 'Relevance gain exceeds finite range')
    return {'MRR@10': mrr, 'NDCG@10': dcg / ideal if ideal > 0 else 0.0,
            'Success@5': float(any(grade > 0 for grade in rel[:5]))}


def plaid_records(path):
    with open(path, encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            if not line.strip(): continue
            item = json.loads(line)
            require(isinstance(item['round'], int) and item['round'] >= 0, f'Invalid round at line {number}')
            require(isinstance(item['query_row'], int), f'Invalid query row at line {number}')
            yield item['round'], item['query_row'], item['ids'], item.get('query_id')


def chimera_records(path, query_rows):
    current = None
    ranks = {}
    seen_queries = set()
    def emit(qid, ranked):
        require(sorted(ranked) == list(range(1, len(ranked) + 1)), 'Chimera result ranks must be contiguous from1')
        return 0, int(query_rows[qid]), [ranked[i] for i in range(1, len(ranked) + 1)], None
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        require({'query_id', 'rank', 'document_id'} <= set(reader.fieldnames or []), 'Missing Chimera TSV columns')
        for item in reader:
            qid = int(item['query_id'])
            require(0 <= qid < len(query_rows), 'Chimera local query row outside selected query list')
            if qid != current:
                if current is not None:
                    yield emit(current, ranks)
                require(qid not in seen_queries, 'Chimera query appears in separated duplicate groups')
                seen_queries.add(qid)
                current, ranks = qid, {}
            rank = int(item['rank'])
            require(rank > 0 and rank not in ranks, 'Invalid or duplicate Chimera rank')
            ranks[rank] = int(item['document_id'])
        if current is not None:
            yield emit(current, ranks)


def evaluate(a):
    data = Path(a.data)
    qpath, dpath = data/'queries_ids.npy', data/'doc_ids.npy'
    qids = np.load(qpath, allow_pickle=False)
    dids = np.load(dpath, mmap_mode='r', allow_pickle=False)
    require(qids.ndim == dids.ndim == 1 and len(qids) > 0 and len(dids) > 0, 'IDs must be nonempty one-dimensional arrays')
    require(len(np.unique(qids)) == len(qids), 'Query external IDs are not unique')
    n_docs = len(dids) if a.corpus_docs is None else a.corpus_docs
    require(0 < n_docs <= len(dids), 'corpus-docs exceeds document ID source')
    selected = np.load(a.query_rows, allow_pickle=False) if a.query_rows else np.arange(len(qids), dtype=np.int64)
    require(selected.ndim == 1 and np.issubdtype(selected.dtype, np.integer) and len(selected) > 0,
            'query-rows must be nonempty integer vector')
    require(len(np.unique(selected)) == len(selected) and np.all((selected >= 0) & (selected < len(qids))),
            'query-rows contain duplicate or invalid rows')
    expected = set(map(int, selected))
    qrels, label_stats = load_qrels(a.qrels)
    selected_qids = {external(qids[i]) for i in expected}
    positive = {qid for qid in selected_qids if any(v > 0 for v in qrels.get(qid, {}).values())}
    label_stats.update(selected_queries_with_any_judgment=sum(qid in qrels for qid in selected_qids),
                       selected_queries_with_positive_judgment=len(positive),
                       selected_queries_without_positive_judgment=len(selected_qids) - len(positive),
                       qrels_queries_outside_selected_set=len(set(qrels) - selected_qids))
    # Only a few thousand judged IDs need reverse membership; avoid a full Python map
    # of 8.8M document IDs. This is descriptive; IDCG always uses unfiltered qrels.
    rel_docids = {did for qid in selected_qids for did, grade in qrels.get(qid, {}).items() if grade > 0}
    indexed_relevant = set()
    try:
        relevant_values = np.asarray(sorted(rel_docids), dtype=dids.dtype)
        compare_as_string = False
    except (ValueError, OverflowError):
        relevant_values = np.asarray(sorted(rel_docids), dtype=str)
        compare_as_string = True
    for start in range(0, n_docs, 100000):
        block = dids[start:min(start+100000, n_docs)]
        comparable = block.astype(str) if compare_as_string else block
        mask = np.isin(comparable, relevant_values)
        indexed_relevant.update(external(x) for x in block[mask])
    label_stats['selected_queries_without_positive_judgment_in_indexed_corpus'] = sum(
        not any(grade > 0 and did in indexed_relevant for did, grade in qrels.get(qid, {}).items())
        for qid in selected_qids)
    rounds = {}
    sources = []
    for file_number, name in enumerate(a.results):
        path = Path(name)
        fmt = a.format
        if fmt == 'auto': fmt = 'plaid-jsonl' if path.suffix == '.jsonl' else 'chimera-tsv'
        sources.append({'path': str(path.resolve()), 'format': fmt, 'sha256': file_sha(path)})
        records = plaid_records(path) if fmt == 'plaid-jsonl' else chimera_records(path, selected)
        for local_round, row, docrows, reported_external_qid in records:
            require(row in expected, 'Result query row outside selected query list')
            if reported_external_qid is not None:
                require(external(reported_external_qid) == external(qids[row]), 'PLAID external qid disagrees with source row mapping')
            key = (file_number, local_round)
            if key not in rounds: rounds[key] = {'seen': set(), 'metrics': [], 'zero_results': 0}
            state = rounds[key]
            require(row not in state['seen'], 'Duplicate query within a result round')
            require(isinstance(docrows, list) and all(isinstance(i, int) for i in docrows), 'Document rows must be integer list')
            require(all(0 <= i < n_docs for i in docrows) and len(set(docrows)) == len(docrows),
                    'Duplicate or invalid document rows')
            docs = [external(dids[i]) for i in docrows[:10]]
            require(len(set(docs)) == len(docs), 'External document IDs duplicate within top10')
            state['seen'].add(row)
            state['metrics'].append(quality(docs, qrels.get(external(qids[row]), {})))
            state['zero_results'] += int(len(docrows) == 0)
    require(len(rounds) > 0, 'No result rounds parsed')
    if a.expected_rounds is not None: require(len(rounds) == a.expected_rounds, 'Unexpected result round count')
    metrics = ('MRR@10', 'NDCG@10', 'Success@5')
    per_round = []
    for (file_number, local_round), state in sorted(rounds.items()):
        missing = sorted(expected - state['seen'])
        require(a.allow_missing_results or not missing, 'Incomplete query coverage; use allow-missing-results only for an explicit partial diagnostic')
        result = {'source_file': sources[file_number]['path'], 'source_round': local_round,
                  'expected_queries': len(expected), 'observed_queries': len(state['seen']),
                  'missing_queries': len(missing), 'missing_query_rows': missing,
                  'queries_with_empty_results': state['zero_results'], 'input_coverage_complete': not missing}
        result.update({metric: sum(item[metric] for item in state['metrics']) / len(expected) for metric in metrics})
        per_round.append(result)
    identity = {'result_files': sources, 'qrels': {'path': str(Path(a.qrels).resolve()), 'sha256': file_sha(a.qrels)},
                'query_ids_file': {'path': str(qpath.resolve()), 'sha256': file_sha(qpath)},
                'document_ids_file': {'path': str(dpath.resolve()), 'sha256': file_sha(dpath)},
                'query_rows_file': None if not a.query_rows else {'path': str(Path(a.query_rows).resolve()), 'sha256': file_sha(a.query_rows)},
                'selected_query_rows_sha256': hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest(),
                'tool_sha256': file_sha(__file__)}
    return {'metric_family': 'qrels relevance ranking; distinct from ANN exact MaxSim recall',
            'protocol': {'denominator': 'every selected query, including queries with no positive judgment',
                         'no_positive_judgment': 'MRR/NDCG/Success all0',
                         'unjudged_document': 'relevance0', 'positive_relevance': '>0',
                         'NDCG_gain': '2**relevance-1', 'discount': 'log2(rank+1)',
                         'IDCG_scope': 'all supplied positive qrels for the query, not filtered by indexed prefix',
                         'missing_result_query': 'error by default; explicit allow-missing-results counts0',
                         'Chimera_query_id': 'local query row mapped through query-rows, or original full row order',
                         'document_id': 'corpus row mapped through doc_ids.npy'},
            'corpus_documents': n_docs, 'label_statistics': label_stats, 'identity': identity,
            'rounds': per_round, 'mean_round_metrics': {m: float(np.mean([v[m] for v in per_round])) for m in metrics},
            'all_result_queries_complete': all(v['input_coverage_complete'] for v in per_round)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--qrels', required=True)
    p.add_argument('--format', choices=['auto', 'plaid-jsonl', 'chimera-tsv'], default='auto')
    p.add_argument('--query-rows'); p.add_argument('--corpus-docs', type=int)
    p.add_argument('--expected-rounds', type=int); p.add_argument('--allow-missing-results', action='store_true')
    p.add_argument('--output', required=True); p.add_argument('results', nargs='+')
    a = p.parse_args(); result = evaluate(a)
    Path(a.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': a.output, 'rounds': len(result['rounds']),
                      'mean_round_metrics': result['mean_round_metrics'],
                      'label_statistics': result['label_statistics']}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
