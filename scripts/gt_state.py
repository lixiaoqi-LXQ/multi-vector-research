"""CPU-only persistence and identity checks for exhaustive MaxSim ground truth."""
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
import numpy as np

SCHEMA = 2
ARITHMETIC = 'FP32 GEMM (TF32 disabled), segmented max then FP32 sum'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def array_sha(a):
    return digest(np.ascontiguousarray(a).tobytes())

def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()

def canonical_sha(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())

def require(condition, message):
    if not condition: raise ValueError(message)

def source_identity(root):
    p = json.loads((Path(root) / 'download-provenance.json').read_text())
    repo, rev = p['repository'], p['revision']
    require(isinstance(repo, str) and len(repo.split('/')) == 2, 'Invalid source repository')
    require(isinstance(rev, str) and re.fullmatch('[0-9a-fA-F]{40}', rev), 'Source revision must be a fixed 40-character SHA')
    base = f'https://huggingface.co/datasets/{repo}/resolve/{rev}/'
    for name, detail in p.get('files', {}).items():
        require(detail.get('url') == base + name, 'Source file URL disagrees with pinned repository/revision: ' + name)
    return {'repository': repo, 'revision': rev, 'documents_url': base + 'documents.npy'}

def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(name): os.unlink(name)

def atomic_json(path, value):
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode()
    atomic_write(path, lambda f: f.write(payload))

def validate_topk(ids, scores, nq, k, docs_done):
    require(ids.shape == scores.shape == (nq, k), 'Top-k array shape mismatch')
    require(np.issubdtype(ids.dtype, np.integer), 'Top-k IDs must be integers')
    require(scores.dtype == np.float32, 'Top-k scores must be FP32')
    valid = ids >= 0
    require(np.all((ids == -1) | ((ids >= 0) & (ids < docs_done))), 'Top-k ID outside scanned document prefix')
    require(np.all(valid.sum(1) == min(k, docs_done)), 'Top-k completion count mismatch')
    require(np.isfinite(scores[valid]).all() and np.all(np.isneginf(scores[~valid])), 'Invalid top-k scores')
    require(np.all(scores[:, 1:] <= scores[:, :-1]), 'Top-k scores are not descending')
    require(all(len(np.unique(row[row >= 0])) == min(k, docs_done) for row in ids), 'Duplicate top-k IDs')

def snapshot_meta(identity, ids, scores, docs_done, prefix_sha, extra=None):
    meta = dict(identity)
    meta.update(extra or {})
    meta.update(schema_version=SCHEMA, docs_done=int(docs_done), scanned_prefix_sha256=prefix_sha,
                ids_payload_sha256=array_sha(ids), scores_payload_sha256=array_sha(scores))
    return meta

def save_checkpoint(path, ids, scores, metadata):
    validate_topk(ids, scores, metadata['nq'], metadata['k'], metadata['docs_done'])
    atomic_write(path, lambda f: np.savez(f, ids=ids, scores=scores,
                 metadata=np.asarray(json.dumps(metadata, sort_keys=True))))

def read_resume(path, identity, lens, query_sha, corpus, k):
    path = Path(path)
    if path.is_dir():
        mp = path / 'manifest.json'
        m = json.loads(mp.read_text()) if mp.exists() else {}
        if m.get('complete') is True:
            require(m.get('schema_version') == SCHEMA, 'Legacy manifest needs explicit verified upgrade before resume')
            for filename, expected in m['artifact_sha256'].items():
                require(file_sha(path / filename) == expected, 'Completed artifact digest mismatch: ' + filename)
            ids = np.load(path / 'ids.npy', allow_pickle=False)
            scores = np.load(path / 'scores.npy', allow_pickle=False)
        else:
            path = path / 'checkpoint.npz'
            m = None
    else: m = None
    if m is None:
        with np.load(path, allow_pickle=False) as state:
            m = json.loads(str(state['metadata'].item()))
            ids, scores = state['ids'].copy(), state['scores'].copy()
    require(m.get('schema_version') == SCHEMA, 'Unsupported checkpoint schema')
    for key in ('corpus_source_id', 'query_sha256', 'query_ids_sha256', 'document_ids_sha256', 'nq', 'k', 'arithmetic'):
        require(m.get(key) == identity.get(key), 'Resume identity mismatch: ' + key)
    ds = int(m['docs_done'])
    require(0 < ds <= len(lens), 'Resume scanned prefix exceeds target corpus')
    require(m['doclens_sha256'] == array_sha(lens[:ds]), 'Resume document-length prefix mismatch')
    require(m['query_sha256'] == query_sha, 'Resume query digest mismatch')
    require(array_sha(ids) == m['ids_payload_sha256'] and array_sha(scores) == m['scores_payload_sha256'], 'Resume top-k payload digest mismatch')
    validate_topk(ids, scores, identity['nq'], k, ds)
    h = corpus.hash_prefix(int(lens[:ds].sum()))
    require(h.hexdigest() == m['scanned_prefix_sha256'], 'Resume corpus prefix content changed')
    return ds, ids, scores, h, m
