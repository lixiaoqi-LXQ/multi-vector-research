"""Exhaustive FP32 MaxSim over complete document blocks; durable, identity-checked resume."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from gt_state import (ARITHMETIC, SCHEMA, array_sha, atomic_json, atomic_write,
                      canonical_sha, digest, file_sha, read_resume, require,
                      save_checkpoint, snapshot_meta, source_identity, validate_topk)


class StreamingCorpus:
    """Pinned inode with append-only source validation; atomic rename is safe."""
    def __init__(self, root, n, wait_download=False):
        self.root = Path(root)
        self.source = source_identity(self.root)
        path = self.root / 'documents.npy'
        if not path.exists(): path = self.root / 'documents.npy.partial'
        self.file = path.open('rb')
        try:
            version = np.lib.format.read_magic(self.file)
            require(version in ((1, 0), (2, 0)), 'Only NPY format v1/v2 is supported')
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = reader(self.file)
            self.offset = self.file.tell()
            require(not fortran and len(shape) == 2 and shape[1] == 128, 'Expected C-order document [N,128] array')
            require(dtype in (np.dtype('<u2'), np.dtype('<f2')), 'Expected FP16 or raw uint16 FP16 bits')
            require(0 < n <= shape[0], 'Requested token prefix exceeds document header')
            self.shape = (n, 128)
            self.header = os.pread(self.file.fileno(), self.offset, 0)
            state = os.fstat(self.file.fileno())
            self.inode = (state.st_dev, state.st_ino)
            self.last_size = state.st_size
            self.expected_bytes = self.offset + int(shape[0]) * 256
            require(state.st_size <= self.expected_bytes, 'Corpus file exceeds its declared shape')
            self.wait_download = wait_download
            self.identity = self.source | {'header_sha256': digest(self.header), 'header_shape': list(shape),
                                           'header_dtype': str(dtype), 'encoding': 'raw IEEE FP16 bits'}
            if not wait_download:
                require(state.st_size >= self.offset + n * 256, 'Requested document prefix is not fully downloaded')
        except Exception:
            self.file.close()
            raise

    def __len__(self): return self.shape[0]

    def check_source(self):
        require(source_identity(self.root) == self.source, 'Download provenance source SHA/repository changed')
        require(os.pread(self.file.fileno(), self.offset, 0) == self.header, 'Corpus NPY header changed')
        state = os.fstat(self.file.fileno())
        require(state.st_size >= self.last_size, 'Corpus file was truncated during processing')
        require(state.st_size <= self.expected_bytes, 'Corpus file exceeds its declared shape')
        self.last_size = state.st_size
        # A rename from .partial to final preserves this inode. Replacements do not.
        current = self.root / 'documents.npy'
        if not current.exists(): current = self.root / 'documents.npy.partial'
        try: now = current.stat()
        except FileNotFoundError: return state.st_size  # rename gap; pinned fd remains valid
        require((now.st_dev, now.st_ino) == self.inode, 'Corpus path was replaced by another file')
        return state.st_size

    def __getitem__(self, key):
        require(isinstance(key, slice) and key.step in (None, 1), 'Corpus supports contiguous slices only')
        start = 0 if key.start is None else int(key.start)
        stop = self.shape[0] if key.stop is None else int(key.stop)
        require(0 <= start < stop <= self.shape[0], 'Invalid document-token slice')
        required = self.offset + stop * 256
        while self.check_source() < required:
            require(self.wait_download, 'Requested document slice is not downloaded')
            print(json.dumps({'waiting_download_bytes': required, 'have': self.last_size}), flush=True)
            time.sleep(10)
        return np.memmap(self.file, mode='r', offset=self.offset + start * 256,
                         dtype='<f2', shape=(stop - start, 128))

    def hash_prefix(self, tokens):
        h = hashlib.sha256()
        for start in range(0, tokens, 32768):
            h.update(np.ascontiguousarray(self[start:min(start + 32768, tokens)]).tobytes())
        return h

    def close(self): self.file.close()


def load_prefix(path, docs, wait_download=False):
    require(docs > 0, 'docs must be positive')
    raw = np.load(Path(path) / 'doclens.npy', allow_pickle=False)
    require(raw.ndim == 1 and np.issubdtype(raw.dtype, np.integer), 'Document lengths must be a one-dimensional integer array')
    require(docs <= len(raw), 'Requested documents exceed source collection')
    lens = raw[:docs].astype(np.int64)
    require(np.all(raw > 0), 'Document lengths must be positive')
    corpus = StreamingCorpus(path, int(lens.sum()), wait_download)
    try:
        require(int(raw.sum(dtype=np.int64)) == corpus.identity['header_shape'][0],
                'Complete document lengths do not cover the source token array')
    except Exception:
        corpus.close()
        raise
    corpus.total_documents = len(raw)
    return lens, corpus


def make_identity(root, lens, q, k, corpus):
    root = Path(root)
    dids = np.load(root / 'doc_ids.npy', mmap_mode='r', allow_pickle=False)
    qids = np.load(root / 'queries_ids.npy', allow_pickle=False)[:len(q)]
    require(len(dids) == corpus.total_documents and len(qids) == len(q), 'ID count mismatch')
    require(len(np.unique(dids)) == len(dids), 'Document external IDs must be unique')
    require(len(np.unique(qids)) == len(qids), 'Query external IDs must be unique')
    return {'schema_version': SCHEMA, 'corpus_source': corpus.identity,
            'corpus_source_id': canonical_sha(corpus.identity), 'query_sha256': array_sha(q),
            'query_ids_sha256': array_sha(qids), 'document_ids_sha256': file_sha(root / 'doc_ids.npy'),
            'doclens_sha256': array_sha(lens), 'nq': len(q), 'k': k, 'arithmetic': ARITHMETIC}


def legacy_upgrade(path):
    """Explicit local provenance backfill for this session's completed FP32 prefix."""
    path = Path(path)
    m = json.loads((path / 'manifest.json').read_text())
    require(m.get('complete') is True and 'schema_version' not in m, 'Expected a completed legacy artifact')
    require(m.get('arithmetic') == ARITHMETIC, 'Legacy arithmetic is not recognized')
    root = Path(m['data']).resolve()
    lens, corpus = load_prefix(root, int(m['docs']))
    try:
        q = np.load(root / 'queries.npy', allow_pickle=False)
        if m['queries']: q = q[:m['queries']]
        require(m['nq'] == len(q) and m['query_sha256'] == array_sha(q), 'Legacy query identity mismatch')
        require(m['doclens_sha256'] == array_sha(lens) and m['num_tokens'] == int(lens.sum()), 'Legacy corpus-length identity mismatch')
        ids = np.load(path / 'ids.npy', allow_pickle=False)
        scores = np.load(path / 'scores.npy', allow_pickle=False)
        validate_topk(ids, scores, len(q), m['k'], len(lens))
        # Keep the attestation explicit: this content digest is measured now.
        old = dict(m)
        meta = snapshot_meta(make_identity(root, lens, q, m['k'], corpus), ids, scores,
                             len(lens), corpus.hash_prefix(int(lens.sum())).hexdigest(), m)
        meta.update(complete=True, schema_version=SCHEMA,
                    identity_backfill={'kind': 'session-provenance-backfill', 'time': time.time(),
                                       'note': 'Input/output identities measured after the original run; original query and doclens digests matched',
                                       'legacy_source_sha256': old['source_sha256']})
        files = ['ids.npy', 'scores.npy', 'groundtruth.tsv']
        meta['artifact_sha256'] = {name: file_sha(path / name) for name in files}
        atomic_json(path / 'manifest.legacy-v1.json', old)
        atomic_json(path / 'manifest.json', meta)
        return meta
    finally: corpus.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data'); p.add_argument('--docs', type=int)
    p.add_argument('--queries', type=int, default=0); p.add_argument('--out')
    p.add_argument('--query-batch', type=int, default=16); p.add_argument('--token-block', type=int, default=131072)
    p.add_argument('--resume'); p.add_argument('--wait-download', action='store_true')
    p.add_argument('--reducer', choices=['torch', 'triton'], default='torch')
    p.add_argument('--k', type=int, default=100); p.add_argument('--threads', type=int, default=4)
    p.add_argument('--upgrade-legacy', metavar='COMPLETED_DIRECTORY')
    a = p.parse_args()
    if a.upgrade_legacy:
        print(json.dumps(legacy_upgrade(a.upgrade_legacy), indent=2)); return
    require(a.data and a.docs and a.out, '--data, --docs and --out are required')
    require(a.docs > 0 and 0 < a.k <= a.docs, 'Require 0 < k <= docs')
    require(a.queries >= 0 and a.query_batch > 0 and a.token_block > 0 and a.threads > 0, 'Invalid count/batch/thread parameter')
    lens, corpus = load_prefix(a.data, a.docs, a.wait_download)
    try: run(a, lens, corpus)
    finally: corpus.close()


def run(a, lens, corpus):
    q = np.load(Path(a.data) / 'queries.npy', allow_pickle=False)
    require(q.ndim == 3 and q.shape[1:] == (32, 128) and len(q) > 0, 'Expected queries [Q,32,128]')
    require(q.dtype == np.float32 and np.isfinite(q).all(), 'Queries must be finite FP32')
    require(a.queries <= len(q), 'Requested queries exceed source query count')
    if a.queries: q = q[:a.queries]
    nq, ql, d = q.shape
    identity = make_identity(a.data, lens, q, a.k, corpus)
    ds = 0; prior = None; h = hashlib.sha256(); prior_ids = prior_scores = None
    if a.resume:
        ds, prior_ids, prior_scores, h, prior = read_resume(a.resume, identity, lens,
                                                        identity['query_sha256'], corpus, a.k)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mp = out / 'manifest.json'
    require(not mp.exists() or json.loads(mp.read_text()).get('complete') is not True,
            'Refusing to overwrite completed GT; use a new output directory')
    atomic_json(mp, identity | {'complete': False, 'status': 'running', 'docs': len(lens)})
    import torch
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    qt = torch.from_numpy(q.copy()).cuda().float()
    top_scores = torch.full((nq, a.k), -torch.inf, device='cuda', dtype=torch.float32)
    top_ids = torch.full((nq, a.k), -1, device='cuda', dtype=torch.long)
    if prior is not None:
        top_scores.copy_(torch.from_numpy(prior_scores).cuda())
        top_ids.copy_(torch.from_numpy(prior_ids).cuda())
    if a.reducer == 'triton': from segmented_maxsim import maxsim
    offsets = np.r_[0, np.cumsum(lens)]
    started = time.time(); blocks = 0; checkpoint_doc = ds
    resume_chain = ([] if prior is None else prior.get('resume_chain', []) + [
        {'path': str(Path(a.resume).resolve()), 'docs_done': ds,
         'reducer': prior.get('reducer', 'torch'), 'source_sha256': prior.get('source_sha256')}])

    def metadata(ids, scores, complete=False):
        current = identity | {'doclens_sha256': array_sha(lens[:ds])}
        extra = vars(a) | {'complete': complete, 'docs': len(lens), 'num_tokens': int(offsets[ds]),
                          'blocks_this_run': blocks, 'elapsed_s': time.time() - started,
                          'torch': torch.__version__, 'cuda': torch.version.cuda,
                          'device': torch.cuda.get_device_name(), 'resume_chain': resume_chain,
                          'source_sha256': file_sha(__file__), 'state_source_sha256': file_sha(Path(__file__).with_name('gt_state.py')),
                          'tie_policy': 'Any top-k by FP32 score; equal-score document order is unspecified'}
        if a.reducer == 'triton': extra['reducer_source_sha256'] = file_sha(Path(__file__).with_name('segmented_maxsim.py'))
        return snapshot_meta(current, ids, scores, ds, h.hexdigest(), extra)

    while ds < len(lens):
        de = min(len(lens), max(ds + 1, int(np.searchsorted(offsets, offsets[ds] + a.token_block, side='right') - 1)))
        raw = np.ascontiguousarray(corpus[offsets[ds]:offsets[de]])
        require(np.isfinite(raw).all(), 'Nonfinite document token vectors')
        h.update(raw.tobytes())
        x = torch.from_numpy(raw.astype(np.float32)).cuda()
        lt = torch.from_numpy(lens[ds:de].copy()).cuda()
        local_offsets = torch.tensor(offsets[ds:de+1] - offsets[ds], device='cuda')
        for qs in range(0, nq, a.query_batch):
            qe = min(qs + a.query_batch, nq)
            sim = qt[qs:qe].reshape(-1, d) @ x.T
            if a.reducer == 'triton':
                scores = maxsim(sim, local_offsets, qe - qs, int(lens[ds:de].max()))
            else:
                seg = torch.segment_reduce(sim.T.contiguous(), reduce='max', lengths=lt, axis=0)
                scores = seg.reshape(de-ds, qe-qs, ql).sum(-1).T.contiguous()
            take = min(a.k, de-ds)
            v, ix = torch.topk(scores, take, dim=1)
            allv = torch.cat([top_scores[qs:qe], v], dim=1)
            alli = torch.cat([top_ids[qs:qe], ix+ds], dim=1)
            nv, ni = torch.topk(allv, a.k, dim=1)
            top_scores[qs:qe] = nv; top_ids[qs:qe] = alli.gather(1, ni)
        torch.cuda.synchronize()
        blocks += 1; ds = de
        if ds - checkpoint_doc >= 10000:
            ids, scores = top_ids.cpu().numpy(), top_scores.cpu().numpy()
            save_checkpoint(out / 'checkpoint.npz', ids, scores, metadata(ids, scores))
            checkpoint_doc = ds
        print(json.dumps({'docs_done': ds, 'docs_total': len(lens), 'blocks': blocks,
                          'elapsed_s': time.time()-started}), flush=True)
    corpus.check_source()
    ids, scores = top_ids.cpu().numpy(), top_scores.cpu().numpy()
    validate_topk(ids, scores, nq, a.k, len(lens))
    final = metadata(ids, scores, complete=True)
    external_qids = np.load(Path(a.data) / 'queries_ids.npy', allow_pickle=False)[:nq]
    for name, value in [('ids.npy', ids), ('scores.npy', scores), ('query_ids.npy', np.arange(nq, dtype=np.int64)),
                        ('query_external_ids.npy', external_qids)]:
        atomic_write(out / name, lambda f, value=value: np.save(f, value, allow_pickle=False))
    def write_tsv(f):
        for qi, row in enumerate(ids):
            f.write(''.join(f'{qi}\t{di}\t{rank}\n' for rank, di in enumerate(row, 1)).encode())
    atomic_write(out / 'groundtruth.tsv', write_tsv)
    final['artifact_sha256'] = {name: file_sha(out / name) for name in ('ids.npy', 'scores.npy', 'query_ids.npy', 'query_external_ids.npy', 'groundtruth.tsv')}
    final['id_convention'] = {'ids.npy': 'zero-based corpus rows', 'query_ids.npy': 'zero-based query rows',
                              'query_external_ids.npy': 'source external query IDs in query-row order',
                              'document_external_ids_source': str((Path(a.data) / 'doc_ids.npy').resolve())}
    atomic_json(mp, final)


if __name__ == '__main__': main()
