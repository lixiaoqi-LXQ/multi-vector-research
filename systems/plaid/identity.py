"""CPU-only binding of PLAID inputs/indexes/results to the verified GT identity."""
import hashlib,json,sys,time
from functools import lru_cache
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'scripts'))
from gt_state import array_sha,file_sha,canonical_sha,require,validate_topk,atomic_json
from exact_gt import StreamingCorpus,make_identity

@lru_cache(maxsize=8)
def input_identity(data,docs):
    data=Path(data);raw_lens=np.load(data/'doclens.npy',allow_pickle=False);lens=raw_lens[:docs].astype(np.int64)
    q=np.load(data/'queries.npy',allow_pickle=False)
    corpus=StreamingCorpus(data,int(lens.sum()));corpus.total_documents=len(raw_lens)
    try:x=make_identity(data,lens,q,100,corpus)
    finally:corpus.close()
    x.update(docs=docs,num_tokens=int(lens.sum()));return x

@lru_cache(maxsize=8)
def gt_identity(gt,data,docs):
    gt=Path(gt);m=json.loads((gt/'manifest.json').read_text());require(m.get('complete') is True,'GT is incomplete');require(m.get('schema_version')==2,'GT requires schema2')
    actual=dict(input_identity(data,docs))
    for key in ['corpus_source_id','query_sha256','query_ids_sha256','document_ids_sha256','doclens_sha256','nq','k','docs','num_tokens']:
        require(actual[key]==m[key],'GT/input identity mismatch: '+key)
    for name in ['ids.npy','scores.npy','query_ids.npy','query_external_ids.npy']:
        require(file_sha(gt/name)==m['artifact_sha256'][name],'GT artifact modified: '+name)
    ids=np.load(gt/'ids.npy',allow_pickle=False);scores=np.load(gt/'scores.npy',allow_pickle=False)
    validate_topk(ids,scores,m['nq'],m['k'],docs)
    require(array_sha(ids)==m['ids_payload_sha256'],'GT ID payload mismatch');require(array_sha(scores)==m['scores_payload_sha256'],'GT score payload mismatch')
    actual.update(gt_manifest_sha256=file_sha(gt/'manifest.json'),gt_ids_sha256=file_sha(gt/'ids.npy'),gt_prefix_sha256=m['scanned_prefix_sha256'])
    return actual

def split_identity(split,data):
    split=Path(split);m=json.loads((split/'manifest.json').read_text());q=np.load(Path(data)/'queries.npy',allow_pickle=False);cal=np.load(split/'calibration-rows.npy',allow_pickle=False);held=np.load(split/'heldout-rows.npy',allow_pickle=False)
    require(m.get('schema_version')==2 and m.get('seed')==42,'Split protocol version/seed mismatch')
    require(cal.shape==(100,) and len(np.unique(cal))==100,'Expected100 unique calibration rows')
    require(np.array_equal(cal,np.sort(np.random.default_rng(42).choice(len(q),100,replace=False))),'Calibration rows disagree with seed42')
    require(np.array_equal(held,np.setdiff1d(np.arange(len(q)),cal)),'Heldout is not exact independent complement')
    for key,actual in [('query_sha256',array_sha(q)),('calibration_query_sha256',array_sha(q[cal])),('calibration_rows_sha256',array_sha(cal)),('heldout_rows_sha256',array_sha(held))]:
        require(m[key]==actual,'Split identity mismatch: '+key)
    return {'manifest_sha256':file_sha(split/'manifest.json'),'calibration_rows_sha256':array_sha(cal),'heldout_rows_sha256':array_sha(held),'query_sha256':array_sha(q),'calibration_query_sha256':array_sha(q[cal])}

def artifact_stats(index,artifacts):
    names=set(artifacts)|{'index-artifact-identity.json','input-identity.json'}
    out={}
    for name in sorted(names):
        st=(index/name).stat();out[name]={'device':st.st_dev,'inode':st.st_ino,'size':st.st_size,'mtime_ns':st.st_mtime_ns}
    return out

def freeze_index_stat_guard(index,verified_at_build=False):
    index=Path(index);artifacts=json.loads((index/'index-artifact-identity.json').read_text())
    row={'artifact_manifest_sha256':file_sha(index/'index-artifact-identity.json'),'input_identity_sha256':file_sha(index/'input-identity.json'),'stats':artifact_stats(index,artifacts),'verified_at':time.time(),'verification':'streamed build SHA plus final metadata SHA' if verified_at_build else 'full payload SHA'}
    atomic_json(index/'index-stat-guard.json',row);return row

def index_identity(index,force_full=False):
    index=Path(index);meta=json.loads((index/'metadata.json').read_text());binding=json.loads((index/'input-identity.json').read_text());artifacts=json.loads((index/'index-artifact-identity.json').read_text())
    require(meta['num_documents']==binding['docs'] and meta['num_embeddings']==binding['num_tokens'],'Index metadata/input dimension mismatch')
    require(meta['config']['nbits'] in [1,2,4,8],'Unsupported index bitwidth')
    for chunk in range(meta['num_chunks']):
        for name in [f'{chunk}.codes.pt',f'{chunk}.residuals.pt',f'doclens.{chunk}.json']:
            require(name in artifacts,'Index chunk missing from artifact identity: '+name)
    guard_path=index/'index-stat-guard.json';guard=json.loads(guard_path.read_text()) if guard_path.exists() else None
    manifest_sha=file_sha(index/'index-artifact-identity.json');input_sha=file_sha(index/'input-identity.json');stats=artifact_stats(index,artifacts)
    if guard:
        require(guard['artifact_manifest_sha256']==manifest_sha,'Index artifact manifest changed after freeze')
        require(guard['input_identity_sha256']==input_sha,'Index input identity changed after freeze')
    full=force_full or not guard or guard['stats']!=stats
    if full:
        for name,entry in artifacts.items():
            expected=entry['sha256'] if isinstance(entry,dict) else entry
            if isinstance(entry,dict):require((index/name).stat().st_size==entry['bytes'],'Index chunk size mismatch: '+name)
            require(file_sha(index/name)==expected,'Index artifact content modified: '+name)
        guard=freeze_index_stat_guard(index)
    return {'input':binding,'bits':meta['config']['nbits'],'metadata_sha256':artifacts['metadata.json'],'artifact_manifest_sha256':manifest_sha,'stat_guard_sha256':file_sha(guard_path),'full_payload_verified_this_call':full,'index':str(index)}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--verify-full',action='store_true');a=p.parse_args();print(json.dumps(index_identity(a.index,force_full=a.verify_full),indent=2))
