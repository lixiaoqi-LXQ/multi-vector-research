"""Adversarial acceptance tests; CPU-only, no GPU context."""
import copy,hashlib,json,shutil,tempfile
from pathlib import Path
import numpy as np
from identity import input_identity,array_sha,file_sha
from summarize import validate_result
root=Path(__file__).parent/'acceptance-test';root.mkdir(exist_ok=True)
base=Path(tempfile.mkdtemp(prefix='fixtures-',dir=root));data=base/'data';data.mkdir();nd=120;nq=4
np.save(data/'doclens.npy',np.full(nd,2,dtype=np.int32));np.save(data/'doc_ids.npy',np.arange(nd,dtype=np.int64));np.save(data/'queries_ids.npy',np.arange(nq,dtype=np.int64));np.save(data/'queries.npy',np.zeros((nq,32,128),dtype=np.float32));raw=np.zeros((nd*2,128),dtype=np.float16).view(np.uint16);np.save(data/'documents.npy',raw)
repo='fixture/immutable';rev='0'*40;baseurl=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/'
(data/'download-provenance.json').write_text(json.dumps({'repository':repo,'revision':rev,'files':{'documents.npy':{'url':baseurl+'documents.npy'}}}))
ix=base/'index';ix.mkdir();binding=input_identity(str(data),nd);binding['scanned_prefix_sha256']=hashlib.sha256(raw.tobytes()).hexdigest();(ix/'input-identity.json').write_text(json.dumps(binding));(ix/'metadata.json').write_text(json.dumps({'num_documents':nd,'num_embeddings':nd*2,'num_chunks':1,'config':{'nbits':4}}))
for name in ['centroids.pt','buckets.pt','avg_residual.pt','ivf.pid.pt']:(ix/name).write_bytes(name.encode())
artifacts={name:file_sha(ix/name) for name in ['metadata.json','centroids.pt','buckets.pt','avg_residual.pt','ivf.pid.pt']};
for name in ['0.codes.pt','0.residuals.pt','doclens.0.json']:
    (ix/name).write_bytes(name.encode());artifacts[name]={'bytes':(ix/name).stat().st_size,'sha256':file_sha(ix/name)}
(ix/'index-artifact-identity.json').write_text(json.dumps(artifacts))
gt=base/'gt';gt.mkdir();ids=np.tile(np.arange(100,dtype=np.int64),(nq,1));scores=np.tile(np.arange(100,0,-1,dtype=np.float32),(nq,1));np.save(gt/'ids.npy',ids);np.save(gt/'scores.npy',scores);np.save(gt/'query_ids.npy',np.arange(nq,dtype=np.int64));np.save(gt/'query_external_ids.npy',np.arange(nq,dtype=np.int64))
m=dict(binding);m.update(schema_version=2,complete=True,docs_done=nd,ids_payload_sha256=array_sha(ids),scores_payload_sha256=array_sha(scores),artifact_sha256={n:file_sha(gt/n) for n in ['ids.npy','scores.npy','query_ids.npy','query_external_ids.npy']});(gt/'manifest.json').write_text(json.dumps(m))
expected=base/'rows.npy';np.save(expected,np.arange(nq,dtype=np.int64));held=base/'held.npy';np.save(held,np.array([1,3]))
original=[{'round':r,'query_row':i,'query_id':i,'elapsed_ms':1.0,'ids':ids[i].tolist(),'scores':scores[i].tolist()} for r in range(3) for i in range(nq)]
checks=[]
def case(name,mutate=None,should_pass=False,contract_change=None,run_change=None):
    out=base/name;out.mkdir();rr=copy.deepcopy(original)
    if mutate:mutate(rr)
    (out/'per-query.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rr));(out/'contract.json').write_text(json.dumps({'query_placement':'host','index_placement':'native','profile_range':False,'expected_rounds':3,'expected_query_rows':list(range(nq)),'topk':100,'gt_manifest_sha256':file_sha(gt/'manifest.json'),'index_artifact_manifest_sha256':file_sha(ix/'index-artifact-identity.json')}))
    ri={'query_placement':'host','index_placement':'native','profile_range':False,'payload_devices':{'codes':'cpu','residuals':'cpu','ivf':'cpu'}}
    if run_change:run_change(ri)
    (out/'run-identity.json').write_text(json.dumps(ri))
    if contract_change:
        cc=json.loads((out/'contract.json').read_text());contract_change(cc);(out/'contract.json').write_text(json.dumps(cc))
    try:validate_result(out,gt,expected_rounds=3,expected_query_rows=expected,topk=100,data=data,index=ix,heldout_rows=held)
    except (ValueError,AssertionError) as e:
        if should_pass:raise
        checks.append({'case':name,'rejected':True,'reason':str(e)});return
    assert should_pass,name+' was incorrectly accepted';checks.append({'case':name,'accepted':True})
case('complete',should_pass=True)
case('missing_round',lambda r:r.__delitem__(slice(8,None)))
case('truncated_row',lambda r:r.pop())
case('duplicate_query',lambda r:r[1].update(query_id=0))
case('wrong_order',lambda r:r.__setitem__(slice(0,2),r[1::-1]))
case('insufficient_k',lambda r:r[0].update(ids=r[0]['ids'][:-1],scores=r[0]['scores'][:-1]))
case('duplicate_id',lambda r:r[0]['ids'].__setitem__(1,0))
case('id_out_of_bounds',lambda r:r[0]['ids'].__setitem__(0,nd))
case('nonfinite_score',lambda r:r[0]['scores'].__setitem__(0,float('nan')))
case('nonfinite_elapsed',lambda r:r[0].update(elapsed_ms=float('inf')))
case('wrong_external_qid',lambda r:r[0].update(query_id=999))
case('profile_rejected',contract_change=lambda c:c.update(profile_range=True))
case('device_query_rejected',contract_change=lambda c:c.update(query_placement='device'))
case('mislabeled_payload_devices',run_change=lambda r:r['payload_devices'].update(codes='cuda:0'))
# Bindings must fail independently of row correctness.
for label,mutator in [
    ('wrong_gt_query_identity',lambda m:m.update(query_sha256='0'*64)),
    ('wrong_gt_corpus_identity',lambda m:m.update(corpus_source_id='0'*64)),
]:
    badgt=base/label;shutil.copytree(gt,badgt);bm=json.loads((badgt/'manifest.json').read_text());mutator(bm);(badgt/'manifest.json').write_text(json.dumps(bm))
    try:validate_result(base/'complete',badgt,expected_rounds=3,expected_query_rows=expected,topk=100,data=data,index=ix,heldout_rows=held)
    except ValueError as e:checks.append({'case':label,'rejected':True,'reason':str(e)})
    else:raise AssertionError(label+' accepted')
badix=base/'tampered-index';shutil.copytree(ix,badix);(badix/'centroids.pt').write_bytes(b'modified')
try:validate_result(base/'complete',gt,expected_rounds=3,expected_query_rows=expected,topk=100,data=data,index=badix,heldout_rows=held)
except ValueError as e:checks.append({'case':'tampered_index','rejected':True,'reason':str(e)})
else:raise AssertionError('tampered index accepted')
(root/'latest-tests.json').write_text(json.dumps({'checks':checks,'fixture':str(base)},indent=2));print(json.dumps(checks,indent=2))
