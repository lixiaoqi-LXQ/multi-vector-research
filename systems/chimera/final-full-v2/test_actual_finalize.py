"""CPU-only finalization using already accepted real owned-device 10k artifacts."""
from pathlib import Path
import plan,json,copy
P=Path(__file__).resolve().parent;F=P/'actual-prefix-finalize-fixture';F.mkdir(exist_ok=False);D=plan.DATA;idx=D/'indexes/chimera/lotte-prefix-10000-c8192-b4-owned-device-qualification';s=plan.read(P/'templates-v3/lotte-b4-k100-main/screen.template.json');s.update(index=str(idx),expected_docs=10000,expected_index=dict(N=1087049,D=128,paddedD=128,n_clusters=8192,ex_bits=3,total_bits=4),gt=str(D/'groundtruth/lotte-prefix-10000/calibration/groundtruth.tsv'),gt_ids=str(D/'groundtruth/lotte-prefix-10000/calibration/ids.npy'),gt_manifest=str(D/'groundtruth/lotte-prefix-10000/manifest.json'),full_gt_ids=str(D/'groundtruth/lotte-prefix-10000/ids.npy'),full_gt=str(D/'groundtruth/lotte-prefix-10000/groundtruth.tsv'),full_query=str(D/'chimera-inputs/lotte-prefix-10000/queries.bin'),fixture_only='already root-accepted real 10k qualification; not a full index acceptance');plan.write(F/'template.json',s);side=plan.read(idx/'ROUTING_DATASET_AUDIT.json');bm=plan.read(idx/'build-provenance.json');files=bm['index_identity']['files'];a={'root_accepted':True,'fixture_only':True,'acceptance_basis':'prior actual root accepted owned-device 10k qualification; no full index approval','index':str(idx),'payload_sha256':{n:v['sha256'] for n,v in files.items()},'routing_sidecar_sha256':plan.sha(idx/'ROUTING_DATASET_AUDIT.json'),'routing_dataset_audit_receipt':side['routing_dataset_audit_receipt'],'routing_dataset_audit_sha256':side['routing_dataset_audit_sha256'],'build_provenance_sha256':plan.sha(idx/'build-provenance.json')};plan.write(F/'accepted-prefix-fixture.json',a)
plan.finalize(F/'template.json',F/'accepted-prefix-fixture.json',F/'finalized-first.json')
old=plan.sha;calls=[]
def guarded(path):
 path=Path(path)
 if path.parent==idx and path.name in files:raise AssertionError('Repeated six-payload hashing: '+str(path))
 calls.append(str(path));return old(path)
plan.sha=guarded;plan.finalize(F/'template.json',F/'accepted-prefix-fixture.json',F/'finalized-reused.json');plan.sha=old;assert plan.read(F/'finalized-first.json')==plan.read(F/'finalized-reused.json')
# A mutated stat witness must reject even though expected root SHA is unchanged.
w=Path(str(F/'accepted-prefix-fixture.json')+'.verified-index.json');raw=w.read_text();m=json.loads(raw);m['payload_stat']['doc_full.bin']['ctime_ns']+=1;w.write_text(json.dumps(m))
try:
 try:plan.finalize(F/'template.json',F/'accepted-prefix-fixture.json',F/'invalid.json')
 except ValueError:pass
 else:raise AssertionError('Changed witness accepted')
finally:w.write_text(raw)
plan.write(P/'actual-finalize-validation.json',{'passed':True,'GPU_used':False,'actual_owned_prefix_index':str(idx),'first_all6_hashes_match_accepted_payloads':True,'repeated_finalize_payload_hash_calls':0,'first_and_reused_specs_exact':True,'stat_witness_change_rejected':True,'query_GT_projection_identity_checked':True,'fixture_only_not_full_acceptance':True,'outputs':str(F)});print('Real prefix finalize + no-rehash reuse + tamper rejection PASS')
