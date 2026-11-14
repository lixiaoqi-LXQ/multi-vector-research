from pathlib import Path
import json,hashlib,shutil,difflib
R=Path(__file__).resolve().parent;BASE=R.parent/'chimera-checkpoint';SRC=BASE/'source';DEST=R/'source'
assert not DEST.exists();shutil.copytree(SRC,DEST,symlinks=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[p.relative_to(SRC) for p in SRC.rglob('*') if p.is_file() and 'third_party' not in p.relative_to(SRC).parts]
before={str(p):sha(SRC/p) for p in files}
p=DEST/'cpp/src/build_index.cu';s=p.read_text();old='''    std::vector<float> centroids(checked_product(n_clusters, dimension, "centroids"));
    for (size_t cluster = 0; cluster < n_clusters; ++cluster) {
        embeddings.copy_to(
            centroids.data() + cluster * dimension,
            sampled_ids[cluster] * dimension, dimension);
    }
''';new='''    std::vector<float> centroids(checked_product(n_clusters, dimension, "centroids"));
    // Preserve reservoir IDs and centroid slots; gather immutable source rows in order.
    std::vector<size_t> read_order(n_clusters);
    std::iota(read_order.begin(), read_order.end(), size_t {0});
    std::sort(read_order.begin(), read_order.end(), [&](size_t a, size_t b) {
        return sampled_ids[a] != sampled_ids[b] ? sampled_ids[a] < sampled_ids[b] : a < b;
    });
    for (size_t cluster : read_order) {
        embeddings.copy_to(
            centroids.data() + cluster * dimension,
            sampled_ids[cluster] * dimension, dimension);
    }
'''
assert s.count(old)==1;p.write_text(s.replace(old,new));after={str(x):sha(DEST/x) for x in files};changed=[x for x in before if before[x]!=after[x]];assert changed==['cpp/src/build_index.cu'],changed
(R/'sorted-gather.patch').write_text(''.join(difflib.unified_diff(s.splitlines(True),p.read_text().splitlines(True),fromfile='a/cpp/src/build_index.cu',tofile='b/cpp/src/build_index.cu')))
(R/'baseline-source-manifest.json').write_text(json.dumps({'source_root':str(SRC),'files':before},indent=2)+'\n')
policy={'candidate':'future-MS-owned-device-checkpoint-source-sorted-gather','enabled_by_default_in_this_private_candidate':True,'original_reservoir_and_rng':'unchanged seed42 mt19937_64 and distribution calls','gather':'sort extra permutation by source row, write original centroid slots','centroid_correctness_requirement':'entire FP32 matrix bitwise identical, not recall approximation','baseline_source':str(SRC),'baseline_builder_sha256':sha(BASE/'build/gpu_build'),'only_changed_engine_source':changed,'other_source_hashes_equal':True,'active_LoTTE_source_build_cache_modified':False,'new_checkpoint_namespace_required':True,'GPU_run_authorized':False}
(R/'candidate-policy.json').write_text(json.dumps(policy,indent=2)+'\n')
(R/'source-manifest.json').write_text(json.dumps({'source_root':str(DEST),'centroid_read_order':'source-sorted with original centroid slots','baseline_source_manifest_sha256':sha(R/'baseline-source-manifest.json'),'candidate_policy_sha256':sha(R/'candidate-policy.json'),'files':after},indent=2)+'\n')
# Supervisors remain byte-identical to the qualified baseline; not an engine variant.
for name in ['run_checkpoint_build.py','supervision.py','exec_guard.c','exec_guard']:
 shutil.copy2(BASE/name,R/name);assert sha(BASE/name)==sha(R/name)
print(json.dumps({'changed_engine_files':changed,'source_manifest_sha256':sha(R/'source-manifest.json')}))
