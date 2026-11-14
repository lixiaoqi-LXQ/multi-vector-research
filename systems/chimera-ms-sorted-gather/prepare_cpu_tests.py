from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
def extract(path):
 s=path.read_text();a=s.index('std::vector<float> sample_centroids(');start=s.index('{',a);depth=1;i=start+1
 while depth:
  depth += (s[i]=='{')-(s[i]=='}');i+=1
 return s[a:i]
records={}
for label,path in [('original',R.parent/'chimera-checkpoint/source/cpp/src/build_index.cu'),('sorted',R/'source/cpp/src/build_index.cu')]:
 body=extract(path);digest=hashlib.sha256(body.encode()).hexdigest();original=body
 # Test-only hooks inject adversarial row IDs and capture actual post-gather IDs.
 needle='    std::vector<float> centroids('
 body=body.replace(needle,'    if (forced_ids) sampled_ids = *forced_ids;\n'+needle,1)
 body=body.replace('    return centroids;','    captured_ids = sampled_ids;\n    return centroids;',1)
 text='namespace '+label+' {\nstatic std::vector<size_t> captured_ids;\nstatic const std::vector<size_t>* forced_ids=nullptr;\n'+body+'\n}\n';(R/(label+'-sample-test.inc')).write_text(text);records[label]={'original_function_sha256':digest,'source_path':str(path),'test_hooks':'inject optional forced IDs before gather; capture sampled_ids after gather; no arithmetic edits'}
(R/'cpu-extraction-provenance.json').write_text(json.dumps(records,indent=2)+'\n')
