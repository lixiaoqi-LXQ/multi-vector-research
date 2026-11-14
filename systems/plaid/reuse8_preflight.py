"""Before full8bit construction, verify official fresh/reused encoding on one source chunk."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
import torch
from colbert.infra import ColBERTConfig
from colbert.indexing.collection_indexer import CollectionIndexer
from colbert.indexing.codecs.residual import ResidualCodec
from identity import index_identity
from seeded_search_runner import prefix
from reuse8_support import official_heldout_global_rows
from reuse8_validate_prefix import validate

def main():
    p=argparse.ArgumentParser()
    for key in ['source4','data','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--threads',type=int,default=20);a=p.parse_args()
    assert os.environ.get('UMA_RUN_GPU_TESTS')=='1'
    source,data,out=map(Path,[a.source4,a.data,a.output]);out.mkdir(parents=True,exist_ok=False)
    identity=index_identity(str(source));meta=json.loads((source/'metadata.json').read_text());provenance=json.loads((source/'build-provenance.json').read_text());environment=json.loads((source/'build-environment.json').read_text())
    assert meta['config']['nbits']==4 and provenance['device']=='cuda' and provenance['sampling']=='official-docs'
    assert environment['torch']==torch.__version__ and environment['python']==sys.version
    lengths=np.load(data/'doclens.npy')[:meta['num_documents']]
    torch.set_num_threads(environment['threads']);rows,sample=official_heldout_global_rows(lengths);torch.set_num_threads(a.threads)
    assert sample['sample_vectors']==provenance['train_vectors']+provenance['heldout_vectors']
    vectors=prefix(str(data/'documents.npy'),int(lengths.sum()));held=torch.from_numpy(np.array(vectors[rows],dtype=np.float16)).cuda()
    centers=torch.load(source/'centroids.pt',map_location='cpu');cfg=ColBERTConfig(dim=128,nbits=8,gpus=1,kmeans_niters=provenance['kmeans_iters']);cfg.total_visible_gpus=1
    trainer=object.__new__(CollectionIndexer);trainer.config=cfg;trainer.use_gpu=True;trainer.verbose=0
    cutoffs,weights,avg=trainer._compute_avg_residual(centers.cuda(),held);codec=ResidualCodec(cfg,centers,avg,cutoffs,weights)
    scratch=out/'codec8';scratch.mkdir();codec.save(str(scratch));meta['config']=cfg.export();(scratch/'metadata.json').write_text(json.dumps(meta))
    del codec,held,trainer;torch.cuda.empty_cache()
    result=validate(scratch,source,data,out/'first-whole-chunk')
    (out/'preflight.json').write_text(json.dumps({'complete':True,'source_index_artifact_manifest_sha256':identity['artifact_manifest_sha256'],'sample':sample,'validation':result},indent=2))
if __name__=='__main__':main()
