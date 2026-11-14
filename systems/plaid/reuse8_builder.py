"""Official8bit residual encoding reusing verified4bit nearest-centroid assignments."""
import argparse,hashlib,json,os,shutil,sys,time,resource
from pathlib import Path
import numpy as np
import torch
from colbert.infra import ColBERTConfig
from colbert.indexing.codecs.residual import ResidualCodec
from colbert.indexing.collection_indexer import CollectionIndexer
from identity import index_identity,input_identity,gt_identity,file_sha,freeze_index_stat_guard
from seeded_search_runner import prefix,thread_affinity_snapshot
from reuse8_support import official_heldout_global_rows,compress_with_existing_codes

def build(a):
    assert os.environ.get('UMA_RUN_GPU_TESTS')=='1','GPU execution requires explicitUMA_RUN_GPU_TESTS=1'
    source=Path(a.codes_from);out=Path(a.index);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();source_id=index_identity(str(source));meta=json.loads((source/'metadata.json').read_text());provenance=json.loads((source/'build-provenance.json').read_text());build_env=json.loads((source/'build-environment.json').read_text())
    assert meta['config']['nbits']==4 and provenance['device']=='cuda','Reuse requires verifiedGPU4bit assignments'
    assert provenance['sampling']=='official-docs' and not build_env['args'].get('train_vectors',0),'Exact official sample reconstruction required'
    assert build_env['torch']==torch.__version__ and build_env['python']==sys.version,'RNG/runtime version differs from source'
    docs=meta['num_documents'];lens=np.load(Path(a.data)/'doclens.npy')[:docs].astype(np.int64);nv=int(lens.sum());actual=dict(input_identity(a.data,docs));gt=gt_identity(a.groundtruth,a.data,docs)
    for key in ['corpus_source_id','query_sha256','query_ids_sha256','document_ids_sha256','doclens_sha256','docs','num_tokens']:assert actual[key]==source_id['input'][key]==gt[key],key
    assert source_id['input']['scanned_prefix_sha256']==gt['gt_prefix_sha256']
    path=Path(a.data)/'documents.npy';path=path if path.exists() else Path(a.data)/'documents.npy.partial';vectors=prefix(str(path),nv)
    source_centers=torch.load(source/'centroids.pt',map_location='cpu');assert source_centers.dtype==torch.float16 and torch.isfinite(source_centers).all()
    previous_threads=torch.get_num_threads();torch.set_num_threads(int(build_env['threads']));rows,sample=official_heldout_global_rows(lens,seed=123);torch.set_num_threads(a.threads)
    assert sample['sample_vectors']==provenance['train_vectors']+provenance['heldout_vectors'] and sample['heldout_vectors']==provenance['heldout_vectors']
    held=torch.from_numpy(np.array(vectors[rows],dtype=np.float16)).cuda();config=ColBERTConfig(dim=meta['config']['dim'],nbits=8,gpus=1,kmeans_niters=provenance['kmeans_iters']);config.total_visible_gpus=1
    trainer=object.__new__(CollectionIndexer);trainer.config=config;trainer.use_gpu=True;trainer.verbose=0;cutoffs,weights,avg=trainer._compute_avg_residual(source_centers.cuda(),held)
    codec=ResidualCodec(config,source_centers,avg,cutoffs,weights);assert torch.equal(codec.centroids.cpu().view(torch.int16),source_centers.view(torch.int16));codec.save(str(out));del held,trainer;torch.cuda.empty_cache()
    torch.save(torch.from_numpy(rows.copy()),out/'heldout-global-rows.pt');corpus_hash=hashlib.sha256();artifacts={};offset=0;doc_begin=0
    for chunk in range(meta['num_chunks']):
        chunk_lens=json.loads((source/f'doclens.{chunk}.json').read_text());assert np.array_equal(lens[doc_begin:doc_begin+len(chunk_lens)],np.asarray(chunk_lens));count=sum(chunk_lens);codes=torch.load(source/f'{chunk}.codes.pt',map_location='cpu');assert codes.dtype==torch.int32 and len(codes)==count
        raw=np.array(vectors[offset:offset+count],dtype=np.float16);corpus_hash.update(memoryview(raw).cast('B'));compressed=compress_with_existing_codes(codec,torch.from_numpy(raw),codes)
        # Preserve source code container bytes exactly; no reassignment or reserialization.
        shutil.copy2(source/f'{chunk}.codes.pt',out/f'{chunk}.codes.pt');torch.save(compressed.residuals,out/f'{chunk}.residuals.pt');shutil.copy2(source/f'doclens.{chunk}.json',out/f'doclens.{chunk}.json')
        for name in [f'{chunk}.codes.pt',f'{chunk}.residuals.pt',f'doclens.{chunk}.json']:
            f=out/name;artifacts[name]={'bytes':f.stat().st_size,'sha256':file_sha(f)}
        assert artifacts[f'{chunk}.codes.pt']['sha256']==source_id_code_sha(source,chunk)
        offset+=count;doc_begin+=len(chunk_lens);print('reused_codes_chunk',chunk,doc_begin,flush=True)
    assert offset==nv and doc_begin==docs and corpus_hash.hexdigest()==source_id['input']['scanned_prefix_sha256']
    shutil.copy2(source/'ivf.pid.pt',out/'ivf.pid.pt');metadata=dict(meta);metadata['config']=config.export();(out/'metadata.json').write_text(json.dumps(metadata,indent=2));actual['scanned_prefix_sha256']=corpus_hash.hexdigest();(out/'input-identity.json').write_text(json.dumps(actual,indent=2))
    for name in ['centroids.pt','buckets.pt','avg_residual.pt','ivf.pid.pt','metadata.json']:artifacts[name]=file_sha(out/name)
    (out/'index-artifact-identity.json').write_text(json.dumps(artifacts,indent=2));freeze_index_stat_guard(out,verified_at_build=True)
    (out/'build-adapter-source.py').write_bytes(Path(__file__).read_bytes());(out/'reuse-support-source.py').write_bytes((Path(__file__).parent/'reuse8_support.py').read_bytes());(out/'build-thread-affinity.json').write_text(json.dumps(thread_affinity_snapshot(),indent=2))
    (out/'build-environment.json').write_text(json.dumps({'torch':torch.__version__,'python':sys.version,'threads':a.threads,'args':vars(a)},indent=2))
    proof={'seconds':time.perf_counter()-start,'kind':'reuse4bit-codes-and-IVF;8bit residual quantiles and official binarize','fresh_independent8bit_build_time':False,'docs':docs,'vectors':nv,'partitions':meta['num_partitions'],'nbits':8,'kmeans_iters':provenance['kmeans_iters'],'centroids_from':str(source/'centroids.pt'),'codes_from':str(source),'source_index_artifact_manifest_sha256':source_id['artifact_manifest_sha256'],'source_prefix_sha256':corpus_hash.hexdigest(),'centroids_fp16_tensor_sha256':hashlib.sha256(source_centers.numpy().tobytes()).hexdigest(),'same_centroid_tensor_bits':True,'sample':sample,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'cuda_peak_reserved':torch.cuda.max_memory_reserved()}
    (out/'reuse-proof.json').write_text(json.dumps(proof,indent=2));(out/'build-provenance.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)

def source_id_code_sha(source,chunk):
    entry=json.loads((source/'index-artifact-identity.json').read_text())[f'{chunk}.codes.pt'];return entry['sha256'] if isinstance(entry,dict) else entry
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--codes-from',required=True);p.add_argument('--index',required=True);p.add_argument('--data',required=True);p.add_argument('--groundtruth',required=True);p.add_argument('--threads',type=int,default=20);a=p.parse_args();torch.set_num_threads(a.threads);build(a)
