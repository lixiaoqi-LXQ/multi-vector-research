"""Official PLAID codecs/scorer adapter for preencoded ragged ColBERT vectors."""
import argparse, json, math, os, time, random, hashlib, platform, sys, resource
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import faiss
from colbert.infra import ColBERTConfig
from colbert.indexing.codecs.residual import ResidualCodec
from colbert.indexing.collection_indexer import CollectionIndexer
from colbert.indexing.utils import optimize_ivf
from colbert.search.index_storage import IndexScorer
from colbert.modeling.colbert import ColBERT
from identity import input_identity,gt_identity,index_identity,file_sha

def prefix(path, rows):
    with open(path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        offset = f.tell()
    assert not fortran
    assert os.path.getsize(path) >= offset + rows * np.prod(shape[1:]) * dtype.itemsize
    a = np.memmap(path, mode='r', offset=offset, dtype=dtype, shape=(rows, *shape[1:]))
    return a.view(np.float16) if dtype == np.uint16 else a

def sync(gpu):
    if gpu: torch.cuda.synchronize()

def build(a):
    out = Path(a.index)
    out.mkdir(parents=True, exist_ok=False)
    lens = np.load(Path(a.data)/'doclens.npy')[:a.docs]
    nv = int(lens.sum()); vectors = prefix(a.documents, nv)
    (out/'build-adapter-source.py').write_bytes(Path(__file__).read_bytes())
    source_root=Path(__file__).parent/'ColBERT-main'
    (out/'build-source-identity.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*source_root.glob('colbert/indexing/**/*.py'),*source_root.glob('colbert/indexing/**/*.cu'),*source_root.glob('colbert/indexing/**/*.cpp')] if p.is_file()},indent=2))
    gpu=a.build_device=='cuda'
    (out/'build-environment.json').write_text(json.dumps({'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'faiss':faiss.__version__,'cpu_affinity':sorted(os.sched_getaffinity(0)),'threads':torch.get_num_threads(),'args':vars(a)},indent=2))
    config = ColBERTConfig(dim=vectors.shape[1], nbits=a.bits, gpus=1 if gpu else 0, kmeans_niters=a.kmeans_iters)
    config.total_visible_gpus=1 if gpu else 0
    torch.manual_seed(123); rng = np.random.default_rng(123)
    partitions = a.partitions or int(2**np.floor(np.log2(16*np.sqrt(nv))))
    if a.sampling=='official-docs':
        random.seed(123)
        sampled_docs=sorted(random.sample(range(a.docs),min(1+int(16*np.sqrt(120*a.docs)),a.docs)))
        offsets=np.r_[0,np.cumsum(lens)]
        sample=np.concatenate([np.asarray(vectors[offsets[i]:offsets[i+1]]) for i in sampled_docs])
        sample=torch.from_numpy(sample.copy()).half()
        sample=sample[torch.randperm(len(sample))]
        if a.train_vectors: sample=sample[:a.train_vectors]
    else:
        sample_count=min(nv,a.train_vectors or 100*partitions)
        sample=torch.from_numpy(np.array(vectors[rng.choice(nv,sample_count,replace=False)],dtype=np.float16))
    sample_count=len(sample)
    heldout_count=min(50000,max(1,sample_count//20))
    train,heldout=sample[:-heldout_count],sample[-heldout_count:]
    start=time.perf_counter()
    if a.centroids_from:
        centroids=torch.load(a.centroids_from,map_location='cpu').float()
        assert centroids.shape==(partitions,config.dim)
    else:
        km=faiss.Kmeans(config.dim, partitions, niter=a.kmeans_iters, gpu=False, verbose=True, seed=123)
        if a.kmeans_device=='cuda':
            gpu_resources=faiss.StandardGpuResources()
            gpu_resources.setTempMemory(a.faiss_temp_mib*1024*1024)
            km.index=faiss.index_cpu_to_gpu(gpu_resources,0,km.index)
        km.train(train.float().numpy())
        centroids=torch.from_numpy(km.centroids.copy())
        del km
        if a.kmeans_device=='cuda': del gpu_resources
    if not a.centroids_from: centroids=torch.nn.functional.normalize(centroids, dim=-1)
    # Invoke official residual quantile training method directly.
    trainer=object.__new__(CollectionIndexer); trainer.config=config; trainer.use_gpu=gpu; trainer.verbose=0
    cutoffs, weights, avg = trainer._compute_avg_residual(centroids.half() if gpu else centroids, heldout.to('cuda').half() if gpu else heldout.float())
    codec=ResidualCodec(config, centroids, avg, cutoffs, weights)
    codec.save(str(out))
    del sample, train, heldout, trainer
    if gpu: torch.cuda.empty_cache()
    codes=[];corpus_hash=hashlib.sha256();artifacts={}
    for chunk, begin in enumerate(range(0,a.docs,a.chunk_docs)):
        end=min(a.docs,begin+a.chunk_docs)
        vb=int(lens[:begin].sum()); ve=vb+int(lens[begin:end].sum())
        raw=np.array(vectors[vb:ve],dtype=np.float16)
        corpus_hash.update(memoryview(raw).cast('B'))
        compressed=codec.compress(torch.from_numpy(raw if gpu else raw.astype(np.float32)))
        compressed.save(str(out/chunk.__str__()))
        for suffix in ['codes.pt','residuals.pt']:
            path=out/f'{chunk}.{suffix}';artifacts[path.name]={'bytes':path.stat().st_size,'sha256':file_sha(path)}
        codes.append(compressed.codes)
        (out/f'doclens.{chunk}.json').write_text(json.dumps(lens[begin:end].tolist()))
        print('compressed_chunk',chunk,end,flush=True)
    allcodes=torch.cat(codes)
    assert allcodes.numel()==nv
    assert int(allcodes.min())>=0 and int(allcodes.max())<partitions
    sorter=torch.argsort(allcodes)
    ivflens=torch.bincount(allcodes.long(),minlength=partitions)
    optimize_ivf(sorter,ivflens,str(out),verbose=0)
    metadata={'config':config.export(),'num_chunks':len(codes),'num_partitions':partitions,'num_embeddings':nv,'num_documents':a.docs,'avg_doclen':float(lens.mean())}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    binding=dict(input_identity(a.data,a.docs));binding['scanned_prefix_sha256']=corpus_hash.hexdigest()
    (out/'input-identity.json').write_text(json.dumps(binding,indent=2))
    for name in ['centroids.pt','buckets.pt','avg_residual.pt','metadata.json','ivf.pid.pt']:
        artifacts[name]=file_sha(out/name)
    (out/'index-artifact-identity.json').write_text(json.dumps(artifacts,indent=2))
    (out/'build-provenance.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'torch_max_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if gpu else 0,'docs':a.docs,'vectors':nv,'train_vectors':sample_count-heldout_count,'heldout_vectors':heldout_count,'seed':123,'kmeans_iters':a.kmeans_iters,'partitions':partitions,'nbits':a.bits,'source_documents':a.documents,'device':a.build_device,'kmeans_device':a.kmeans_device,'sampling':a.sampling,'centroids_from':a.centroids_from,'faiss_temp_mib':a.faiss_temp_mib,'official_search':'IndexScorer.rank','adapter':f'preencoded; FAISS {a.kmeans_device} kmeans; official residual training/compression/IVF'},indent=2))

def load_scorer(a):
    gpu=a.device=='cuda'
    runtime_config=ColBERTConfig.load_from_index(a.index)
    runtime_config.configure(gpus=1 if gpu else 0);runtime_config.total_visible_gpus=1 if gpu else 0
    with patch.object(ColBERTConfig,'load_from_index',return_value=runtime_config):
        return IndexScorer(a.index,use_gpu=gpu)

def search(a,scorer=None):
    gpu=a.device=='cuda'
    metadata=json.loads((Path(a.index)/'metadata.json').read_text())
    scorer=load_scorer(a) if scorer is None else scorer
    if not gpu: ColBERT.try_load_torch_extensions(False)
    config=ColBERTConfig(ncells=a.ncells, centroid_score_threshold=a.threshold, ndocs=a.ndocs,query_maxlen=32)
    query_data=np.load(Path(a.data)/'queries.npy',mmap_mode='r')
    ids=np.load(Path(a.data)/'queries_ids.npy')
    a.docs_limit=sum(len(json.loads(path.read_text())) for path in Path(a.index).glob('doclens.*.json'))
    rows=np.load(a.query_rows).astype(np.int64) if a.query_rows else np.arange(len(ids))
    assert len(np.unique(rows))==len(rows) and np.all((rows>=0)&(rows<len(ids)))
    n=min(a.queries,len(rows)); rows=rows[:n]
    queries=torch.from_numpy(np.array(query_data[rows])).float()
    if gpu and a.query_placement=='device':queries=queries.cuda().half()
    def rank_one(i):
        q=queries[i:i+1]
        if gpu and a.query_placement=='host':q=q.cuda().half()
        return scorer.rank(config,q)
    output=Path(a.output); output.mkdir(parents=True,exist_ok=True)
    assert not (output/'per-query.jsonl').exists(), 'Refusing to overwrite results'
    (output/'runner-loaded-source.py').write_bytes(Path(__file__).read_bytes())
    source_root=Path(__file__).parent/'ColBERT-main'
    identity={'python':sys.version,'torch':torch.__version__,'faiss':faiss.__version__,'machine':platform.machine(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'torch_threads':torch.get_num_threads(),'timer':'perf_counter_ns','timing_boundary':'IndexScorer.rank including returned Python IDs/scores; per-query synchronize CUDA; query vectors preloaded; file writes excluded','source_sha256':{},'extension_sha256':{}}
    for path in [Path(__file__),*source_root.glob('colbert/search/*.py'),*source_root.glob('colbert/search/*.cpp'),*source_root.glob('colbert/indexing/codecs/*')]:
        if path.is_file(): identity['source_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in Path(os.environ.get('TORCH_EXTENSIONS_DIR',str(Path(__file__).parent/'torch-extensions'))).glob('**/*.so'):
        identity['extension_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    identity['query_placement']=a.query_placement
    identity['timing_boundary']='host FP32 preencoded query to host IDs/scores, including query H2D/half conversion' if a.query_placement=='host' else 'diagnostic: preloaded device FP16 query to host IDs/scores'
    (output/'run-identity.json').write_text(json.dumps(identity,indent=2))
    ix=index_identity(str(a.index));gt=gt_identity(str(a.groundtruth),str(a.data),metadata['num_documents'])
    (output/'contract.json').write_text(json.dumps({'expected_query_rows':rows.tolist(),'expected_rounds':a.rounds,'topk':a.topk,'query_placement':a.query_placement,'gt_manifest_sha256':gt['gt_manifest_sha256'],'index_artifact_manifest_sha256':ix['artifact_manifest_sha256']},indent=2))
    with torch.inference_mode():
        for i in range(a.warmup): rank_one(i%n)
        sync(gpu)
        with open(output/'per-query.jsonl','w') as f:
            for r in range(a.rounds):
                for i in range(n):
                    sync(gpu); start=time.perf_counter_ns()
                    pids,scores=rank_one(i); sync(gpu)
                    elapsed=(time.perf_counter_ns()-start)/1e6
                    assert len(pids)==len(scores) and len(set(pids))==len(pids)
                    assert all(0<=p<a.docs_limit for p in pids)
                    assert all(math.isfinite(s) for s in scores)
                    assert all(scores[j] >= scores[j+1] for j in range(len(scores)-1))
                    f.write(json.dumps({'round':r,'query_row':int(rows[i]),'query_id':int(ids[rows[i]]),'elapsed_ms':elapsed,'ids':pids[:a.topk],'scores':scores[:a.topk]})+'\n')
                f.flush()
    (output/'config.json').write_text(json.dumps(vars(a),indent=2))

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build'); b.add_argument('--documents',required=True); b.add_argument('--docs',type=int,default=10000); b.add_argument('--bits',type=int,default=4); b.add_argument('--partitions',type=int,default=0); b.add_argument('--kmeans-iters',type=int,default=20); b.add_argument('--train-vectors',type=int,default=0); b.add_argument('--centroids-from'); b.add_argument('--sampling',choices=['official-docs','vectors'],default='official-docs'); b.add_argument('--build-device',choices=['cpu','cuda'],default='cpu'); b.add_argument('--kmeans-device',choices=['cpu','cuda'],default='cpu'); b.add_argument('--chunk-docs',type=int,default=10000); b.add_argument('--faiss-temp-mib',type=int,default=512)
    s=sub.add_parser('search'); s.add_argument('--device',choices=['cpu','cuda'],default='cpu'); s.add_argument('--queries',type=int,default=100); s.add_argument('--rounds',type=int,default=3); s.add_argument('--warmup',type=int,default=10); s.add_argument('--ncells',type=int,default=4); s.add_argument('--threshold',type=float,default=.45); s.add_argument('--ndocs',type=int,default=1024); s.add_argument('--topk',type=int,default=100); s.add_argument('--output',required=True); s.add_argument('--query-rows'); s.add_argument('--query-placement',choices=['host','device'],default='host'); s.add_argument('--groundtruth',required=True)
    for parser in [b,s]:
        parser.add_argument('--data',required=True); parser.add_argument('--index',required=True); parser.add_argument('--threads',type=int,default=8)
    a=p.parse_args(); torch.set_num_threads(a.threads); faiss.omp_set_num_threads(a.threads)
    (build if a.command=='build' else search)(a)

if __name__=='__main__':main()
