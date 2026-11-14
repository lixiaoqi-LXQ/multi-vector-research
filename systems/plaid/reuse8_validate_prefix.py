"""GPU-only encoding/search equivalence check on one whole original chunk."""
import argparse,json,os,gc,hashlib
from pathlib import Path
import numpy as np
import torch
from colbert.indexing.codecs.residual import ResidualCodec
from colbert.indexing.utils import optimize_ivf
from colbert.infra import ColBERTConfig
import seeded_search_runner as runner
from reuse8_support import compress_with_existing_codes

def validate(source8,source4,data,out,queries=8):
    assert os.environ.get('UMA_RUN_GPU_TESTS')=='1'
    source8,source4,data,out=map(Path,[source8,source4,data,out]);out.mkdir(parents=True,exist_ok=False)
    codec=ResidualCodec.load(str(source8));assert codec.nbits==8
    lengths=json.loads((source4/'doclens.0.json').read_text());nv=sum(lengths);raw=runner.prefix(str(data/'documents.npy'),nv);embs=torch.from_numpy(np.array(raw,dtype=np.float16));oldcodes=torch.load(source4/'0.codes.pt',map_location='cpu')
    fresh=codec.compress(embs);reused=compress_with_existing_codes(codec,embs,oldcodes)
    codes_equal=torch.equal(fresh.codes,oldcodes) and torch.equal(fresh.codes,reused.codes);residuals_equal=torch.equal(fresh.residuals,reused.residuals)
    assert codes_equal and residuals_equal,'Reused8bit encoding differs from default official compress'
    meta=json.loads((source8/'metadata.json').read_text());meta.update(num_chunks=1,num_embeddings=nv,num_documents=len(lengths),avg_doclen=float(np.mean(lengths)))
    encoded={'fresh':fresh,'reused':reused}
    for name,values in encoded.items():
        ix=out/name;ix.mkdir();codec.save(str(ix));values.save(str(ix/'0'));(ix/'doclens.0.json').write_text(json.dumps(lengths));(ix/'metadata.json').write_text(json.dumps(meta))
        ordered=torch.argsort(values.codes);counts=torch.bincount(values.codes.long(),minlength=meta['num_partitions']);optimize_ivf(ordered,counts,str(ix),verbose=0)
    del fresh,reused,encoded,codec;gc.collect();torch.cuda.empty_cache()
    q=torch.from_numpy(np.load(data/'queries.npy')[:queries]).cuda().half();cfg=ColBERTConfig(ncells=min(128,meta['num_partitions']),ndocs=max(8192,len(lengths)*4),centroid_score_threshold=0.,query_maxlen=32);results={};strides={}
    from types import SimpleNamespace
    for name in ['fresh','reused']:
        scorer=runner.load_scorer(SimpleNamespace(index=str(out/name),device='cuda',index_placement='native'))
        strides[name]={'ivf':scorer.ivf.strides,'codes':scorer.embeddings_strided.codes_strided.strides,'residuals':scorer.embeddings_strided.residuals_strided.strides}
        with torch.inference_mode():results[name]=[scorer.rank(cfg,q[i:i+1]) for i in range(queries)]
        del scorer;gc.collect();torch.cuda.empty_cache()
    same_ids=all(x[0]==y[0] for x,y in zip(results['fresh'],results['reused']));same_scores=all(x[1]==y[1] for x,y in zip(results['fresh'],results['reused']));assert same_ids and same_scores
    receipt={'complete':True,'scope':'first whole source chunk; default8bit full centroid argmax+residual encoding versus verified4bit codes+8bit residual encoding; functional search not QPS','source8':str(source8),'source4':str(source4),'docs':len(lengths),'vectors':nv,'queries':queries,'same_codes_bitwise':codes_equal,'same_residuals_bitwise':residuals_equal,'same_all_returned_ids':same_ids,'same_all_returned_scores':same_scores,'same_init_strides':strides['fresh']==strides['reused'],'returned_counts':[len(x[0]) for x in results['fresh']],'ncells':cfg.ncells,'ndocs':cfg.ndocs,'search_init_seed':123}
    (out/'validation.json').write_text(json.dumps(receipt,indent=2));(out/'query-results.json').write_text(json.dumps(results));return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source8',required=True);p.add_argument('--source4',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--queries',type=int,default=8);a=p.parse_args();torch.set_num_threads(20);print(json.dumps(validate(a.source8,a.source4,a.data,a.output,a.queries),indent=2))
