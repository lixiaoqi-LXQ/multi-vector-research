"""CPU-only generation of explicit 100k/full build and staged calibration specs."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=Path('/home/lxq/graduate/data/multivector-colbertv2')
PLACES=','.join('{'+str(i)+'}' for i in [7,5,6,8,9,15,16,17,18,19,0,1,2,3,4,10,11,12,13,14])
SOURCE=dict(repository='tuskanny/lotte_pooled_colbertv2',revision='e3745bd22a3be48d167c438fc8a7a839b86860b1',documents_url='https://huggingface.co/datasets/tuskanny/lotte_pooled_colbertv2/resolve/e3745bd22a3be48d167c438fc8a7a839b86860b1/documents.npy')

def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'plans'));p.add_argument('--full-gt',default=str(DATA/'groundtruth/lotte-pooled-dev-full'));a=p.parse_args();out=Path(a.out);entries=[]
 datasets=[('lotte-prefix-100000',100000,10353669,[65536],DATA/'groundtruth/lotte-prefix-100000'),('lotte-pooled-dev-full',2428854,266205513,[2097152,1048576],Path(a.full_gt))]
 for dataset,docs,tokens,clusters,gt in datasets:
  for nc in clusters:
   for ex in (3,7):
    bits=ex+1;name=f'{dataset}-c{nc}-b{bits}';index=DATA/'indexes/chimera'/name;folder=out/name
    build=dict(dataset=dataset,docs=docs,tokens=tokens,input=str(DATA/'chimera-inputs'/dataset),index=str(index),n_clusters=nc,ex_bits=ex,threads=20,cpu_affinity='0-19',omp_places=PLACES,omp_proc_bind='true',source_identity=SOURCE,timeout_s=21600,choice_note='Author README recommends about N/200 to N/100; exact paper cluster count unavailable. Full c2^21 primary; c2^20 optional exploration, not an additional required build.')
    write(folder/'build.json',build)
    write(folder/'build-fp16.json',dict(build,input_mode='fp16_npy',fp16_npy=str(DATA/'lotte-pooled-dev/documents.npy')))
    expected=dict(N=tokens,D=128,paddedD=128,n_clusters=nc,ex_bits=ex,total_bits=bits)
    for k in (10,100):
     ladder=([(64,3000,100),(128,10000,300),(256,20000,1000),(512,40000,3000)] if k==10 else [(64,3000,300),(128,10000,1000),(256,20000,3000),(512,40000,10000)])
     points=[dict(nprobe=n,k_refine=min(docs,r),k_full_bit=min(docs,f),num_chunks=1,cagra_itopk_size=min(n+32,544)) for n,r,f in ladder]
     common=dict(dataset=dataset,phase='calibration',index=str(index),query=str(DATA/'splits/lotte-calibration-seed42/queries.bin'),query_rows=str(DATA/'splits/lotte-calibration-seed42/calibration-rows.npy'),split_manifest=str(DATA/'splits/lotte-calibration-seed42/manifest.json'),gt=str(gt/'calibration/groundtruth.tsv'),gt_ids=str(gt/'calibration/ids.npy'),gt_manifest=str(gt/'manifest.json'),full_gt_ids=str(gt/'ids.npy'),full_query=str(DATA/'chimera-inputs'/dataset/'queries.bin'),full_gt=str(gt/'groundtruth.tsv'),heldout_rows_source=str(DATA/'splits/lotte-calibration-seed42/heldout-rows.npy'),nq=100,k=k,rounds=1,warmup=5,threads=20,cpu_affinity='0-19',omp_places=PLACES,omp_proc_bind='true',expected_queries=2931,calibration_queries=100,require_identity=True,require_build_identity=True,expected_index=expected,timeout_per_point_s=7200,query_input_placement='host FP32, upload included',goals=([.90,.95] if k==10 else [.90,.95,.99]),selection_policy='Use calibration rows only. Freeze fastest measured eligible point before one heldout evaluation. Failed heldout targets stay failed; do not retune on heldout.')
     # Stage 0 is a finite quality ladder, not the full product of all parameters.
     write(folder/f'k{k}-stage0.json',dict(common,points=points))
     # Atomic stages allow stopping/switching bitwidth without rerunning accepted points.
     for i,point in enumerate(points):write(folder/f'k{k}-ladder{i}.json',dict(common,points=[point]))
    entries.append(dict(name=name,primary=nc==clusters[0],build=str(folder/'build.json'),k10_calibration=str(folder/'k10-stage0.json'),k100_calibration=str(folder/'k100-stage0.json')))
 write(out/'plan-index.json',dict(execution='prepared only; no GPU work launched',CPU_policy='Paper uses32 CPU threads with B1. GB10 main results use all20 physical cores; core7/4fast are separate controls, not paper defaults.',source_identity=SOURCE,entries=entries,order=['100k 4/8bit build and independent k10/k100 calibration','full LoTTE primary c2^21 4/8bit build','calibration-only local coordinate refinement if needed','freeze one point per bitwidth/K/target; all2931x3 QPS plus heldout2831 quality','one matched chunk c1/cbest comparison and CPU organization comparison','32 or100-query Nsight diagnostic at final parameters; no profiling QPS claim'],not_required='Full Cartesian parameter grids, speculative algorithm rewrites, and all paper kernel ablations are not automatic work.'))
 print(out/'plan-index.json')
if __name__=='__main__':main()
