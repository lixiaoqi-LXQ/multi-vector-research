from pathlib import Path
r=Path(__file__).resolve().parent/'source/cpp';p=r/'src/kernels.cu';s=p.read_text()
def extract(name):
 pos=s.index('__global__ void '+name);brace=s.index('{',pos);end=brace+1;level=1
 while level:
  level+=(s[end]=='{')-(s[end]=='}');end+=1
 return s[pos:end]
full=extract('partial_score_kernel');start=full.index('        float ip = 0.0f;');end=full.index('        const float dist =',start);ip=full[start:end]
direct=full.replace('partial_score_kernel','partial_score_direct_kernel',1).replace('d_lut','d_queries');direct=direct.replace('    __shared__ float smem_lut[LUT_ENTRIES_PER_QUERY];\n','');a=direct.index('    const float* lut_ptr =');b=direct.index('    const uint32_t* my_labels',a);direct=direct[:a]+direct[b:];direct=direct.replace(ip,'        const uint32_t words[4] = {code128.x,code128.y,code128.z,code128.w};\n        const float ip = direct_binary_ip(d_queries + query_idx * PADDED_DIM, words);\n')
def materialized(f,name):
 f=f.replace(f.split('void ')[1].split('(')[0],name,1);f=f.replace('    size_t          n_clusters\n)', '    size_t          n_clusters,\n    const size_t* __restrict__ query_offsets,\n    float* __restrict__ token_scores,\n    int* __restrict__ token_doc_ids\n)');a=f.index('        if (dist > 0.0f');b=f.rfind('\n    }\n}');return f[:a]+'''        const size_t output = query_offsets[query_idx] + static_cast<size_t>(flat_idx);
        token_scores[output] = dist;
        token_doc_ids[output] = doc_id;
'''+f[b:]
matlut=materialized(full,'partial_score_materialize_lut_kernel');matdirect=materialized(direct,'partial_score_materialize_direct_kernel')
extra='''__global__ void retrieved_token_counts_kernel(const uint32_t* labels,const size_t* cluster_pos,
    size_t* counts,int nprobe,size_t nclusters) {
    const int q=threadIdx.x;
    if(q>=Q_DOCLEN)return;
    size_t sum=0;
    for(int p=0;p<nprobe;++p) {const uint32_t c=labels[q*nprobe+p];if(c<nclusters)sum+=cluster_pos[c+1]-cluster_pos[c];}
    counts[q]=sum;
}

__global__ void aggregate_materialized_partial_kernel(const float* scores,const int* docs,
    const size_t* offsets,float* partial,const candidate_bitmap_bucket_t* bitmap,
    const candidate_bitmap_offset_t* bitmap_offsets,int ncandidates,size_t ndocs) {
    const int q=blockIdx.y;
    const size_t count=offsets[q+1]-offsets[q];
    for(size_t i=threadIdx.x+static_cast<size_t>(blockIdx.x)*blockDim.x;i<count;
        i+=static_cast<size_t>(gridDim.x)*blockDim.x) {
        const size_t row=offsets[q]+i;const int doc=docs[row];const float score=scores[row];
        if(score>0.0f && static_cast<size_t>(doc)<ndocs) {
            const int dense=candidate_dense_index(bitmap,bitmap_offsets,doc);
            if(dense>=0 && dense<ncandidates) atomicMaxFloat(partial+static_cast<size_t>(dense)*Q_DOCLEN+q,score);
        }
    }
}

__global__ void one_bit_token_score_direct_kernel(const float* queries,const char* codes,
    const float* factors,const float* cb1,const uint32_t* positions,float* scores,
    size_t total_tokens,size_t batch_tokens) {
    for(size_t i=threadIdx.x+static_cast<size_t>(blockIdx.x)*blockDim.x;i<batch_tokens;
        i+=static_cast<size_t>(gridDim.x)*blockDim.x) {
        const uint32_t pos=positions[i];
        const uint4 code=*reinterpret_cast<const uint4*>(codes+static_cast<size_t>(pos)*CODE_BYTES);
        const uint32_t words[4]={code.x,code.y,code.z,code.w};const float factor=factors[pos];
        for(int q=0;q<Q_DOCLEN;++q) {
            const float ip=direct_binary_ip(queries+q*PADDED_DIM,words);
            scores[static_cast<size_t>(q)*total_tokens+i]=(ip-cb1[q])*factor;
        }
    }
}

__global__ void one_bit_document_score_naive_kernel(const float* scores,const size_t* offsets,
    float* document_scores,size_t total_tokens,size_t ndocs) {
    const size_t doc=threadIdx.x+static_cast<size_t>(blockIdx.x)*blockDim.x;
    if(doc>=ndocs)return;
    const size_t begin=offsets[doc],end=offsets[doc+1];
    float maxima[Q_DOCLEN];
    #pragma unroll
    for(int q=0;q<Q_DOCLEN;++q) {
        float value=-FLT_MAX;
        for(size_t token=begin;token<end;++token)
            value=fmaxf(value,scores[static_cast<size_t>(q)*total_tokens+token]);
        maxima[q]=value;
    }
    // Preserve the full kernel's32-query binary reduction tree, in one thread.
    #pragma unroll
    for(int stride=Q_DOCLEN/2;stride>0;stride>>=1) {
        #pragma unroll
        for(int q=0;q<stride;++q)maxima[q]+=maxima[q+stride];
    }
    document_scores[doc]=maxima[0];
}
'''
allfns=[direct,matlut,matdirect,extra];code='\n\n'.join(allfns);(r/'src/kernel_ablation.cuh').write_text(code)
protos=[]
# Parse exact global declarations, including the four additional kernels.
for text in allfns:
 cursor=0
 while True:
  start=text.find('__global__ void ',cursor)
  if start<0:break
  brace=text.index('{',start);protos.append(text[start:brace].rstrip()+';');depth=1;cursor=brace+1
  while depth:
   depth+=(text[cursor]=='{')-(text[cursor]=='}');cursor+=1
p.write_text(s.replace('#include <cfloat>','#include <cfloat>\n#include "chimera/binary_ip_reference.hpp"').replace('}  // namespace Chimera','#include "kernel_ablation.cuh"\n\n}  // namespace Chimera'))
p=r/'include/chimera/kernels.cuh';s=p.read_text();p.write_text(s.replace('}  // namespace Chimera','\n'+ '\n\n'.join(protos)+'\n\n}  // namespace Chimera'))
