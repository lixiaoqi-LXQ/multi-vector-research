__global__ void partial_score_direct_kernel(
    const float*    __restrict__ d_queries,
    const char*     __restrict__ d_clustered_code,
    const float*    __restrict__ d_clustered_factor,
    const float*    __restrict__ d_cb1_sumq,
    const uint32_t* __restrict__ d_cagra_labels,
    const size_t*   __restrict__ d_cluster_pos,
    const int*      __restrict__ d_clustered_doc_ids,
    float*          __restrict__ d_partial_query_scores,
    const candidate_bitmap_bucket_t* __restrict__ d_candidate_bitmap,
    const candidate_bitmap_offset_t* __restrict__ d_candidate_bitmap_offsets,
    int             num_candidate_docs,
    size_t          num_docs,
    int             nprobe,
    size_t          n_clusters
) {
    __shared__ uint32_t smem_cstart[kCandidateRefinementMaxNprobe];
    __shared__ int smem_prefix[kCandidateRefinementMaxNprobe + 1];

    const int query_idx = blockIdx.y;
    if (query_idx >= Q_DOCLEN) {
        return;
    }

    const uint32_t* my_labels = d_cagra_labels + query_idx * nprobe;
    for (int p = threadIdx.x; p < nprobe; p += blockDim.x) {
        const uint32_t label = my_labels[p];
        if (label < static_cast<uint32_t>(n_clusters)) {
            const size_t start = d_cluster_pos[label];
            smem_cstart[p] = static_cast<uint32_t>(start);
            smem_prefix[p + 1] = static_cast<int>(d_cluster_pos[label + 1] - start);
        } else {
            smem_cstart[p] = 0;
            smem_prefix[p + 1] = 0;
        }
    }
    if (threadIdx.x == 0) {
        smem_prefix[0] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i <= nprobe; ++i) {
            smem_prefix[i] += smem_prefix[i - 1];
        }
    }
    __syncthreads();

    const int total_elements = smem_prefix[nprobe];
    const float cb1_sumq = d_cb1_sumq[query_idx];

    for (int flat_idx = threadIdx.x + blockIdx.x * blockDim.x;
         flat_idx < total_elements;
         flat_idx += blockDim.x * gridDim.x) {
        int lo = 0;
        int hi = nprobe;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (smem_prefix[mid + 1] <= flat_idx) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        const int cluster_size = smem_prefix[lo + 1] - smem_prefix[lo];
        const int cluster_local = flat_idx - smem_prefix[lo];
        const uint32_t emb_pos = cluster_virtual_to_token_position(
            smem_cstart[lo], cluster_size, cluster_local);

        const uint4 code128 = *reinterpret_cast<const uint4*>(
            d_clustered_code + static_cast<size_t>(emb_pos) * CODE_BYTES);
        const float factor = d_clustered_factor[emb_pos];
        const int doc_id = d_clustered_doc_ids[emb_pos];

        const uint32_t words[4] = {code128.x,code128.y,code128.z,code128.w};
        const float ip = direct_binary_ip(d_queries + query_idx * PADDED_DIM, words);
        const float dist = (ip - cb1_sumq) * factor;
        if (dist > 0.0f && static_cast<size_t>(doc_id) < num_docs) {
            const unsigned int active_mask = __activemask();
            const unsigned int doc_group = warp_matching_doc_mask(doc_id, active_mask);
            const int lane = threadIdx.x & 31;
            const int leader_lane = __ffs(doc_group) - 1;
            const float group_max = warp_group_max(dist, doc_group, active_mask);

            if (lane == leader_lane) {
                const int candidate_index = candidate_dense_index(
                    d_candidate_bitmap,
                    d_candidate_bitmap_offsets,
                    doc_id);
                if (candidate_index < 0 ||
                    candidate_index >= num_candidate_docs) {
                    continue;
                }
                atomicMaxFloat(
                    &d_partial_query_scores[
                        static_cast<size_t>(candidate_index) * Q_DOCLEN +
                        query_idx],
                    group_max);
            }
        }
    }
}

__global__ void partial_score_materialize_lut_kernel(
    const float*    __restrict__ d_lut,
    const char*     __restrict__ d_clustered_code,
    const float*    __restrict__ d_clustered_factor,
    const float*    __restrict__ d_cb1_sumq,
    const uint32_t* __restrict__ d_cagra_labels,
    const size_t*   __restrict__ d_cluster_pos,
    const int*      __restrict__ d_clustered_doc_ids,
    float*          __restrict__ d_partial_query_scores,
    const candidate_bitmap_bucket_t* __restrict__ d_candidate_bitmap,
    const candidate_bitmap_offset_t* __restrict__ d_candidate_bitmap_offsets,
    int             num_candidate_docs,
    size_t          num_docs,
    int             nprobe,
    size_t          n_clusters,
    const size_t* __restrict__ query_offsets,
    float* __restrict__ token_scores,
    int* __restrict__ token_doc_ids
) {
    __shared__ float smem_lut[LUT_ENTRIES_PER_QUERY];
    __shared__ uint32_t smem_cstart[kCandidateRefinementMaxNprobe];
    __shared__ int smem_prefix[kCandidateRefinementMaxNprobe + 1];

    const int query_idx = blockIdx.y;
    if (query_idx >= Q_DOCLEN) {
        return;
    }

    const float* lut_ptr = d_lut + query_idx * LUT_ENTRIES_PER_QUERY;
    for (int i = threadIdx.x; i < LUT_ENTRIES_PER_QUERY; i += blockDim.x) {
        smem_lut[i] = lut_ptr[i];
    }

    const uint32_t* my_labels = d_cagra_labels + query_idx * nprobe;
    for (int p = threadIdx.x; p < nprobe; p += blockDim.x) {
        const uint32_t label = my_labels[p];
        if (label < static_cast<uint32_t>(n_clusters)) {
            const size_t start = d_cluster_pos[label];
            smem_cstart[p] = static_cast<uint32_t>(start);
            smem_prefix[p + 1] = static_cast<int>(d_cluster_pos[label + 1] - start);
        } else {
            smem_cstart[p] = 0;
            smem_prefix[p + 1] = 0;
        }
    }
    if (threadIdx.x == 0) {
        smem_prefix[0] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i <= nprobe; ++i) {
            smem_prefix[i] += smem_prefix[i - 1];
        }
    }
    __syncthreads();

    const int total_elements = smem_prefix[nprobe];
    const float cb1_sumq = d_cb1_sumq[query_idx];

    for (int flat_idx = threadIdx.x + blockIdx.x * blockDim.x;
         flat_idx < total_elements;
         flat_idx += blockDim.x * gridDim.x) {
        int lo = 0;
        int hi = nprobe;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (smem_prefix[mid + 1] <= flat_idx) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        const int cluster_size = smem_prefix[lo + 1] - smem_prefix[lo];
        const int cluster_local = flat_idx - smem_prefix[lo];
        const uint32_t emb_pos = cluster_virtual_to_token_position(
            smem_cstart[lo], cluster_size, cluster_local);

        const uint4 code128 = *reinterpret_cast<const uint4*>(
            d_clustered_code + static_cast<size_t>(emb_pos) * CODE_BYTES);
        const float factor = d_clustered_factor[emb_pos];
        const int doc_id = d_clustered_doc_ids[emb_pos];

        float ip = 0.0f;
        #pragma unroll
        for (int n = 0; n < 8; ++n)
            ip += smem_lut[n * LUT_SIZE + ((code128.x >> (n * 4)) & 0xF)];
        #pragma unroll
        for (int n = 0; n < 8; ++n)
            ip += smem_lut[(8 + n) * LUT_SIZE + ((code128.y >> (n * 4)) & 0xF)];
        #pragma unroll
        for (int n = 0; n < 8; ++n)
            ip += smem_lut[(16 + n) * LUT_SIZE + ((code128.z >> (n * 4)) & 0xF)];
        #pragma unroll
        for (int n = 0; n < 8; ++n)
            ip += smem_lut[(24 + n) * LUT_SIZE + ((code128.w >> (n * 4)) & 0xF)];

        const float dist = (ip - cb1_sumq) * factor;
        const size_t output = query_offsets[query_idx] + static_cast<size_t>(flat_idx);
        token_scores[output] = dist;
        token_doc_ids[output] = doc_id;

    }
}

__global__ void partial_score_materialize_direct_kernel(
    const float*    __restrict__ d_queries,
    const char*     __restrict__ d_clustered_code,
    const float*    __restrict__ d_clustered_factor,
    const float*    __restrict__ d_cb1_sumq,
    const uint32_t* __restrict__ d_cagra_labels,
    const size_t*   __restrict__ d_cluster_pos,
    const int*      __restrict__ d_clustered_doc_ids,
    float*          __restrict__ d_partial_query_scores,
    const candidate_bitmap_bucket_t* __restrict__ d_candidate_bitmap,
    const candidate_bitmap_offset_t* __restrict__ d_candidate_bitmap_offsets,
    int             num_candidate_docs,
    size_t          num_docs,
    int             nprobe,
    size_t          n_clusters,
    const size_t* __restrict__ query_offsets,
    float* __restrict__ token_scores,
    int* __restrict__ token_doc_ids
) {
    __shared__ uint32_t smem_cstart[kCandidateRefinementMaxNprobe];
    __shared__ int smem_prefix[kCandidateRefinementMaxNprobe + 1];

    const int query_idx = blockIdx.y;
    if (query_idx >= Q_DOCLEN) {
        return;
    }

    const uint32_t* my_labels = d_cagra_labels + query_idx * nprobe;
    for (int p = threadIdx.x; p < nprobe; p += blockDim.x) {
        const uint32_t label = my_labels[p];
        if (label < static_cast<uint32_t>(n_clusters)) {
            const size_t start = d_cluster_pos[label];
            smem_cstart[p] = static_cast<uint32_t>(start);
            smem_prefix[p + 1] = static_cast<int>(d_cluster_pos[label + 1] - start);
        } else {
            smem_cstart[p] = 0;
            smem_prefix[p + 1] = 0;
        }
    }
    if (threadIdx.x == 0) {
        smem_prefix[0] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i <= nprobe; ++i) {
            smem_prefix[i] += smem_prefix[i - 1];
        }
    }
    __syncthreads();

    const int total_elements = smem_prefix[nprobe];
    const float cb1_sumq = d_cb1_sumq[query_idx];

    for (int flat_idx = threadIdx.x + blockIdx.x * blockDim.x;
         flat_idx < total_elements;
         flat_idx += blockDim.x * gridDim.x) {
        int lo = 0;
        int hi = nprobe;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (smem_prefix[mid + 1] <= flat_idx) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        const int cluster_size = smem_prefix[lo + 1] - smem_prefix[lo];
        const int cluster_local = flat_idx - smem_prefix[lo];
        const uint32_t emb_pos = cluster_virtual_to_token_position(
            smem_cstart[lo], cluster_size, cluster_local);

        const uint4 code128 = *reinterpret_cast<const uint4*>(
            d_clustered_code + static_cast<size_t>(emb_pos) * CODE_BYTES);
        const float factor = d_clustered_factor[emb_pos];
        const int doc_id = d_clustered_doc_ids[emb_pos];

        const uint32_t words[4] = {code128.x,code128.y,code128.z,code128.w};
        const float ip = direct_binary_ip(d_queries + query_idx * PADDED_DIM, words);
        const float dist = (ip - cb1_sumq) * factor;
        const size_t output = query_offsets[query_idx] + static_cast<size_t>(flat_idx);
        token_scores[output] = dist;
        token_doc_ids[output] = doc_id;

    }
}

__global__ void retrieved_token_counts_kernel(const uint32_t* labels,const size_t* cluster_pos,
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
    document_scores[doc]=naive_document_score_body(scores,offsets[doc],offsets[doc+1],total_tokens);
}
