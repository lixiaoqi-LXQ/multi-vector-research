from pathlib import Path
r=Path(__file__).resolve().parent/'source/cpp'
p=r/'include/chimera/chimera_index.cuh';s=p.read_text().replace('    int num_chunks = 5;','    int num_chunks = 5;\n    bool use_lut = true;\n    bool use_aggregation = true;');p.write_text(s)
p=r/'src/chimera_index_internal.cuh';s=p.read_text().replace('    float* d_lut = nullptr;','''    float* d_lut = nullptr;
    // Actual packed routed-token materialization, only for no-aggregation.
    size_t* d_ablation_offsets = nullptr;
    size_t* h_ablation_offsets = nullptr;
    float* d_ablation_scores = nullptr;
    int* d_ablation_doc_ids = nullptr;
    size_t ablation_token_capacity = 0;''');p.write_text(s)
p=r/'src/utils.cu';s=p.read_text();needle='    check_cuda(cudaMalloc(&ws_->d_lut, LUT_TOTAL_FLOATS * sizeof(float)));';assert needle in s;s=s.replace(needle,'''    if (options_.use_lut) {
        check_cuda(cudaMalloc(&ws_->d_lut, LUT_TOTAL_BYTES));
    }
    if (!options_.use_aggregation) {
        check_cuda(cudaMalloc(&ws_->d_ablation_offsets, (Q_DOCLEN+1)*sizeof(size_t)));
        check_cuda(cudaMallocHost(&ws_->h_ablation_offsets, (Q_DOCLEN+1)*sizeof(size_t)));
    }''');s=s.replace('        (void)cudaFree(ws_->d_lut);','''        (void)cudaFree(ws_->d_lut);
        (void)cudaFree(ws_->d_ablation_offsets);
        (void)cudaFreeHost(ws_->h_ablation_offsets);
        (void)cudaFree(ws_->d_ablation_scores);
        (void)cudaFree(ws_->d_ablation_doc_ids);''');p.write_text(s)
p=r/'apps/utils.cpp';s=p.read_text();needle='        if (arg == "--thread-info") {';assert needle in s;s=s.replace(needle,'''        if (arg == "--kernel-mode") {
            const auto mode = require_value(argc, argv, i, arg);
            if (mode != "full" && mode != "no-lut" && mode != "no-aggregation" && mode != "neither")
                throw std::invalid_argument("--kernel-mode expects full/no-lut/no-aggregation/neither");
            args.options.use_lut = mode != "no-lut" && mode != "neither";
            args.options.use_aggregation = mode != "no-aggregation" && mode != "neither";
        } else if (arg == "--thread-info") {''');p.write_text(s)
p=r/'apps/search.cpp';s=p.read_text();needle='    index.load(args.index_dir, args.options, args.load_options);';s=s.replace(needle,needle+'''\n    std::cout << "[KERNEL_ABLATION] use_lut=" << args.options.use_lut
              << " use_aggregation=" << args.options.use_aggregation << '\\n';''');p.write_text(s)
p=r/'src/search_impl.cu';s=p.read_text();a=s.index('    precompute_one_bit_lut_kernel<<<');b=s.index('\n\n',s.index('    check_cuda(cudaGetLastError());',a));block=s[a:b];s=s[:a]+'    if (options_.use_lut) {\n'+block+'\n    }'+s[b:]
a=s.index('    partial_score_kernel<<<');b=s.index('    check_cuda(cudaGetLastError());',a)+len('    check_cuda(cudaGetLastError());');full=s[a:b];direct=full.replace('partial_score_kernel','partial_score_direct_kernel',1).replace('ws_->d_lut,','ws_->d_queries,',1)
# Keep Full's original kernel call and body. Only dispatch differs.
params=full[full.index('('):full.index(');')+1];matparams=params[:-1]+''',
        ws_->d_ablation_offsets,
        ws_->d_ablation_scores,
        ws_->d_ablation_doc_ids)'''
matlut='    partial_score_materialize_lut_kernel<<<grid, threads_per_block, 0, stream>>>('+matparams[1:]+';\n    check_cuda(cudaGetLastError());'
matdirect=matlut.replace('partial_score_materialize_lut_kernel','partial_score_materialize_direct_kernel',1).replace('ws_->d_lut,','ws_->d_queries,',1)
prep='''        retrieved_token_counts_kernel<<<1,Q_DOCLEN,0,stream>>>(
            ws_->d_cagra_labels,d_cluster_pos_,ws_->d_ablation_offsets+1,
            static_cast<int>(nprobe),ivf->n_clusters);
        check_cuda(cudaGetLastError());
        check_cuda(cudaMemcpyAsync(ws_->h_ablation_offsets+1,ws_->d_ablation_offsets+1,
            Q_DOCLEN*sizeof(size_t),cudaMemcpyDeviceToHost,stream));
        check_cuda(cudaStreamSynchronize(stream));
        size_t total=0;
        for(int q=0;q<Q_DOCLEN;++q) {
            const size_t count=ws_->h_ablation_offsets[q+1];
            if(count>static_cast<size_t>(std::numeric_limits<int>::max()) ||
                total>std::numeric_limits<size_t>::max()-count)
                throw std::overflow_error("Routed token offsets exceed supported capacity");
            ws_->h_ablation_offsets[q]=total;total+=count;
        }
        ws_->h_ablation_offsets[Q_DOCLEN]=total;
        if(total>std::numeric_limits<size_t>::max()/sizeof(float) ||
           total>std::numeric_limits<size_t>::max()/sizeof(int))
            throw std::overflow_error("Packed materialization size overflow");
        if(total>ws_->ablation_token_capacity) {
            float* scores=nullptr;int* docs=nullptr;
            check_cuda(cudaMalloc(&scores,total*sizeof(float)));
            try {check_cuda(cudaMalloc(&docs,total*sizeof(int)));}
            catch(...) {(void)cudaFree(scores);throw;}
            check_cuda(cudaFree(ws_->d_ablation_scores));
            check_cuda(cudaFree(ws_->d_ablation_doc_ids));
            ws_->d_ablation_scores=scores;ws_->d_ablation_doc_ids=docs;
            ws_->ablation_token_capacity=total;
        }
        check_cuda(cudaMemcpyAsync(ws_->d_ablation_offsets,ws_->h_ablation_offsets,
            (Q_DOCLEN+1)*sizeof(size_t),cudaMemcpyHostToDevice,stream));
        ProfileRange materialized(profiling_enabled_,"Chimera/naive_CR_materialized_tokens",total);
'''
replacement='''    if (options_.use_aggregation) {
        if (options_.use_lut) {
'''+full+'''\n        } else {
'''+direct+'''\n        }
    } else {
'''+prep+'''        if(options_.use_lut) {
'''+matlut+'''\n        } else {
'''+matdirect+'''\n        }
        aggregate_materialized_partial_kernel<<<grid,threads_per_block,0,stream>>>(
            ws_->d_ablation_scores,ws_->d_ablation_doc_ids,ws_->d_ablation_offsets,
            ws_->d_partial_query_scores,d_candidate_bitmap_,d_candidate_bitmap_offsets_,
            num_candidate_docs,num_docs);
        check_cuda(cudaGetLastError());
    }'''
s=s[:a]+replacement+s[b:]
a=s.index('            one_bit_token_score_kernel<<<');b=s.index('            check_cuda(cudaGetLastError());',a)+len('            check_cuda(cudaGetLastError());');full=s[a:b];direct='''            one_bit_token_score_direct_kernel<<<token_blocks,256,0,stream>>>(
                ws_->d_queries,d_clustered_one_bit_codes_,d_clustered_one_bit_factors_,
                ws_->d_cb1_sumq,ws_->d_refined_token_positions+token_start,
                ws_->d_one_bit_token_scores+token_start,total_tokens,token_count);
            check_cuda(cudaGetLastError());''';s=s[:a]+'            if(options_.use_lut) {\n'+full+'\n            } else {\n'+direct+'\n            }'+s[b:]
a=s.index('        one_bit_document_score_kernel<<<');b=s.index('        check_cuda(cudaGetLastError());',a)+len('        check_cuda(cudaGetLastError());');full=s[a:b];# chunk-safety wraps this within chunk_count>0
naive='''        one_bit_document_score_naive_kernel<<<(chunk_count+255)/256,256,0,stream>>>(
            ws_->d_one_bit_token_scores,ws_->d_refined_doc_offsets+chunk_start,
            ws_->d_one_bit_document_scores+chunk_start,total_tokens,chunk_count);
        check_cuda(cudaGetLastError());''';s=s[:a]+'        if(options_.use_aggregation) {\n'+full+'\n        } else {\n'+naive+'\n        }'+s[b:];p.write_text(s)
