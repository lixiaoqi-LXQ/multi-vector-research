# Table4 relevance指标工具

工具：`/home/lxq/graduate/ai-workspace/research/multivector/scripts/qrels_metrics.py`，仅NumPy/标准库，不调用GPU。输入已经验收并冻结的逐查询排名；不改engine runner。

PLAID单JSONL内三轮：

```sh
/home/lxq/graduate/ai-workspace/research/multivector/plaid/.venv/bin/python \
 /home/lxq/graduate/ai-workspace/research/multivector/scripts/qrels_metrics.py \
 --data /home/lxq/graduate/data/multivector-colbertv2/lotte-pooled-dev \
 --qrels /home/lxq/graduate/data/multivector-colbertv2/lotte-pooled-dev/lotte_pooled_qas.search.tsv \
 --expected-rounds 3 --output /path/to/plaid-qrels-metrics.json \
 /path/to/accepted/per-query.jsonl
```

Chimera三个TSV：使用同样data/qrels，追加三个round TSV路径；输出按文件分别作为一轮。MS MARCO使用data=`/home/lxq/graduate/data/multivector-colbertv2/msmarco-v1-passage`、qrels=`qrels.dev.small.tsv`。

原始query行通过queries_ids.npy映射external qid，返回doc row通过doc_ids.npy映射external docid。PLAID显式核记录query_id与query_row映射一致；Chimera TSV里的query_id是本次输入query数组的local row。calibration子集需加 `--query-rows <calibration-rows.npy>`；全量不加，分母包括全部source queries。

默认要求每轮覆盖全部选定query，缺失报错。显式 `--allow-missing-results` 用于partial诊断，缺失query记0且输出missing count/rows、coverage=false。`--corpus-docs`指定前缀索引规模，检查返回doc row边界并统计有多少query在前缀内没有正标签；**IDCG始终使用原提供qrels中的全部正标签**，不会因prefix缩小而提高ideal分数。

输出每轮MRR@10、NDCG@10、Success@5及轮均值。相关性>0即relevant，NDCG gain=2**rel-1、discount=log2(rank+1)；unjudged document记0。没有正相关标签的query各指标都记0，保留在全query分母。重复query-doc标签取最大grade并记录duplicate/conflict计数。

每份结果文件、qrels、query/doc ID文件、selected row映射以及工具本身均绑定SHA。标签统计包括：选定query数、无正标签query数、索引前缀内无正标签query数、qrels超出当前query集合的数量。

这是relevance标签评测，完全独立于ANN精确MaxSim GT与Recall@K。

CPU手算验证见validation.json：qA等级3/1文档排在rank2/1，qB唯一相关文档在rank2，qC无正标签，分母3：MRR=.5，NDCG=.44691316498944095，Success5=2/3。PLAID三轮与Chimera三TSV一致；还验证缺失query处理、错误外部qid/duplicate doc/round count拒绝，以及local rows经[2,0]映射到原始query行。
