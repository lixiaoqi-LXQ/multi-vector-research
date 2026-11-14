# 官方 PLAID：GB10/ARM64 复现入口

源码为官方 ColBERT 提交 `cc4f3dc91c0b45d2d08c251d9d95178285c65f1c`，目录 `ColBERT-main/`，原始来源见 `upstream-commit.json` 和 `upstream.tar.gz`。调用原生 `ResidualCodec`、`optimize_ivf`、`IndexScorer.rank`，未替换搜索算法。

## 环境

`.venv` 使用Python3.13.12和torch2.11.0+cu130；`reused-torch.pth`只读复用既有多向量环境。FAISS GPU1.14.3为独立ARM64 conda环境 `faiss-gpu-env/`。依赖分别见 `environment-freeze.txt`、`faiss-gpu-conda-explicit.txt`。CPU四个C++扩展及GPU两个CUDA扩展已实际运行；GPU缓存为 `torch-extensions-cuda/`，CPU缓存为 `torch-extensions/`，避免同名解压扩展冲突。

## 当前协议

- 数据统一位于 `/home/lxq/graduate/data/multivector-colbertv2/lotte-pooled-dev`，索引位于同一data根下的 `indexes/`；输出留在本目录 `scale-runs/`。
- K为 `2^floor(log2(16*sqrt(Ntokens)))`；全量主索引取官方当前默认kmeans4轮、seed123；Chimera论文未公布迭代数。100k保留本轮20轮高迭代运行验证。文档采样遵循官方实际实现 `min(Ndocs,1+int(16*sqrt(120*Ndocs)))`，不截断100*K；残差heldout为5%，上限50000。
- 全量LoTTE为2428854文档、266205513 tokens、K131072，官方采样实际29971049 tokens。100k前缀为10353669 tokens、K32768、sample5739761 tokens。
- 4bit先训练中心；8bit通过 `--centroids-from` 复用相同FP16中心，重新训练残差桶。FAISS GPU scratch限定512MiB，记录实际RSS及PyTorch分配峰值。
- 参数为独立60点网格：ncells4/8/16/32/64 × ndocs1024/2048/4096/8192 × threshold.45/.3/0；顺序seed42打乱。
- 校准使用统一seed42随机100查询。R10与R100、.90/.95等目标分别按合格点的cal mean latency选择最快参数；每目标最快3个候选在cal100上重复3轮后固定选择。
- `selection-frozen.json` 在独立heldout前写定。全2931查询三轮，同时报告独立heldout2831实际recall；heldout miss只报告，不据此继续选参。
- 正式计时为host FP32预编码query→计时内H2D和half转换→原生rank→host IDs/scores及完成同步。模型编码、索引载入与JSON写入排除。`--query-placement device`只供旧预置GPU输入诊断，不与该边界混用。
- 全量主配置为本机20物理核预算0–19、torch20thread，显式OMP_PLACES以master7开头且逐物理CPU绑定。论文原平台使用32 CPU threads；B1表示一次一条query。历史100k core7/thread1及5–8/4thread为同参预算控制，不混入全量20核主结果。warmup后记录每个实际线程亲和性与CPU ticks，主线程{7}不表示20核预算失效。
- 共享GPU flock、连续10秒idle门槛、整个窗口NVML1秒温度/SM时钟/利用率/功率/pstate/compute PID采样。GB10不支持mem-clock和memory-info时记录原因并省略。

## 执行

获得主agent资源交接后运行：

```sh
flock /tmp/uma-paper-device0.lock /home/lxq/graduate/ai-workspace/research/multivector/plaid/.venv/bin/python /home/lxq/graduate/ai-workspace/research/multivector/plaid/scale_run.py --docs 100000 --bits 4 --tag lotte-prefix100000 --groundtruth /home/lxq/graduate/data/multivector-colbertv2/groundtruth/lotte-prefix-100000
```

100k入口 `scale_run.py` 保持20轮；全量入口 `full_scale_run.py --kmeans-iters 4` 动态命名并记录迭代数，负责构建与持续监测，对应worker一次载入官方scorer后完成独立网格、候选复测、冻结及固定配置三轮。全量预建使用 `full_scale_run.py --docs 2428854 --tag lotte-full --kmeans-iters 4 --build-only`；8bit复用对应 `km4/centroids.pt`。`--quality-targets '10:.90,100:.90'` 可先限定.90目标。8bit高质量目标可显式传入 `100:.99`。

## 验收与溯源

`summarize.py`强制显式给定expected rounds/query rows/topk，拒绝缺轮、截断、重复/错序qid、少于K、非有限值、越界/重复ID及非降序分数；绑定GT manifest、corpus payload、source revision、query及index内容身份。每个构建保存源码/环境/输入与chunk SHA，搜索保存源码快照、contract、逐查询ID/score/elapsed、summary；`test_acceptance.py`提供CPU畸形制品拒绝检查。

早期1万文档km4及预置device-query结果、Ninja失败、CPU中断保留在原有结果和日志中，均为诊断材料。正式入口与当前规则见 `next-grid.json`、`full-scale-plan.md`。

## 放置控制与独立profile

`runner.py --index-placement native|gpu-code-payload|gpu-all-payload` 默认native。code模式仅将codes/residuals逐chunk载入GPU；all模式另移动IVF packed tensor并重建views，doclens/offsets小元数据保留CPU，centroids保持原生GPU。两者不改候选、量化和评分策略。旧100k的selector `gpu-resident`仅表示code模式。`placement_control.py --resident-mode gpu-all-payload`读取已冻结的native参数，先32条同参ID/score对照，通过后native/resident各执行全查询三轮；不会据控制结果重选参数。

`--profile-range --nvtx-stages` 仅用于独立profile，warmup后触发cudaProfilerStart/Stop，可捕获32或100条校准查询；NVTX标注rank/retrieve/score/lookup与PID-offset准备。profile结果带明确标志，不作为QPS证据。正式QPS默认关闭这两个选项。

UMA内存独立报告系统MemAvailable/MemFree、进程RSS和PyTorch分配统计。持续采样及阶段切换快照覆盖before/build/load/search，记录阶段最低MemAvailable；不将RSS与CUDA分配相加充当物理峰值。容量结论由实际全量运行支撑。
