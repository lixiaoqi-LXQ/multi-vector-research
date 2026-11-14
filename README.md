# Multi-query：Chimera / PLAID 多向量检索

当前项目统一位于 `/home/lxq/multi-query`。研究 DGX Spark（GB10、20核ARM、128GB UMA）上的多向量搜索，包含原Chimera、GPU完整4bit精排的Chimera+及官方PLAID GPU路径。

## 入口

- [统一实验报告](reports/lotte-r10-brief.md)：LoTTE R10/R100、平均/P99延迟和瓶颈。
- [Chimera+单query流程图](reports/chimera-plus.md)：24次kernel调用、传输及同步。
- [Chimera+源码说明](Chimera/README.md)。

| 路径 | 内容 |
|---|---|
| `Chimera/` | 当前Chimera+ C++/CUDA源码和数值测试 |
| `systems/plaid/ColBERT-main/` | 官方ColBERT/PLAID源码 |
| `systems/plaid/*.py` | 预编码数据接入、原生GPU搜索、验收与实验入口 |
| `systems/chimera/` | ARM适配、原Chimera搜索/构建监督与验收工具 |
| `systems/chimera-ms-sorted-gather/` | MS MARCO主索引恢复使用的冻结构建源码与工具 |
| `systems/chimera-alg-ablation/source/` | 已测原Chimera二进制对应源码，正式模式为Full |
| `configs/`、`scripts/` | 项目设置与数据准备入口 |
| `reports/`、`report-assets/` | 当前报告、配图与绘图源码 |
| `indexes/` | 本地索引，不提交Git |
| `ai-workspace/` | 构建缓存、临时材料、迁移收据与状态，不提交Git |
| `data` | 链接到 `/home/lxq/data/multivector-colbertv2` |

数据集、查询和真值在 `~/data/multivector-colbertv2/`；索引实际保存在本项目`indexes/`。数据目录里的旧`indexes`入口链接到此目录，构建缓存链接到本项目`ai-workspace/index-build-cache/`。

## Git与兼容路径

项目已统一为一个Git仓库，当前分支`Chimera+`。原Chimera+提交`c97f24c`及`baseline`分支历史保留；本次迁移提交将源码归入`Chimera/`并纳入PLAID和实验工具。数据、索引、日志、环境和原始实验结果留在本地，不进Git。

原graduate项目、运行工作区和数据目录保留符号链接。冻结实验规格、编译缓存和既有Python环境因此仍可通过旧绝对路径访问；迁移没有改写冻结源码、二进制或检查点。

## 当前后台状态

MS MARCO数据已完整下载。Chimera主4bit索引在迁移前已安全停止：142,344,192 / 597,909,919 tokens，2172个已提交批次。检查点保存在`ai-workspace/index-build-cache/`；PLAID全量MS主索引尚未开始。迁移后不自动恢复计算。

继续重负载任务前按 `~/device.lock` 取得独占锁，同一时间只构建一个索引。新运行输出统一放`ai-workspace/`。现有`systems/`内历史运行材料按忽略规则保留，源码与必要配置由Git追踪。
