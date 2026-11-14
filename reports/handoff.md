# 多向量搜索课题交接

## 当前目标与状态

本课题研究 DGX Spark UMA 平台上的多向量搜索，优先复现 Chimera 的实验设置，与官方 PLAID GPU 对比；Chimera+ 是仅把 CPU 完整位宽精排替换成 GPU 的独立优化。

**本轮完成目录及版本维护整理；后台计算保持停止。下一位 Agent 先确认用户要继续的实验，再启动计算。**

| 工作 | 状态与入口 |
|---|---|
| LoTTE 全量实验 | R10/R100 × 0.90/0.95，原 Chimera、Chimera+、官方 PLAID 已有端到端结果；见[唯一性能报告](lotte-r10-brief.md) |
| 瓶颈与 breakdown | 原 Chimera 阶段分析、线程/chunks 微调已写主报告；Chimera+ R100 0.95 全流程已有短样本观测及带宽/计算估算 |
| Chimera+ 流程 | [查询流程文档](chimera-plus.md)，含单 query 的 kernel、传输和同步 |
| MS MARCO 数据 | 已下载；8,841,823 篇文档、597,909,919 条 D128 向量、6,980 条查询 |
| MS Chimera 主 4bit 索引 | 安全暂停，已提交 142,344,192 / 597,909,919 tokens（约 23.8%，2172 批） |
| MS PLAID 主索引 | 尚未开始；原后台交接已取消，不自动启动 |
| R100 0.95 chunks 追加探索 | 用户已要求暂停，不作为待自动执行任务 |

本轮只做文件、Git 和静态检查，没有新增性能实验。

## 后续测试规划

1. 用户确认继续 MS 后，优先恢复**当前主索引检查点**，不要另建一份同规模索引；恢复前核对冻结控制、原二进制、已提交批次和设备锁。
2. Chimera 主索引完成并通过语义/完整性验收后，再串行构建官方 PLAID；同一时间只有一个索引 build，CPU 重量化也算构建。
3. 先验证一小段查询能完成、结果正确，并估算单点耗时；再按召回目标选参，正式测试完整查询集。批量或 B1 选参取决于实际总耗时，以快出可核验结果为目标。
4. 正式比较使用符合质量目标的最好已测参数或明确的默认值；记录实际召回。新数据集是否测 R10/R100、0.90/0.95，由用户决定，不自动沿用所有历史组合。
5. 未获新指令前不做 8bit、额外消融、新数据集或大规模参数扫描；先交付一个数据集的完整结果。

LoTTE 已完成点的参数保存在 [`configs/lotte-endpoints.json`](../configs/lotte-endpoints.json)。搜索参数不是本轮迁移后重新调出的。

## 实验约束与用户偏好

- **复现优先，源码少改。** ARM/平台必要适配要可追踪；性能优化单独说明。Chimera+ 只调整完整位宽精排，沿用原搜索参数和索引。
- **正式测试用 B1。** LoTTE 使用全部 2,931 条查询 × 3 轮，每轮预热 32 条，主配置 20 CPU 线程；无自定义留出集。查询及文档向量均预编码，原始文档数据是 uint16 承载 FP16 位模式。
- **质量和计时边界一致。** 分别实际返回 Top-10/Top-100，评价对应 recall；计时从主机 FP32 query 到主机结果完成，含上传，排除编码、索引加载、预热、写盘及召回计算。
- **PLAID 汇报官方 GPU native 路径。** GPU-all 是历史放置探索，当前主报告不把它当主基线。官方 ColBERT 源码现与固定上游逐文件相同，适配在外层包装。
- **性能展示简明。** QPS 与延迟并列，延迟包含平均值和各轮 P99 均值；不同质量目标放在一起。breakdown 重点分析 Chimera，不给 PLAID 另铺一套阶段图。区分正式全量结果与短样本诊断。
- **不预设增加线程/chunks 一定更快。** 已测主配置 chunks=1；历史微调及重叠/调度开销见主报告，不靠直觉覆盖实测。
- **设备协调。** CPU/GPU 统一 `~/device.lock`；构建和正式性能测试持互斥 `flock`，覆盖初始化、子进程与同步。持续留意外部计算干扰，不停止其他用户进程。
- **交付。** 一个持续更新的 Markdown 性能报告，结论短、配图直观；Chimera+ 流程文档是用户另行要求的独立材料。不要 PDF，不自动送 doc-review，用户直接审阅。长任务完成后通知用户。

## 目录和版本维护

```text
~/multi-vector-research/
├── Chimera/                 主源码；CPU/GPU 精排可选
├── systems/                 PLAID、适配包装及按主题组织的研究代码
├── scripts/  configs/       共用工具、绘图源码、静态设置和来源
├── reports/                 当前报告、配图与本交接
├── local/                   本机环境、索引、冻结运行和历史材料；忽略
└── ai-workspace/            AI 临时脚本、草稿和输出；忽略
~/data/multivector-colbertv2/  数据集、查询、真值
```

- 主仓库：[multi-vector-research](https://github.com/lixiaoqi-LXQ/multi-vector-research)，维护整个课题。
- Chimera 镜像：[Chimera-uma](https://github.com/lixiaoqi-LXQ/Chimera-uma)，`Chimera+` 为当前源码，`baseline` 保留 ARM 基线，原上游 `main` 保留。主仓库提交后运行 `scripts/sync_chimera.sh` 同步，避免两处各自修改。
- 来源与历史节点见 [`configs/source-provenance.json`](../configs/source-provenance.json)。PLAID 源码、测试包装和补丁均可直接查看 Git diff。
- 新想法用主题目录组织；临时探索先进入 `ai-workspace/`，整理成熟后提升为维护代码。面向用户的文档润色后进入 `reports/`。`local/` 和 `ai-workspace/` 不强制加入 Git。
- `~/multi-query` 已移除。原 graduate 项目、数据与运行目录保留链接；另一个历史 multi-query 项目不属于本次迁移范围。

## 恢复暂停任务的入口

以下是**本机冻结材料**，不作为新的启动授权：

- 运行树：`local/legacy-systems/`。
- MS 恢复控制：`local/legacy-systems/chimera/ms-resume-device-v3/`，先读 `spec.json`、原始 `freeze.json` 和 `post-CPU-audit-freeze.json`。后者记录已授权的完整性校验更新，核验时要结合两份冻结声明。
- 停止回执：`local/legacy-systems/chimera/runs/full-ms-owned-sorted-checkpoint-resume-v3/stop-for-migration-receipt.json`。
- 最终索引计划输出：`local/indexes/chimera/msmarco-v1-passage-full-c4194304-b4-owned-sorted-gather-v1/`，尚未生成；恢复依据为下述现存缓存与冻结控制。
- 构建缓存：`local/index-build-cache/msmarco-v1-passage-full-c4194304-b4-owned-sorted-gather-v1/`。
- 当前恢复使用 source-sorted gather 构建分支；原冻结源码、监督进程脚本、环境和二进制都在原运行树，迁移不改写其内容。
- 正式 PLAID R100 两个协议脚本已从旧临时目录整理到 `systems/plaid/protocols/r100/` 追踪；对应冻结输入仍在本机历史运行树。

不要直接复用旧日志中的 PID 或进度启动命令。恢复时核对真实制品与当前进程；暂停以来没有自动继续计算。
