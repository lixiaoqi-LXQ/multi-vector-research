# Multi-vector research

DGX Spark UMA 平台上的多向量搜索研究：复现 Chimera、官方 PLAID，并探索 GPU 精排等改进。

| 目录 | 用途 |
|---|---|
| `Chimera/` | Chimera / Chimera+ 主源码，CPU/GPU 精排可选 |
| `systems/` | PLAID 官方源码、适配包装及按研究主题组织的实验代码 |
| `scripts/`、`configs/` | 共用工具、绘图源码、实验参数与来源声明 |
| `reports/` | 面向用户的报告、配图及交接文档 |
| `local/` | 本机环境、索引、冻结实验与历史材料；Git 忽略 |
| `ai-workspace/` | AI 临时脚本、草稿、输出；Git 忽略 |

数据集在 `~/data/multivector-colbertv2/`；本机目录为 `~/multi-vector-research`。

- [实验报告](reports/lotte-r10-brief.md)：LoTTE R10/R100，端到端性能与瓶颈。
- [Chimera+ 查询流程](reports/chimera-plus.md)。
- [交接与后续计划](reports/handoff.md)：当前状态、测试约束和用户偏好。

## 版本维护

本仓库 [multi-vector-research](https://github.com/lixiaoqi-LXQ/multi-vector-research) 是整个课题的修改入口。源码、测试包装、配置、补丁和正式报告随 Git 提交；新尝试按主题放在 `systems/<主题>/`，共用工具保持一份。临时代码先放 `ai-workspace/`，整理成熟后再进入维护目录。

`Chimera/` 以 subtree 方式同步到 [Chimera-uma](https://github.com/lixiaoqi-LXQ/Chimera-uma) 的 `Chimera+` 分支；原 ARM 基线为 `baseline`，上游原 `main` 保留。修改在本仓库完成并提交后执行：

```bash
git push origin main
./scripts/sync_chimera.sh
```

PLAID 上游代码固定版本保存在 `systems/plaid/ColBERT-main/`，当前未修改官方文件；包装改动在相邻目录直接追踪。[来源声明](configs/source-provenance.json) 记录两套系统的版本，[端点参数](configs/lotte-endpoints.json) 对应报告中的四组测试。

本机历史运行树完整保留在 `local/legacy-systems/`；原 graduate 路径通过链接指向它。冻结实验文件保持原身份，用于恢复暂停任务。`data`、`indexes` 和 `report-assets` 是本机兼容入口，不提交。
