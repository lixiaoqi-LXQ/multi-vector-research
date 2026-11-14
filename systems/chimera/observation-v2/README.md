# 当前 Full 搜索路径的独立观测构建

本目录从已接纳算法 Full 搜索版本 `c54539b5…` 复制源码，仅修改 `profiling.hpp` 和 `search_impl.cu`，增加 CG、CR 两个 NVTX 子范围。移除三行范围脚手架后，搜索计算文本与来源逐字一致；其他源码文件 SHA 保持一致。既有 chunk 空尾与零候选处理随来源保留。

`source-freeze.json` 绑定完整源码，`CPU-validation.json` 记录实际 sm_121 编译与链接、日志和二进制身份。当前观测二进制为 `build/gpu_search`，SHA256 `65589e36d4772f31c1dc112468658cc2ffe64363521ceb1256477d5735331875`。旧 `../observation/` 与全部 QPS 二进制保持原样。

下一步先在同一已审计索引上比较主版本与观测版本的 32 查询完整 ID/FP32 score 位模式，再进入实际 NSYS capture。新入口准备于 `../profile-v2/`，需真实独占设备窗口、冻结校准点与完整进程监督。正式 QPS 使用未启用 profiler 的主版本。

现有 `../observation/parse_khat.py` 从 `Chimera/cpu_fullbit` 的 INT64 payload 按 query/chunk 汇总实际 full-bit 文档评分次数。CPU 解析检查已在旧目录通过；真实 NSYS 导出格式、capture 查询数、计数和 CPU/GPU 重叠仍待运行验证。
