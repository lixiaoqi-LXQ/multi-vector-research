# R100 正式测试协议源码

这两个文件逐字节保存了已交付 R100 官方 PLAID 全量三轮测试的驱动与 worker。原始执行上下文和冻结输入在 `local/legacy-systems/plaid/ai-workspace/r100-native-device-lock-v2/`；此处用于追踪协议源码，保留当时的相对路径与输入约束。

新的实验直接使用 `systems/plaid/endpoint_search_runner.py`，端点参数见 `configs/lotte-endpoints.json`。调整协议时新增明确的维护入口，不改写原冻结执行目录。
