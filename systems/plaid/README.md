# 官方 PLAID：维护入口

当前官方源码在 `ColBERT-main/`，上游版本见 `../../configs/source-provenance.json`；逐文件核对未发现源码修改。适配与计时包装在本目录单独追踪。

当前报告使用官方 native GPU 路径、完整查询集，无自定义留出集；参数见 `../../configs/lotte-endpoints.json`。搜索入口为 `endpoint_search_runner.py`，结果校验为 `summarize.py`，R100 正式协议源码保存在 `protocols/r100/`。GPU-all 等放置控制是历史探索，不纳入当前主报告。

运行环境和历史冻结执行目录在 `../../local/legacy-systems/plaid/`。算力统一通过 `~/device.lock` 协调，正式测试持互斥锁；后台任务当前保持暂停。原历史说明保留在该本地目录的 README 中。
