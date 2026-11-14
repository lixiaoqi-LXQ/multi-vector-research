# 非拥有输入 mmap 构建补丁

补丁来源：未修改的 `Chimera-main`，commit SHA `f6c3a7925924283fc0bc4da64ce78f9cf129bca6`，tree SHA `ddd781995f72211f3f71d025e5674a3c31ea9779`。

应用：在ARM移植源码根目录执行 `git apply --check /home/lxq/graduate/ai-workspace/research/multivector/chimera-protocol/mmap-patch/mmap-build.patch`，再 `git apply` 同一路径。无需CMake新增source列表。补丁仅改public header、build_index.cu、build CLI和新增CLI-only header；不改SIMD评分路径。

新增接口：`build_from_ptr(const float*, size_t embedding_values, size_t dimension, const vector<int>& doc_lens, size_t n_clusters, size_t ex_bits, const SearchOptions&={})`。embedding_values是float元素数，调用期间保持输入有效；内部FloatView只读。原vector接口委托此接口，Python接口不变。

CLI仍接受原格式`[int32 N,int32 D,float32 N*D]`文件。MappedEmbeddings用`PROT_READ|MAP_PRIVATE`，验证正shape、regular file、精确长度、溢出，禁止隐式加载整份anonymous vector；不mlock、不MAP_POPULATE，madvise(SEQUENTIAL)，由OS回收clean文件页。输入文件须完成下载并保持不变；不是NPY读取器。

内存改动保持字节布局和采样/旋转/量化语义：

1. assignments/centroids/rotated_centroids在IVF和graph完成后释放。
2. full_codes以vector<char>生成（写入时cast uint8_t指针），结束直接move给成员，避免全量fullcode复制。
3. cluster code/factors/doc_ids/token_positions直接写入最终序列化buffer，保持原4段相同顺序，避免28*N中间副本。
4. token_doc_ids完成重排后释放。

内存模型在memory-model.json。D128时encoded=(24+B)*N，最终拥有数组=(56+B)*N，重排峰值数组=(60+B)*N，B为64或128 bytes/token。LoTTE核心数组峰值约30.74GiB(4bit) /46.61GiB(8bit)。另加CAGRA持有数据/图与构建workspace、少量文档数组、allocator/cache；mmap输入136.3GB是虚拟映射和可回收文件页，非136.3GB不可回收匿名内存。需用实际进程RSS/cgroup共享内存和CUDA内存记录最终峰值；本任务未运行GPU。

验证：
- 原源码 `git apply --check`通过。
- c++17 -Wall -Wextra -Werror编译mmap CPU测试，成功读取payload并拒绝不存在/截断/负shape/额外尾部文件。
- c++17/OpenMP抽取原版与新版重排函数，123-token非平凡置换、四段字节完全相同。
- 完整CUDA构建及索引search等价验证由ARM agent整合后执行。

测试源码test_mmap.cpp、test_reorder.cpp和二进制在validation目录；均为临时本地材料。
