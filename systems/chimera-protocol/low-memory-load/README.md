# Chimera 显式低内存load补丁

补丁：`low-memory-load.patch`；基线为本任务ARM移植源码的本地快照，含mmap FP32构建和nprobe<=512 guard。基线各文件SHA在base-source-sha256.json，修改后SHA在modified-source-sha256.json。没有修改正在使用的arm-source，也没有GPU运行。

## 接口与默认行为

保留原来的 `load(index_dir, SearchOptions)`，实现委托默认 `LoadOptions{}`。新增三参overload：

```cpp
Chimera::LoadOptions storage;
storage.low_memory = true;
storage.transfer_chunk_bytes = 64ull * 1024ull * 1024ull;
storage.prefault_full_codes = true;
index.load(index_dir, search_options, storage);
```

CLI显式增加 `--low-memory`。不指定时保留原版拥有数组的load和save行为。build/build_from_ptr/save正常路径保持不变。低内存load后save()在写文件、创建目录或清除旧输出之前抛logic_error；错误信息提示用默认load后保存。原索引文件不变。

## 实现

1. `doc_full.bin`以PROT_READ|MAP_PRIVATE映射为普通CPU只读地址；CPU评分只把原vector数据指针换为mapping指针，位宽、解码、NEON/AVX、MaxSim、prefetch和CPU/GPU分工均不变。
2. `doc_1bit.bin`读完原header/rotator后跳过CPU搜索不用的doc1bit code16N和onebit factor4N，只读full-bit factor4N。
3. `cluster_1bit.bin`的code/factor/docid/tokenposition四段通过单一64MiB可配置CPU缓冲，依次同步cudaMemcpy至原来的四个GPU数组；不保留hostcluster28N完整副本。
4. CPU full-code文件页在load阶段逐页prefault，排除首次磁盘加载进入查询计时；页保持可回收。该过程不是mlock、mappedGPU或zero-copy GPU算法。查询阶段GPU仍仅访问原GPU驻留cluster数组，CPU仍随机访问完整full-code字节。
5. 切换同一对象从default load到low-memory load时，显式丢弃旧vector容量。release_resources在原同步和清理后解除mapping并重置storage标记；再次default load/build恢复原路径。

输入索引文件在对象存活期应保持不变。mmap和流式上传都检查精确payload大小，阻止截断/尾部额外数据。map只读，不修改源文件。

## 内存模型

D128，B=logical_bits*16 bytes/token：

| 数据集/位宽 | 默认核心数组 | low-memory全部CPUcode页驻留核心 | 节省 |
|---|---:|---:|---:|
| LoTTE4 | 36.69GiB | 24.79GiB | 11.90GiB |
| LoTTE8 | 52.56GiB | 40.66GiB | 11.90GiB |
| MS MARCO4 | 82.41GiB | 55.68GiB | 26.73GiB |
| MS MARCO8 | 118.05GiB | 91.32GiB | 26.73GiB |

默认=(84+B)N；新路径36N匿名/设备常驻+BN可回收映射文件页。36N包括fullfactor4N、IVF4N和GPUcluster28N。另有CAGRA图/中心及workspace、candidate arena/CUB、refined token scores、doclens/ptr、runtime/driver/OS和其他进程。

MS8bit普通krefine8000/docmax300的refined token score上界约293MiB，加candidate arena和CAGRA后仍有较原版明显的容量空间；最终128GB运行结果需要整合GPU验证。把krefine扩大到全体数百万文档时，评分工作区可以单独超过容量，该配置不在本预算内。详细项见memory-model.json。

## 已执行CPU验证

`test_file_io.cpp`通过C++17 -Wall -Wextra -Werror编译运行：

- 4/8bit文件与原vector每个token字节完全一致，prefault不修改文件。
- 四段cluster在chunk=1/7/64/1024/65536时全部字节一致，传入callback的数据量不超过buffer。
- missing/nonregular/empty/wrong-size payload、零buffer和错误segment总长均拒绝。
- full-factor跳过20N后读取正确。
- save guard先抛错，既有输出文件保留。
- git apply --check在独立base快照和当前ARM源码均通过。

详细范围在cpu-validation.json。未进行完整CUDA编译、GPU搜索等价、GPU峰值内存或性能测量。

## 整合步骤

在空闲源码版本上先 `git apply --check <本目录>/low-memory-load.patch`，再应用；无需新增CMake编译单元，新增helper是header-only。编译后先用10k/100k索引分别比较default和--low-memory的query IDs、分数、Recall及三轮输出，再记录真实共享内存和CPU page-fault/IO。若比较性能，要分别披露load方式和prefault设置；低内存模式的内存收益不能混同为算法QPS收益。

当前Python绑定继续使用原两参load默认路径；本补丁提供C++与CLI opt-in，未增加Python绑定参数。run_search_grid.py还需由整合者显式透传--low-memory并写入运行身份，不会自动开启。
