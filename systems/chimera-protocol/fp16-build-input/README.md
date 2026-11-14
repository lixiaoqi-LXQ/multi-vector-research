# FP16 mmap构建输入补丁

交付 `fp16-input.patch`，只在本目录生成；没有改arm-source、没有GPU运行。基线/修改SHA和CPU检查分别在base-source-sha256.json、modified-source-sha256.json、cpu-validation.json。

## 输入与接口

原 `--data <raw-FP32.bin>` / `build_from_ptr(float*)` 保留。新增：

```sh
gpu_build --index /home/lxq/graduate/data/multivector-colbertv2/indexes/chimera/msmarco-full-b8 \
  --doclens /home/lxq/graduate/data/multivector-colbertv2/chimera-inputs/msmarco-full/doclens.bin \
  --fp16-npy /home/lxq/graduate/data/multivector-colbertv2/msmarco-v1-passage/documents.npy \
  --n-clusters 2097152 --ex-bits 7
```

clusters为可调整示例，非论文公开固定值。data与fp16-npy只能指定一个。doclens.bin仍为原int32 count + int32 lengths，来自小doclens.npy即可，不再需要embeddings.bin。索引编码格式不变。

新增C++ `build_from_f16_ptr(const uint16_t* bits, size_t values, size_t dimension, const vector<int>& doc_lens, size_t clusters, size_t ex_bits, const SearchOptions&={})`；values指元素数，源必须在整个build期间保持有效且已处理前缀不变。

MappedFp16Npy支持NPY v1/v2、C-order、<u2原始FP16位模式或<f2矩阵。CLI按sum(doclens)映射精确token前缀，故完整下载前可以使用含完整所需行的.partial文件；不会补零、不会读尚未下载的行。full-MS要求全部597,909,919行完整。映射只读，clean文件页允许OS回收。

## 与FP32路径一致的实现

FloatInputView只改变源读取：

1. seed42 reservoir sampling ID序列不变，选中row在CPU精确展开到原FP32 centroids。
2. assignment仍使用原65,536-row FP32 GPU batch；FP16路径额外使用最大32MiB CPU展开buffer，cudaMemcpy及CAGRA输入仍为FP32。
3. 每个OpenMP编码worker用512B FP32 row scratch展开，传入同一rotator、onebit/fullbit编码器。原FP32路径直接返回原row指针。

转换实现放在新增普通CXX `cpp/src/fp16_input.cpp`。ARM整数NEON构造IEEE FP32位模式，CUDA.cu不include arm_neon.h；非ARM走标量整数实现。不依赖FPCR舍入或flush-to-zero设置。

转换函数覆盖全部65536位模式：±0保留符号、subnormal精确正规化、normal精确展开、±Inf保留、NaN保留符号/payload并quiet，逐位匹配本机NumPy `view(float16).astype(float32)`。转换函数返回是否全部finite；构建在sample/assignment的checked copy中拒绝Inf/NaN，防止非finite进入聚类/量化。encode阶段位于全量assignment验证之后，不在OMP内部抛异常。

## 磁盘与内存

| 数据集 | 必须保留的FP16原始向量 | 避免新增的FP32文件 |
|---|---:|---:|
| LoTTE | 68.15GB | 136.30GB（126.94GiB） |
| MS MARCO | 153.06GB | 306.13GB（285.11GiB） |

新路径额外匿名内存为assignment阶段最多32MiB和编码阶段每worker512B；没有完整FP32语料副本。索引code/factor/IVF/cluster数组峰值保持已有mmap-FP32构建模型：4bit约124N、8bit约188N bytes，另计CAGRA/OS/runtime。MS8bit核心构建峰值仍约104.69GiB，输入方式解决磁盘重复与读取展开，不消除索引自身的容量需求。详细项见memory-disk-model.json。

现有LoTTE FP32文件、全部原始数据与索引都未删除或改动。本补丁与low-memory-load补丁解决不同阶段：此处是build读取，后者是search载入。二者可组合，构建和查询的算法/索引格式保持一致。

## CPU验证与整合

- 全65536 half位模式的bulk NEON与逐项scalar输出均和NumPy FP32位完全一致；不对NaN做近似比较。
- 不对齐输入、各种tail、u16/f16及NPY v1/v2、完整prefix都通过；非法格式/长度/字节序等拒绝。
- 从原源码抽取seed42采样函数，对同一finite语料比较原FP32、新FP16、新FP32，centroids逐字节相同；assignment分块及每encode row也逐字节相同。
- 实际修改后的CLI parser用纯CPU编译测试：FP32/FP16各自接受，两者同时或均未指定均拒绝。
- C++17 -Wall -Wextra -Werror编译运行通过。git apply --check在快照和当前arm-source通过。

整合时在无运行源码改动窗口应用patch，CMake增加fp16_input.cpp这个CXX单元。随后先做GPU小库FP16输入与原FP32输入的build/search等价，再进行full-MS。当前未进行完整CUDA编译或GPU索引对照；CPU验证确定的是展开字节、采样和输入组织一致。
