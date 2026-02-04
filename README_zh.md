# SpyInfer: 高性能大语言模型推理引擎

SpyInfer 是一个从零开始、用 C++ 构建的轻量级、高性能大语言模型（LLM）推理引擎。它专为追求极致性能和效率而设计，集成了业界前沿的推理服务技术。

## ✨ 核心特性与性能亮点

SpyInfer 旨在追求极致的速度与效率，集成了诸多前沿优化技术：

- **高性能后端**:
    - **CPU (x64)**: 充分利用SIMD **AVX512 内在函数**进行向量化运算，并为关键神经网络原语提供手写优化的内核。
    - **GPU (CUDA)**: 集成了 **FlashAttention**（预填充阶段）和 **PagedAttention**（解码阶段），以实现高效的注意力计算和 KV 缓存管理, 并采用自定义 CUDA 内核以达到最大吞吐量。支持 **混合精度 (FP16)** 提升性能并减少内存占用。
- **先进的 LLM 引擎**: 具备精密的 **动态内存管理 (BlockManager)** 和 **请求调度** 功能，以实现高吞吐量的连续批处理。
- **双后端支持**: 灵活部署在 CPU 和支持 CUDA 的 GPU 上。
- **现代化工具链**: 使用 [Xmake](https://xmake.io/) 进行构建，并支持 `SafeTensors` 格式的模型权重。
- **可扩展架构**: 易于扩展以支持新的模型架构，目前已初步支持 Qwen3。

## 🚀 快速开始

### 环境要求

- 支持 C++17 的现代 C++ 编译器
- [Xmake](https://xmake.io/#/zh-cn/guide/installation)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (如需 GPU 支持)

### 编译项目

1.  **克隆仓库:**
    ```bash
    git clone https://github.com/your-username/spyinfer.git
    cd spyinfer
    ```

2.  **使用 Xmake 配置和编译:**

    - **CPU 版本:**
      ```bash
      xmake f -m release/debug --backend=cpu
      # 运行编译
      xmake
      ```

    - **CUDA 版本:**
      在编译前，请确保 CUDA 工具链已正确安装。Xmake 会自动检测。
      ```bash
      # 配置项目以启用 CUDA
      xmake f -m release/debug --backend=cuda --cuda=version #(12.2)
      # 运行编译
      xmake
      ```

3.  **运行程序:**
    编译成功后，可执行文件位于 `build` 目录下。
    ```bash
    # 可通过脚本快速启动
    ./run.sh
    ```

## 使用说明

项目的主要入口点是 `main.cpp`。您可以修改此文件来加载您指定的模型和分词器，然后执行推理。

```cpp
// main.cpp 中的示例用法
int main(int argc, char** argv) {
    // 1. 初始化后端 (CPU 或 CUDA)
    // 2. 加载模型和权重
    // 3. 创建分词器
    // 4. 输入提示词并执行推理
    // 5. 打印结果
    return 0;
}
```

项目中的 `run.sh` 脚本是一个很好的起点，展示了如何执行编译后的程序。如果您需要进行性能分析，可以使用 `run_perf.sh` (适用于 Linux `perf`) 或 `run_nc.sh` (适用于 NVIDIA Nsight Compute)。

## 路线图 (Roadmap)

- [ ] 增加分布式推理支持 (张量并行)。
- [ ] 支持更多模型架构 (例如 Llama, Gemma)。
- [ ] 实现量化支持 (例如 GGUF, AWQ)。
- [ ] 开发 Python API 以便更轻松地集成。
## 如何贡献

欢迎任何形式的贡献！如果您有任何想法，随时可以提交 Pull Request 或开启一个 Issue 进行讨论。
