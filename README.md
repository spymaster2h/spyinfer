# SpyInfer: A High-Performance LLM Inference Engine

SpyInfer is a lightweight, high-performance LLM (Large Language Model) inference engine built from scratch in C++. It is designed for maximum performance and efficiency, incorporating modern techniques for serving large models.

## ✨ Key Features & Performance Highlights

SpyInfer is built for speed and efficiency, incorporating cutting-edge optimizations:

- **High-Performance Backends**:
    - **CPU (x64)**: Leverages **AVX512 intrinsics** for vectorized operations and hand-tuned kernels for critical neural network primitives.
    - **GPU (CUDA)**: Integrates **FlashAttention** (prefill) and **PagedAttention** (decode) for efficient attention computation and KV cache management. Supports **mixed-precision (FP16)** for improved performance and reduced memory footprint.
- **Advanced LLM Engine**: Features sophisticated **dynamic memory management (BlockManager)** and **request scheduling** to enable high-throughput continuous batching.
- **Dual Backend Support**: Flexible deployment on both CPU and CUDA-enabled GPUs.
- **Modern Tooling**: Uses [Xmake](https://xmake.io/) for building and supports `SafeTensors` for model weights.
- **Extensible Architecture**: Easily extendable to new model architectures, with initial support for Qwen3.

## 🚀 Getting Started

### Prerequisites

- A modern C++ compiler (supporting C++17)
- [Xmake](httpss://xmake.io/#/guide/installation)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (Required for GPU support)

### Building the Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/spyinfer.git
    cd spyinfer
    ```

2.  **Configure and Build with Xmake:**

    - **For CPU Backend:**
      ```bash
      xmake f -m release/debug --backend=cpu
      xmake
      ```

    - **For CUDA Backend:**
      Before building, make sure the CUDA toolkit is correctly installed and accessible. Xmake will automatically detect it.
      ```bash
      # This command configures the project to build with CUDA
      xmake f -m release/debug --backend=cuda --cuda=version #(12.2)
      # Run the build
      xmake
      ```

3.  **Run the executable:**
    After a successful build, the main executable will be located in the `build` directory.
    ```bash
    # Example using the run script
    ./run.sh
    ```

## Usage

The main entry point is `main.cpp`. You can modify it to load your desired model and tokenizer, and then run inference.

```cpp
// Example usage from main.cpp
int main(int argc, char** argv) {
    // 1. Initialize backend (CPU or CUDA)
    // 2. Load model and weights
    // 3. Create a tokenizer
    // 4. Run inference with a prompt
    // 5. Print results
    return 0;
}
```

The provided `run.sh` script is a good starting point to see how to execute the compiled program. For performance analysis, you can use `run_perf.sh` for Linux `perf` or `run_nc.sh` for NVIDIA's Nsight Compute.

## Roadmap

- [ ] Add support for distributed inference (Tensor Parallelism).
- [ ] Support more model architectures (e.g., Llama, Gemma).
- [ ] Implement quantization support (e.g., GGUF, AWQ).
- [ ] Develop a Python binding for easier integration.
## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss your ideas.
