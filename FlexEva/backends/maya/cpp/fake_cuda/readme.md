# Fake CUDA
    
FakeCuda 是一个用于模拟 CUDA 环境的开源项目，旨在为开发者提供一个无需实际 GPU 硬件即可测试和调试 CUDA 代码的平台。通过 FakeCuda，用户可以在 CPU 上运行 CUDA 程序，从而简化开发流程并降低硬件依赖。

## 主要功能
- **CUDA API 模拟**：实现了大量常用的 CUDA API，使得大部分 CUDA 程序可以在 FakeCuda 上运行，但是不执行真实的计算任务。
- **日志系统**：提供详细的日志记录功能，帮助用户跟踪程序执行过程中的各类事件和错误。
- **设备管理**：模拟多 GPU 环境，支持设备选择和管理。

## Trace Payload Notes

cuBLAS GEMM wrappers preserve operation/layout metadata for Maya-lite provider
features. `cublasGemmEx` and `cublasGemmStridedBatchedEx` emit shape and
leading-dimension fields; strided-batched calls also emit batch and stride
fields. When the wrapped API has an `algo` argument, fake-cuda emits both
`algorithm` and the raw alias `algo` with the same numeric value so Python
consumers can canonicalize the material signature without losing raw payload
provenance.

# 安装与使用

## 编译

从仓库根目录运行：

```bash
cmake -S fake-cuda -B fake-cuda/build
cmake --build fake-cuda/build -j
```

## 运行项目

环境依赖：

```text
Python 3.12
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers
```

`fake-cuda/frun` 需要可执行的 `fake-cuda/proot`。如默认库发现失败，再检查
`fake-cuda/frun` 中的 PyTorch/CUDA library 发现逻辑。

使用 `frun` 运行 CUDA 程序：

```bash
chmod +x fake-cuda/frun
fake-cuda/frun python tests/workloads/fake_cuda/test.py
```
