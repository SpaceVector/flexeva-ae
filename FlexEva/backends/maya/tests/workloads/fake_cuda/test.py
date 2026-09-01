import torch
import os
print("=== PyTorch编译信息 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA版本: {torch.version.cuda}")
# 查看支持的架构列表
arch_list = torch.cuda.get_arch_list()
print(f"\n支持的架构: {arch_list}")