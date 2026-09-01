import torch
import torch.nn as nn
import torch.optim as optim

# 1. 设置设备：如果有 GPU 就用 GPU，否则用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")

# 2. 定义一个超简单的网络 (输入维度10 -> 隐藏层5 -> 输出维度1)
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
).to(device)  # <--- 关键：将模型移动到 GPU

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 生成虚拟数据 (Batch Size=32)
# <--- 关键：将数据移动到 GPU
inputs = torch.randn(32, 10).to(device) 
targets = torch.randn(32, 1).to(device)

print("\n开始训练...")

# 5. 训练循环 (演示 5 步)
for step in range(5):
    # --- 前向传播 (Forward) ---
    outputs = model(inputs)           # 模型预测
    loss = criterion(outputs, targets) # 计算损失

    # --- 反向传播 (Backward) ---
    optimizer.zero_grad()             # 清空上一步的梯度
    loss.backward()                   # 计算新梯度
    optimizer.step()                  # 更新模型参数

    print(f"Step {step+1}: Loss = {loss.item():.6f}")

print("\n训练完成！")
