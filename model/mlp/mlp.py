import torch
import torch.nn as nn
import torch.onnx
import numpy as np

# 定义 MLP 模型
class MiniMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 3)

        # 固定参数，便于验证
        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9,10,11,12],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ], dtype=torch.float32))
            self.fc1.bias.copy_(torch.tensor([1, 1, 1, 0, 0], dtype=torch.float32))

            self.fc2.weight.copy_(torch.tensor([
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
            ], dtype=torch.float32))
            self.fc2.bias.copy_(torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32))

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 实例化与导出
model = MiniMLP()
model.eval()

dummy_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)

# 保存输入输出以供验证
torch.onnx.export(
    model,
    dummy_input,
    "mini_mlp.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    do_constant_folding=True,
    dynamic_axes=None
)

# 保存为 .npy 验证数据
np.save("input.npy", dummy_input.numpy())
np.save("output.npy", model(dummy_input).detach().numpy())

print("✅ 导出完成：mini_mlp.onnx + input.npy + output.npy")
