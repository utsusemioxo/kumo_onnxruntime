import torch
import torch.nn as nn
import torch.onnx

# build minimal linear model
class MiniLinear(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(4, 3, bias=True)
    # fix param for later test
    with torch.no_grad():
      self.linear.weight.copy_(torch.tensor([[1,2,3,4],
                                             [5,6,7,8],
                                             [9,10,11,12]], dtype=torch.float32))
      self.linear.bias.copy_(torch.tensor([1,1,1], dtype=torch.float32))

  def forward(self, x):
    return self.linear(x)

model = MiniLinear()
model.eval()

dummy_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

# export onnx model
torch.onnx.export(
  model,
  dummy_input,
  "mini_linear.onnx",
  input_names=["input"],
  output_names=["output"],
  opset_version=11,
  do_constant_folding=True,
  dynamic_axes=None
)

print("Export Success: mini_linear.onnx")