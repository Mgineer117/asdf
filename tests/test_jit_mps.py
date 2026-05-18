import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3)
    def forward(self, x):
        return self.conv(x)

device = "mps"
model = MyModel().to(device)
dummy = torch.zeros(1, 1, 84, 84, device=device)
try:
    traced = torch.jit.trace(model, dummy)
    print("Success on", device)
except Exception as e:
    print("Error:", e)
