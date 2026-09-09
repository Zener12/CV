import torch.nn as nn
import torch

def criterion(pred, y):
    b = pred - y
    L = b ** 2
    return L

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(3.0, requires_grad=True))
        self.c = nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward(self, x):
        a = self.w * x + self.c
        return a

x = torch.tensor(2.0)
y = torch.tensor(4.0)

model = MyModule()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for i in range(2):
    optimizer.zero_grad()
    pred = model(x)
    L = criterion(pred, y)
    L.backward()
    optimizer.step()
    print(L)

# torch.save(model.state_dict(), "00_pytorch_basics/00_2/state_dict.pth")
model2 = MyModule()
model2.load_state_dict(torch.load("00_pytorch_basics/00_2/state_dict.pth", weights_only=True))

optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)
for i in range(2):
    optimizer2.zero_grad()
    pred2 = model2(x)
    L2 = criterion(pred2, y)
    L2.backward()
    optimizer2.step()
    print(L2)