import torch.nn as nn

# Linear_model = nn.Linear(3,2)
# print(list(Linear_model.named_parameters()))

# BNlayer = nn.BatchNorm1d(4)
# print("Named parameters")
# print(list(BNlayer.named_parameters()))
# print("State dict")
# print(list(BNlayer.state_dict()['running_mean']))

model = nn.Sequential(
    nn.Linear(3,2),
    nn.Linear(5,1)
)

# for param in model.parameters():
#     print(param)

# print(list(model.named_parameters()))

# model[0].eval()
# print(model[0].training)
# model[0].train()
# print(model[0].training)