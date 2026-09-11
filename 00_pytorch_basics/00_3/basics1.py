import torch
import torch.nn as nn

# Эксперимент А
logits = torch.tensor([[101.0, 102.0, 103.0]])
softmax = torch.softmax(logits, dim=1)
# print(softmax)

# Эксперимент Б
# target = torch.tensor([2], dtype=torch.long)
criterion = nn.CrossEntropyLoss()
# ce = torch.tensor([])
# print(criterion(logits, target))

# Эксперимент В
# exp_tensor = torch.exp(logits)
# print(softmax)
# softmax2 = torch.tensor([[exp_tensor[0][0] / (exp_tensor[0][0] + exp_tensor[0][1] + exp_tensor[0][2]),
#                           exp_tensor[0][1] / (exp_tensor[0][0] + exp_tensor[0][1] + exp_tensor[0][2]),
#                           exp_tensor[0][2] / (exp_tensor[0][0] + exp_tensor[0][1] + exp_tensor[0][2]),]])
# print(softmax2)

# Эксперимент Г
# print(criterion(logits, target))
# print(criterion(torch.softmax(logits, dim=1), target))

# Эксперимент Д
logits_D = torch.tensor([[0.0 for i in range(37)]])
target_D = torch.tensor([36], dtype=torch.long)
Loss_D = criterion(logits_D, target_D)
print(Loss_D)