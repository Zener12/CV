import torch

def forward(x,y,w,c):
    a = w * x + c
    b = a - y
    L = b ** 2
    return a, b, L

def main():
    lr_list = [0.05, 0.2, 0.3]
    for i in range(3):
        w = torch.tensor(3.0, requires_grad=True)
        c = torch.tensor(1.0, requires_grad=True)
        x = torch.tensor(2.0)
        y = torch.tensor(4.0)

        print(x.is_leaf)

        # optimizer = torch.optim.SGD([w,c], lr=lr_list[i])

        # L_list = []
        # for j in range(10):
        #     optimizer.zero_grad()
        #     _,_,L = forward(x,y,w,c)
        #     L_list.append(L.item())
        #     L.backward()
        #     optimizer.step()
        # print(f"{i+1}. L = {L_list}")

if __name__ == '__main__':
    main()