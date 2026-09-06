import torch

def forward(x,y,w,c):
    a = w*x+c
    b = a-y
    L = b ** 2
    return a,b,L

def main():

    w = torch.tensor(3.0, requires_grad=True)
    c = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor(2.0)
    y = torch.tensor(4.0)

    optimizer = torch.optim.SGD([w,c], lr=0.1)
    optimizer.zero_grad()
    
    a,b,L = forward(x,y,w,c)
    L.backward()
    L.backward()

    optimizer.step()

if __name__ == '__main__':
    main()