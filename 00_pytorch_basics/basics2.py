import torch

def forward(x,y,w,c):
    a = w*x + c
    b = a - y
    L = b ** 2
    return a,b,L

def main():
    w = torch.tensor(3.0, requires_grad=True)
    c = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor(2.0)
    y = torch.tensor(4.0)
    # smth = torch.tensor(5.0, requires_grad=True)
    # print(f"smth: {smth.is_leaf}")

    optimizer = torch.optim.SGD([w,c], lr=0.1)

    a,b,L = forward(x,y,w,c)
    print(f"w: {w.grad, w.grad_fn, w.is_leaf}")
    print(f"a: {a.grad_fn, a.is_leaf}")
    print(f"b: {b.grad_fn, b.is_leaf}")
    print(f"L: {L.grad_fn, L.is_leaf}")

    # b.retain_grad()
    L.backward()
    # print(f"b: {b.grad, b.grad_fn, b.is_leaf}")

    optimizer.step()

if __name__ == '__main__':
    main()