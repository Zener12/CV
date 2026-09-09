import torch

def forward(w, c, x, y):
    a = w*x + c
    b = a - y
    L = b ** 2
    return a, b, L

def backward(x, b):
    dLdb = 2 * b
    dbda = 1
    dadw = x
    dadc = 1
    dLdw = dLdb * dbda * dadw
    dLdc = dLdb * dbda * dadc
    return dLdw, dLdc

def main(*args, **kwargs):
    w = torch.tensor(3.0, requires_grad=True)
    c = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor(2.0)
    y = torch.tensor(4.0)
    print(w.grad, w.grad_fn, w.is_leaf)
    print(c.grad, c.grad_fn, c.is_leaf)

    optimizer = torch.optim.SGD([w,c], lr = 0.1)

    # a,b,L = forward(w,c,x,y)

    # print("Forward 1:")
    # print(w.grad, w.grad_fn, w.is_leaf)
    # print(c.grad, c.grad_fn, c.is_leaf)

    # # backward(x, b, w, c)
    # L.backward()

    # print("Backward 1:")
    # print(w.grad, w.grad_fn, w.is_leaf)
    # print(c.grad, c.grad_fn, c.is_leaf)

    for i in range(2):
        optimizer.zero_grad()
        a,b,L = forward(w,c,x,y)
        L.backward()
        # w = w - lr * w.grad
        # c = c - lr * c.grad
        # w -= lr * w.grad
        # c -= lr * c.grad
        optimizer.step()
        print(f"{i}. Лосс L={L}. Градиент после backward: w={w.grad}, c={c.grad}")



if __name__ == '__main__':
    main()