import torch

def main():
    w = torch.tensor(3.0, requires_grad=True)
    c = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor(2.0)
    y = torch.tensor(4.0)

    a = 1 + c
    a.retain_grad()
    print(f"a: {a, a.grad, a.grad_fn, a.grad_fn.next_functions}")
    with torch.no_grad():
        a = w * x + c
        print(f"a: {a, a.grad, a.grad_fn}")

if __name__ == '__main__':
    main()