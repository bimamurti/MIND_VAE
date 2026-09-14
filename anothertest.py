import torch
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y**2
grads = torch.autograd.grad(z, x)
print(z)
print(grads)