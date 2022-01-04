import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print('Gradient function for z =', z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)

loss.backward() # loss 必须为标量
print(w.grad) # loss 关于 w 的梯度
print(b.grad)

z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

# 向量矩阵求导
inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print("First call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nSecond call\n", inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)

# 计算 jacobian 矩阵
def exp_adder(x, y):
    return 2 * x.exp() + 3 * y

inputs = (torch.rand(2), torch.rand(2)) # arguments for the function
print(inputs)

out = exp_adder(*inputs)

delta_out = torch.autograd.functional.jacobian(exp_adder, inputs)
print(delta_out)