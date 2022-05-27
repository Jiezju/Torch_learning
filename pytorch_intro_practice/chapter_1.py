import torch

a = torch.arange(0, 12).reshape(3, 4)
idx = torch.argmax(a)
print(idx)  # 1d index

def argmax_index(a):
    idx = torch.argmax(a).item()

    coordinates = []
    shape = list(a.shape)

    for dimension in reversed(shape):
        coordinates.append(idx % dimension)
        idx = idx // dimension

    return coordinates[::-1]

idx_ = argmax_index(a)
print(idx_)

tpk = torch.topk(a.reshape(-1), k=2)
print(tpk)

# auto grad for non-differential grad
def onehot(x):
    y = torch.zeros(x.shape)
    y[x.argmax()] = 1
    return (y - x).detach() + x # 存在导数

x = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
y = onehot(x)
print(y)
y.backward(torch.randn(4))
print(x.grad)

# 矩阵滑窗展开
mat_x = torch.arange(0, 16).reshape(4, 4)
window = mat_x.unfold(0, 3, 1).unfold(1, 3, 1)
print(mat_x)
print(window.shape)
print(window.reshape(window.shape[0], window.shape[1], -1))
