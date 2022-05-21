from torch.autograd import Function
import torch

'''
一个新算子：
1. 必须继承 Function
2. 必须实现 forward backward

'''
class ReLU(Function):  # 继承Function类，自定义新的前向反向计算
    @staticmethod
    def forward(ctx, input):  # 输入参数列表，可以多个输入input
        output = torch.clamp(input, min=0)  # 计算max(input, 0)
        ctx.save_for_backward(output)  # ctx保存中间结果，用以反传过程使用
        return output

    @staticmethod
    def backward(ctx, grad_output):  # grad_output个数与output个数相同， 上一层 传下来的导数
        output = ctx.saved_tensors[0]  # 从ctx获取暂存的中间结果
        return (output > 0).float() * grad_output  # 计算I(x > 0) * grad_output


'''
guided bp
更加关注前景物体的梯度
'''
class GuidedReLU(Function):  # 继承Function类
    @staticmethod
    def forward(ctx, input):  # 与标准ReLU相同
        output = torch.clamp(input, min=0)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0]  # 以下为修改部分
        mask = (output > 0).float() * (grad_output > 0).float() # 计算I(x > 0) * I(grad_output > 0) * grad_output
        return mask * grad_output

# test
x = torch.randn(2, 3, requires_grad=True)
y = ReLU.apply(x)
loss = y.sum()
loss.backward()
print(x.grad)

# 叶子节点作为中间变量，导致计算图的梯度跟踪失败
a = torch.randn(2, 3, requires_grad=True)
for i in range(2):
    a = a ** 2 # 中间变量使用

loss = a.sum()
loss.backward()
print(a.grad)