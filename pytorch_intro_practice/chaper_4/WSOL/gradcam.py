import torch


def get_layer(model, key_list):
    a = model
    for key in key_list:
        a = a._modules[key]
    return a


class GradCAM(object):
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = get_layer(
            model, target_layer
        )
        # 用来存储输出响应
        self.activations = []
        # 用来存储梯度
        self.gradients = []
        # 对某一层添加 hook 函数
        self.register_hooks()

    def register_hooks(self):
        # 前向 记录输出
        def forward_hook(module, input, output):
            self.activations.append(output.data.clone())

        # 反向记录梯度
        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].data.clone())

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def reset(self):
        self.activations.clear()
        self.gradients.clear()

    def generate_saliency_map(self, input, label):
        self.reset()

        output = self.model(input)
        # 对输出 以 one hot 形式反向传播
        grad_output = output.data.clone()
        grad_output.zero_()
        grad_output.scatter_(1, label.unsqueeze(0).t(), 1.0)
        output.backward(grad_output)

        # 获取激活
        act = self.activations[0]
        # 获取梯度
        grad = self.gradients[0]

        weight = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weight * act).sum(dim=1, keepdim=True)
        cam = torch.clamp(cam, min=0)

        return cam

