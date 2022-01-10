import torch
from torch import nn
from torchsummary import summary

'''
Encoder :
    
    - word embedding 词的稠密连续向量表示
    
    - position encoding  通过 sin/cos 进行位置表征（推广到更长的测试句子）；通过残差链接使得位置信息流入深层

    - multi-head self-attention 由多组 Q、K、V 构成，每组单独计算一个 attention 向量；采用 矩阵想成得到最终向量
    
    - FFN linear 层

Decoder :

    - word embedding
    
    - mask multi-head self-attention

    - FFN
    
    - softmax
    
    
- Multi-head attention    
         
         ^
         |
       MatMul
       ^     ^  
       |     |
    Softmax  |
       ^     |
       |     |
      Mask   |           将为 0 的位置 设为 非常小的数（-1e9）
       ^     | 
       |     |
     Scale   |            
       ^     |
       |     |
     MatMul  |            QK^T
     ^   ^   |
     |   |   |
     Q   K   V             Q/K/V - ecoder output 
     
     
    
'''

transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))
out = transformer_model(src, tgt)