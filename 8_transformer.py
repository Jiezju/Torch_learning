import torch
from torch import nn
from torchsummary import summary
from torch.nn import functional as F

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

# torch api
# transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
# src = torch.rand((10, 32, 512))
# tgt = torch.rand((20, 32, 512))
# out = transformer_model(src, tgt)

# manual (序列翻译)

# 序列构造

# 单词表大小
max_num_src_words = max_num_tgt_words = 8

# 序列的最大长度
max_src_seq_len = max_tgt_seq_len = 5

# embeding 特征大小
model_dim = 8

# 位置的最大长度
max_position_len = 5

# 两个序列长度
src_len = torch.Tensor([2, 4]).to(torch.int32)
tgt_len = torch.Tensor([4, 3]).to(torch.int32)

# 由单词索引构建的源句子和目标句子，并且padding 到等长序列
src_seq = torch.cat(
    [torch.unsqueeze(F.pad(torch.randint(1, max_num_src_words, (L,)), (0, max_src_seq_len - L)), 0)
                     for L in src_len])

tgt_seq = torch.cat(
    [torch.unsqueeze(F.pad(torch.randint(1, max_num_tgt_words, (L,)), (0, max_src_seq_len - L)), 0)
                     for L in src_len])

print(src_seq)
print(tgt_seq)

# word embeding +1 是为了 第 0 个用作padding
src_embeding_table = nn.Embedding(max_num_src_words + 1, model_dim)
tgt_embeding_table = nn.Embedding(max_num_tgt_words + 1, model_dim)

src_embedding = src_embeding_table(src_seq)
print(src_embeding_table.weight) # [9, 8]
print(src_seq) # [2, 5]
print(src_embedding) # [2, 5, 8]

# position embdding 采用 sin/cos 函数进行拟合 因为序列的语义 是线性相关的 采用 sin /cos 是可以相互线性表达的
# PE(pos, 2i) = sin(pos / 10000^(2i/model_dim))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/model_dim)) pos 表示 单词 在句子中的位置，i 表示 单词 在 单词表的位置
# broadcast
pos_mat = torch.arange(max_position_len).reshape((-1,1))
print(pos_mat.shape)
i_mat = torch.pow(10000, torch.arange(0,8,2).reshape((1,-1))/model_dim)
print(i_mat.shape)
pe_embedding_table = torch.zeros(max_position_len, model_dim)
pe_embedding_table[:, 0::2] = torch.sin(pos_mat / i_mat)
pe_embedding_table[:, 1::2] = torch.cos(pos_mat / i_mat)
print(pe_embedding_table.shape) # [5, 8]

pe_embedding = nn.Embedding(max_position_len, model_dim)
pe_embedding.weight = nn.Parameter(pe_embedding_table, requires_grad=False)

# 构建 位置 序列
src_pos = torch.cat([torch.unsqueeze(torch.arange(max(src_len)), 0) for _ in src_len]).to(torch.int32)
tgt_pos = torch.cat([torch.unsqueeze(torch.arange(max(tgt_len)), 0) for _ in tgt_len]).to(torch.int32)

src_pe_embedding = pe_embedding(src_pos)
tgt_pe_embedding = pe_embedding(tgt_pos)
print(src_pe_embedding.shape)
print(tgt_pe_embedding.shape)
