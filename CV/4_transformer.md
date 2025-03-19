- self-Attention 讲解和代码
    - self-attention通过softmax(K @ Q / sqrt(d))计算自注意力权重图，显示每个元素和其他元素之间产生联系，实现上下文信息整合；K @ Q点积计算相关性，人脸识别和向量检索就是使用类似的方式计算两条向量之间的相关性，加上sqrt(d)是一个归一化操作，防止相关性随dim纬度增加而增大，softmax则将相关性权重转化为0-1的概率分布。
    - multi-head attention:  将纬度切割成独立的的N个部分，例如将2048纬度切成8个256纬度，这样可以让不同head之间学到不同的特征，使得特征表述更为丰富，以下是论文原文解释
  ```
  Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.
  ```
    - 手写代码
    ```PYTHON
    class Attention(nn.Module):
		def __init__(self, dim, num_head):
				self.q = nn.Linear(dim, dim)
				self.k = nn.Linear(dim, dim)
				self.v = nn.Linear(dim, dim)
				self.dim = dim
				self.num_head = num_head
				
				self.proj = nn.Linear(dim, dim)
				
		def forward(self, x)
				B, N, C = x.shape
				Q = self.q(x).reshape(B, -1， N, self.dim / self.num_head)
				K = self.k(x).reshape(B, -1， N, self.dim / self.num_head)
				V = self.v(x).reshape(B, -1， N, self.dim / self.num_head)
				
				# （B, num_head, N, head_dim) @ (B, num_head, head_dim, N)
				# -> (B, num_head, N, N)
				attn = F.softmax(Q @ K.transpose(-2, -1) / sqrt(self.dim), dim=-1)
				output = attn @ V # （B, num_head, N, N) * (B, num_head, N, head_dim)
				# (B, num_head, N, head_dim) -> (B, N, num_head, head_dim)
				output = output.transpose(1, 2).reshape(B, N, C)
				
				output = self.proj(output)
				return output
    ```
  
- Postional Embedding：自注意力机制是权重之和，当序列位置变换时结果也一致，所以不能表示位置信息，因此在序列中加入sincosin标识；每个dim都是一个sin/cos函数，随着位置sequence变化
  - Sinuous绝对位置编码代码
  ```python
  def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle = pos / np.power(10000, (2 * (i//2)) / d_model)
    
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])  # 偶数维度
   
    pe[:, 1::2] = np.cos(angle[:, 1::2])  # 奇数维度
    return pe
  ```
  - ROPE旋转位置编码，旋转位置编码将旋转矩阵应用到位置编码，用旋转矩阵表示向量之间的相对关系，是一种相对位置编码，相比绝对位置编码有更好的外推性
```python
# 生成旋转矩阵
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis

# 旋转位置编码计算
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```