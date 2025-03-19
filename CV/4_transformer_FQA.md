- Transformer为什么使用多头注意力机制
    - 论文中说明了多头注意力机制输出的特征会比单头更加丰富，而且多个特征一起决策可以让结果更加鲁棒
- Transformer为什么Q和K使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘?
    - QK点乘是为了得到一个二维的attention map，使用不同的权重矩阵可以拉开QK的相似性，使得表征更加丰富。假设使用自身值进行点积那得到的会是一个对称矩阵，这样其中一半特征就浪费了，所以自身点积是不合算的行为
- Transformer计算attention的时候为何选择点乘而不是加法?两者计算复杂度和效果上有什么区别?
    - 点乘和加法作为attention机制都是可行的，当时主要的attention有add-attention和dot-product attention两种，都在文章中提及了
    - 加法和乘法的计算复杂度是一致的都是O(N)，但因为矩阵乘法的优化，在实际应用中乘法会更快
    
    ```
    While the two are similar in theoretical complexity, dot-product attention is
    much faster and more space-efficient in practice, since it can be implemented using highly optimized
    matrix multiplication code
    ```
    
- 为什么在进行softmax之前需要对attention进行scaled，并使用公式推导进行讲解
    - 不进行scale就没有做归一化，attention score会随着维度提高，点乘的结果太大，使得softmax函数的梯度较小
    
    ```
    While for small values of dk the two mechanisms perform similarly, additive attention outperforms
    dot product attention without scaling for larger values of dk [3]. We suspect that for large values of
    dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has
    extremely small gradients 4. To counteract this effect, we scale the dot products by √1dk 
    ```
- 在计算attention score的时候如何对padding做mask操作
- 为什么在进行多头注意力的时候需要对每个head进行降维?
    - 降维是为了保证计算量不增加，多头和单头的计算量是完全一致的。假设有8个头，就将单头的维度平均拆分为8分。
    
    ```
    In this work we employ h = 8 parallel attention layers, or heads. For each of these we use
    dk = dv = dmodel/h = 64. Due to the reduced dimension of each head, the total computational cost
    is similar to that of single-head attention with full dimensionality.
    ```
    
- 大概讲-下Transformer的Encoder模块和Decoder模块?
- 简单介绍-下Transformer的位置编码?有什么意义和优缺点
    - self-attention是向量的加权求和结果，打乱向量的顺序结果依然相同，所以需要增加位置编码用以表示序列先后信息。缺点是位置编码后，对不同长度的序列信息处理能力会变差，目前rope具有良好的外推能力，但对于超出训练长度太多的结果依然较差。
- 你还了解哪些关于位置编码的技术，各自的优缺点是什么?
    - 可学习的pe、sin/cos、rope
- 简单讲-下Transformer中的残差结构以及意义
    - 残差结构是借鉴resnet的，主要是为了解决网络过深导致的梯度消失，有了残差结构网络才能到几十层。
    - 残差结构有效的最简单解释就是恒等映射直连，从输入到输出不管多深都有一条直连的线，由于恒等映射的存在，当某些层学习不太好时，也不影响最终结果，大大降低了学习难度
- 为什么transformer块使用LayerNorm而不是BatchNorm，谈一谈LN，BN, RMS Norm
    - 为什么用normalization：输入数据在分布上可能存在很大的差异，导致训练不稳定，将数据归一化减均值除方差，再通过alpha和beta平移缩放到一个统一的分布上，有助于稳定收敛
    - 真正的决定因素是实验结果，在视觉识别任务中两者的最终结果是相似的，许多基础网络结构的文章都探讨过BN和LN的实验结果，例如ConvNeXt把BN替换成LN版本，效果略微提高一点点。
    - BN有自己的缺陷，在batch-size较小或者等于1时训练会不稳定，对于语义分割任务之前就受到BN影响
    - BN, LN的实现是基本相似的，都是对数据减均值除方差，BN的统计纬度是batch，而LN的统计纬度是dim，可学习的参数有alpha和beta
    
    ```python
    # model train
    mean = x.mean(-1)
    std = x.std(-1)
    x = alpha * (x - mean) / (std + eps) + beta # alpha和beta是可学习参数
    
    # model eval, 推理是bn使用的是训练时保存的参数moving mean和moving std，bn总共有四个参数
    # ln则只有两个参数
    mean = moving_mean
    std = moving_std
    ```