- 为什么diffusion model训练的时候需要1000 timesteps，推理时只需要几十步
    
    训练采用的逻辑是基于DDPM的马尔可夫链逻辑，完整执行从t到t+1时刻的扩散过程；推理时采用的是DDIM类似的采样方法，将公式转化为非马尔可夫链的形式，求解任意两个时刻之间的对应公式，因此根据该公式可以在sample过程中跨步。
    
- epsilon- prediction和v prediction
    
    epsilon-predction模型预测的是去噪步骤中的噪声，而v-prediction模型预测的是两个时刻之间的速度，可视化预测结果可以看到epsilon-prediction输出的是高斯噪声，而v-prediction输出的是带有一定语义信息的差分图像。DDPM采用的是epsilon-prediction，而flow-matching采用的是v-prediction
    
- sd为什么不能生成纯黑图
    
    纯黑图片代表均值为0，方差为0的均匀分布，而diffusion model的理论基础是将训练集拟定为正态分布，从一个标准正态分布逐步去噪；
    
    另外SD的组件VAE也是无法生存纯黑图片的（除非数值溢出），VAE从隐层空间采样高斯分布，会引入一个符合正态分布的随机噪声满足z = mean + var * noise，随机噪声的方差不可能为0
    
- 文本控制和图像控制强弱，或者如何解耦
- zero terminal snr是什么?
    
    扩散模型理论假定的是当时刻T足够大时，加噪过后的图片会是一个标准的正态分布，即信噪比为0的高斯噪声。但是如果我们将alpha_bar的数值打印出来，发现最后时刻的alpha_bar并不会归零，而是极其接近0的数字，根据公式$x_t=\sqrt{\alpha\_bar}*x_{0} + \sqrt{1-\alpha\_bar}*noise$，会残留部分输入图像的信息。这通常会导致生成图片的对比度偏低、亮度也偏低。所以zero terminal snr修改alpha_bar，将最后时刻T的alpha_bar置0。
    
- 训练时加入noise-offset为什么能生成对比度更高的图片
- 生成式VAE和原始VAE的区别
- classifier free guidance和classifier guidance的区别
- 条件信息可以通过什么方式加入到diffusion中进行去噪（类似DIT这种)