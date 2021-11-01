## ResNet
- motivation。解决网络过深带来的梯度消失问题，通过恒等映射，数据可以直连任何后续层。

- 残差和恒等映射$x'=x+f(x)$
```python
def bottlenetck(self, x):
    x0 = x
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)

    x = x + x0 # 恒等映射
    x = self.relu(x)
    return x
```
![identity](cache/resnet.png)

## MobileNet-v2
- 特点：一般convgroup数等于channel数，这里group为1
