
import torch
import torch.nn as nn
import numpy as np

"""
Reference: https://github.com/lpj0/MWCNN_PyTorch/tree/master
"""



def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


def iwt_init(x):
    r = 2
    
    x1, x2, x3, x4 = x[0], x[1], x[2],x[3]
    
    in_batch, in_channel, in_height, in_width = x1.size()
    # print([in_batch, in_channel, in_height, in_width]) #[1, 12, 56, 56]
    out_batch, out_channel, out_height, out_width = in_batch, in_channel, r * in_height, r * in_width
    # print(out_batch, out_channel, out_height, out_width) #1 3 112 112
    x1 = x1 / 2
    x2 = x2 / 2
    x3 = x3 / 2
    x4 = x4 / 2
    
    device = torch.device("cuda:0" if x1.is_cuda else "cpu")
    
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


# 离散小波变换
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

# 逆离散小波变换
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)
    
if __name__ == "__main__":
    inp = torch.rand(1, 1, 4, 4)  # 创建一个随机张量作为输入
    dwt = DWT()  # 实例化 DWT 类
    output = dwt(inp)  # 调用实例对象，进行前向传播

    print("Input:\n", inp)
    print("Output LL:\n", output[0])
    print("Output HL:\n", output[1])
    print("Output LH:\n", output[2])
    print("Output HH:\n", output[3])