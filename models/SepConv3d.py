import torch
from torch import nn


class SepConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(SepConv3d, self).__init__()
        self.conv_s = nn.Conv3d(in_planes, out_planes, kernel_size=(1, kernel_size, kernel_size),
                                stride=(1, stride, stride), padding=(0, padding, padding), bias=False)
        self.bn_s = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_s = nn.ReLU()

        self.conv_t = nn.Conv3d(out_planes, out_planes, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1),
                                padding=(padding, 0, 0), bias=False)
        self.bn_t = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_t = nn.ReLU()

    def forward(self, x):
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu_s(x)

        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.relu_t(x)
        return x


# 简洁创建SepConv3d实例的函数
def SepConv(in_planes, out_planes, kernel_size, stride, padding=0):
    return SepConv3d(in_planes, out_planes, kernel_size, stride, padding)


# 示例使用
if __name__ == "__main__":
    # 假设输入为batch_size=1, channels=3, depth=16, height=112, width=112
    input_tensor = torch.randn(1, 3, 16, 112, 112)

    # 使用 SepConv 创建 SepConv3d 模块
    sep_conv_layer = nn.Sequential(
        SepConv(3, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True)
    )

    output_tensor = sep_conv_layer(input_tensor)
    print(output_tensor.shape)  # 打印输出张量的形状
