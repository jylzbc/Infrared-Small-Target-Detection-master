import torch.nn as nn
from torch.nn import BatchNorm2d
import torch
# 融合来自不同层（高层和低层）的特征图，使信息交互更充分，同时利用注意力机制提高特征选择性。
class AsymBiChaFuse(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AsymBiChaFuse, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)

        # top-down 分支（高层特征调制低层特征）最终得到 topdown_wei，表示高层特征对每个通道的重要性。
        self.topdown = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_channels=self.channels, out_channels=self.bottleneck_channels, kernel_size=1, stride=1,padding=0),
        nn.BatchNorm2d(self.bottleneck_channels,momentum=0.9),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=self.bottleneck_channels,out_channels=self.channels,  kernel_size=1, stride=1,padding=0),
        nn.BatchNorm2d(self.channels,momentum=0.9),
        nn.Sigmoid()
        )

        # bottom-up 分支（低层特征调制高层特征）最终得到 bottomup_wei，表示低层特征对高层特征的调制强度。
        self.bottomup = nn.Sequential(
        nn.Conv2d(in_channels=self.channels,out_channels=self.bottleneck_channels, kernel_size=1, stride=1,padding=0),
        nn.BatchNorm2d(self.bottleneck_channels,momentum=0.9),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=self.bottleneck_channels,out_channels=self.channels, kernel_size=1, stride=1,padding=0),
        nn.BatchNorm2d(self.channels,momentum=0.9),
        nn.Sigmoid()
        )

        # 融合后的卷积操作，进一步处理融合后的特征图。
        self.post = nn.Sequential(
        nn.Conv2d(in_channels=channels,out_channels=channels, kernel_size=3, stride=1, padding=1, dilation=1),
        nn.BatchNorm2d(channels,momentum=0.9),
        nn.ReLU(inplace=True)
        )

    def forward(self, xh, xl):

        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * torch.mul(xl, topdown_wei) + 2 * torch.mul(xh, bottomup_wei)
        xs = self.post(xs)
        return xs
    
    """
    
    实现了 双向注意力引导的跨层特征融合；

    相比传统直接拼接或加权融合，更加自适应；

    常用于编码器-解码器结构（如 U-Net、HCFNet、DNANet 等）的跳跃连接中。
    """