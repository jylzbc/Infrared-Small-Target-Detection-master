import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    """标准卷积块：Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class DNANet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, nb_filter=[16, 32, 64, 128, 256], deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # --------------------------------------------------------
        # 编码器 (Encoder)
        # --------------------------------------------------------
        self.conv0_0 = UNetBlock(input_channels, nb_filter[0])       # [B, 16, 256, 256]
        self.conv1_0 = UNetBlock(nb_filter[0], nb_filter[1])         # [B, 32, 128, 128]
        self.conv2_0 = UNetBlock(nb_filter[1], nb_filter[2])         # [B, 64, 64, 64]
        self.conv3_0 = UNetBlock(nb_filter[2], nb_filter[3])         # [B, 128, 32, 32]
        self.conv4_0 = UNetBlock(nb_filter[3], nb_filter[4])         # [B, 256, 16, 16]
        
        # --------------------------------------------------------
        # 解码器 (Decoder)
        # --------------------------------------------------------
        self.conv3_1 = UNetBlock(nb_filter[3]+nb_filter[4], nb_filter[3])
        self.conv2_2 = UNetBlock(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv1_3 = UNetBlock(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv0_4 = UNetBlock(nb_filter[0]+nb_filter[1], nb_filter[0])
        
        # --------------------------------------------------------
        # Final Output
        # --------------------------------------------------------
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[1], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[2], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[3], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        # 获取输入尺寸，用于最后的上采样
        input_size = x.size()[2:]
        
        # --- Encoder ---
        x0_0 = self.conv0_0(x)                        # [B, 16, 256, 256]
        x1_0 = self.conv1_0(self.pool(x0_0))          # [B, 32, 128, 128]
        x2_0 = self.conv2_0(self.pool(x1_0))          # [B, 64, 64, 64]
        x3_0 = self.conv3_0(self.pool(x2_0))          # [B, 128, 32, 32]
        x4_0 = self.conv4_0(self.pool(x3_0))          # [B, 256, 16, 16]

        # --- Decoder ---
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))    # [B, 128, 32, 32]
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))    # [B, 64, 64, 64]
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))    # [B, 32, 128, 128]
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))    # [B, 16, 256, 256]

        if self.deep_supervision:
            output1 = self.final1(x1_3)
            output2 = self.final2(x2_2)
            output3 = self.final3(x3_1)
            output4 = self.final4(x0_4)
            
            # 关键修改：将深层监督的中间结果上采样到原图大小
            output1 = F.interpolate(output1, size=input_size, mode='bilinear', align_corners=True)
            output2 = F.interpolate(output2, size=input_size, mode='bilinear', align_corners=True)
            output3 = F.interpolate(output3, size=input_size, mode='bilinear', align_corners=True)
            # output4 本身就是原图大小，不需要处理
            
            return [output1, output2, output3, output4]
        else:
            return self.final(x0_4)