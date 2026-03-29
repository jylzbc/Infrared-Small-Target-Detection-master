import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 1. 基础组件 (保持标准) ====================

def conv_block(in_features, out_features, kernel_size=3, padding=1, norm_type='bn', activation=True):
    layers = [nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                        padding=padding, bias=False)]
    if norm_type == 'bn':
        layers.append(nn.BatchNorm2d(out_features))
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class UNetBlock(nn.Module): 
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

# ==================== 2. 新增：CSA 模块 (基于论文逻辑) ====================

class ImprovedCCA(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.gate = nn.Sequential(nn.Linear(channels * 2, 1), nn.Sigmoid())

    def forward(self, Fx, Fg):
        b, c, _, _ = Fx.size()
        vec_x = self.gap(Fx).view(b, c)
        vec_g = self.gap(Fg).view(b, c)
        w_x, w_g = self.mlp(vec_x), self.mlp(vec_g)
        gate_val = self.gate(torch.cat([vec_x, vec_g], dim=1))
        alpha = torch.sigmoid(gate_val * w_x + (1 - gate_val) * w_g).view(b, c, 1, 1)
        return F.relu(Fx * alpha)

class ImprovedSMA(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.matchers = nn.Parameter(torch.randn(num_heads, 1, self.head_dim))
        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x_reshaped = x.view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)
        similarity = F.cosine_similarity(x_reshaped, self.matchers.unsqueeze(0), dim=-1) 
        avg_weight = similarity.mean(dim=1).view(b, 1, h, w)
        return self.project(x * avg_weight)

class CSA_Module(nn.Module):
    """
    集成的 CSA 模块：结合通道交叉注意力与空间匹配
    """
    def __init__(self, channels):
        super().__init__()
        self.cca = ImprovedCCA(channels)
        self.sma = ImprovedSMA(channels)

    def forward(self, Fx, Fg):
        f_fused = self.cca(Fx, Fg)
        return self.sma(f_fused)

# ==================== 3. 你的创新组件 (FEDA, FrequencyBranch等) ====================

class FrequencyBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.hpf = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).float().view(1, 1, 3, 3) / 8.0
        with torch.no_grad():
            self.hpf.weight.copy_(kernel.repeat(channels, 1, 1, 1))
        self.hpf.weight.requires_grad = False 
        self.enhance = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
    
    def forward(self, x):
        high_freq = self.hpf(x)
        attn = self.enhance(high_freq.abs())
        return x + attn * high_freq

class DynamicACC(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dilation1 = nn.Sequential(nn.Conv2d(channels, channels//2, 3, padding=1, dilation=1, bias=False), nn.BatchNorm2d(channels//2), nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(nn.Conv2d(channels, channels//2, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(channels//2), nn.ReLU(inplace=True))
        self.weight_predictor = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 8, 1), nn.ReLU(), nn.Conv2d(8, 2, 1), nn.Softmax(dim=1))
    
    def forward(self, x):
        feat1, feat2 = self.dilation1(x), self.dilation2(x)
        weights = self.weight_predictor(x)
        return torch.cat([weights[:, 0:1] * feat1, weights[:, 1:2] * feat2], dim=1)

class FEDA(nn.Module):
    def __init__(self, in_features, filters, scale_level=1):
        super().__init__()
        self.scale_level = scale_level
        self.use_freq = scale_level <= 3
        self.skip = nn.Sequential(nn.Conv2d(in_features, filters, 1, bias=False), nn.BatchNorm2d(filters))
        self.c1 = conv_block(in_features, filters, 3, 1)
        self.c2 = conv_block(filters, filters, 3, 1)
        self.c3 = nn.Sequential(nn.Conv2d(filters, filters, 3, padding=1, bias=False), nn.BatchNorm2d(filters))
        self.freq_branch = FrequencyBranch(filters)
        self.dynamic_acc = DynamicACC(filters)
        self.lga = nn.Sequential(nn.Conv2d(filters, filters, 3, padding=1, groups=filters, bias=False), nn.BatchNorm2d(filters), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1), nn.Conv2d(filters, filters, 1), nn.Sigmoid())
        gate_in = filters * (4 if self.use_freq else 3)
        self.fusion_gate = nn.Sequential(nn.Conv2d(gate_in, 4 if self.use_freq else 3, 1), nn.Softmax(dim=1))
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        s1, s2 = self.c1(x), self.c2(self.c1(x))
        s3 = self.c3(s2)
        acc_feat, lga_feat = self.dynamic_acc(s2), s3 * self.lga(s1)
        if self.use_freq:
            freq_feat = self.freq_branch(s3)
            combined = torch.cat([s3, lga_feat, acc_feat, freq_feat], dim=1)
            g_s3, g_lga, g_acc, g_freq = self.fusion_gate(combined).chunk(4, dim=1)
            fused = g_s3 * s3 + g_lga * lga_feat + g_acc * acc_feat + g_freq * freq_feat
        else:
            combined = torch.cat([s3, lga_feat, acc_feat], dim=1)
            g_s3, g_lga, g_acc = self.fusion_gate(combined).chunk(3, dim=1)
            fused = g_s3 * s3 + g_lga * lga_feat + g_acc * acc_feat
        return self.final_relu(identity + fused)


class FAM(nn.Module):
    def __init__(self, channels):
        super(FAM, self).__init__()
        # 深度可分离卷积：groups=channels 要求输入通道必须等于 channels
        self.align = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        return self.align(x)

# ----------------------------------------------------------------
# 核心组件：CAFF (对比度感知融合)
# ----------------------------------------------------------------
class CAFF(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CAFF, self).__init__()
        self.sa = SpatialAttention() 
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PA(dim)
        self.lcb = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1)
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x_shallow, x_deep):
        # 此时要求 x_shallow 和 x_deep 通道数都是 dim
        raw = x_shallow + x_deep
        cattn = self.ca(raw)
        sattn = self.sa(raw)
        contrast = self.sigmoid(self.lcb(raw)) 
        mix_att = (sattn + cattn) * contrast
        new_weight = self.sigmoid(self.pa(raw, mix_att))
        out = raw + new_weight * x_shallow + (1 - new_weight) * x_deep
        return self.conv_out(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(self.avg_pool(x))

class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, 1)
    def forward(self, raw, att):
        return self.pa_conv(raw * att)



# ==================== 4. 主网络结构 (集成 CSA) ====================

class DNANet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, nb_filter=[16, 32, 64, 128, 256], deep_supervision=False):
        super(DNANet, self).__init__()
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder (FEDA)
        self.conv0_0 = FEDA(input_channels, nb_filter[0], scale_level=1)
        self.conv1_0 = FEDA(nb_filter[0], nb_filter[1], scale_level=2)
        self.conv2_0 = FEDA(nb_filter[1], nb_filter[2], scale_level=3)
        self.conv3_0 = FEDA(nb_filter[2], nb_filter[3], scale_level=4)
        self.conv4_0 = FEDA(nb_filter[3], nb_filter[4], scale_level=5)

        # Decoder Components + CSA Module 集成
        # Level 3
        self.adapter3 = nn.Conv2d(nb_filter[4], nb_filter[3], kernel_size=1)
        self.fam3_s, self.fam3_d = FAM(nb_filter[3]), FAM(nb_filter[3])
        self.caff3 = CAFF(nb_filter[3])
        self.csa3 = CSA_Module(nb_filter[3]) # <--- 新增 CSA
        self.conv3_1 = UNetBlock(nb_filter[3], nb_filter[3])

        # Level 2
        self.adapter2 = nn.Conv2d(nb_filter[3], nb_filter[2], kernel_size=1)
        self.fam2_s, self.fam2_d = FAM(nb_filter[2]), FAM(nb_filter[2])
        self.caff2 = CAFF(nb_filter[2])
        self.csa2 = CSA_Module(nb_filter[2]) # <--- 新增 CSA
        self.conv2_2 = UNetBlock(nb_filter[2], nb_filter[2])

        # Level 1
        self.adapter1 = nn.Conv2d(nb_filter[2], nb_filter[1], kernel_size=1)
        self.fam1_s, self.fam1_d = FAM(nb_filter[1]), FAM(nb_filter[1])
        self.caff1 = CAFF(nb_filter[1])
        self.csa1 = CSA_Module(nb_filter[1]) # <--- 新增 CSA
        self.conv1_3 = UNetBlock(nb_filter[1], nb_filter[1])

        # Level 0
        self.adapter0 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1)
        self.fam0_s, self.fam0_d = FAM(nb_filter[0]), FAM(nb_filter[0])
        self.caff0 = CAFF(nb_filter[0])
        self.csa0 = CSA_Module(nb_filter[0]) # <--- 新增 CSA
        self.conv0_4 = UNetBlock(nb_filter[0], nb_filter[0])

        # Output Heads
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[1], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[2], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[3], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.size()[2:]
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Level 3
        up4 = self.adapter3(self.up(x4_0))
        f3_init = self.caff3(self.fam3_s(x3_0), self.fam3_d(up4))
        f3 = self.csa3(f3_init, x3_0) # CSA 处理融合特征
        x3_1 = self.conv3_1(f3)

        # Level 2
        up3 = self.adapter2(self.up(x3_1))
        f2_init = self.caff2(self.fam2_s(x2_0), self.fam2_d(up3))
        f2 = self.csa2(f2_init, x2_0) # CSA 处理融合特征
        x2_2 = self.conv2_2(f2)

        # Level 1
        up2 = self.adapter1(self.up(x2_2))
        f1_init = self.caff1(self.fam1_s(x1_0), self.fam1_d(up2))
        f1 = self.csa1(f1_init, x1_0) # CSA 处理融合特征
        x1_3 = self.conv1_3(f1)

        # Level 0
        up1 = self.adapter0(self.up(x1_3))
        f0_init = self.caff0(self.fam0_s(x0_0), self.fam0_d(up1))
        f0 = self.csa0(f0_init, x0_0) # CSA 处理融合特征
        x0_4 = self.conv0_4(f0)

        if self.deep_supervision:
            out1 = F.interpolate(self.final1(x1_3), size=input_size, mode='bilinear', align_corners=True)
            out2 = F.interpolate(self.final2(x2_2), size=input_size, mode='bilinear', align_corners=True)
            out3 = F.interpolate(self.final3(x3_1), size=input_size, mode='bilinear', align_corners=True)
            out4 = F.interpolate(self.final4(x0_4), size=input_size, mode='bilinear', align_corners=True)
            return [out1, out2, out3, out4]
        else:
            out = self.final(x0_4)
            return F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)

# 注意：请确保你的代码库中已定义 FAM 和 CAFF 模块，否则会报错。