import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 基础组件 ====================

def conv_block(in_features, out_features, kernel_size=3, padding=1, norm_type='bn', activation=True):
    layers = [nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                        padding=padding, bias=False)]
    if norm_type == 'bn':
        layers.append(nn.BatchNorm2d(out_features))
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNetBlock(nn.Module):
    """改进的UNetBlock：深层细化（针对NUDT-SIRST的小目标特性）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 第二个卷积块
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 改进：深层细化 - 分离卷积（depth-wise）用于特征细化
        # 原理：分离卷积参数少，计算高效，适合细化已有特征
        self.dw_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1,
                                 groups=out_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(out_channels)

        # 点卷积（1×1）用于特征融合
        self.pw_conv = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 标准两层卷积
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # 深层细化路径：分离卷积 + 点卷积
        x_refined = self.relu(self.bn_dw(self.dw_conv(x)))
        x_refined = self.relu(self.bn_pw(self.pw_conv(x_refined)))

        # 残差连接：原特征 + 细化特征
        return x + 0.5 * x_refined


# ==================== FEDA模块 ====================

class FrequencyBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.hpf = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        kernel = torch.tensor([
            [-1.0, -1.0, -1.0],
            [-1.0, 8.0, -1.0],
            [-1.0, -1.0, -1.0]
        ]).float().view(1, 1, 3, 3) / 8.0
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
        self.dilation1 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(channels // 2), nn.ReLU(inplace=True)
        )
        self.dilation2 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels // 2), nn.ReLU(inplace=True)
        )
        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 8, 1), nn.ReLU(),
            nn.Conv2d(8, 2, 1), nn.Softmax(dim=1)
        )

    def forward(self, x):
        feat1 = self.dilation1(x)
        feat2 = self.dilation2(x)
        weights = self.weight_predictor(x)
        w1 = weights[:, 0:1, :, :]
        w2 = weights[:, 1:2, :, :]
        return torch.cat([w1 * feat1, w2 * feat2], dim=1)


class FEDA(nn.Module):
    def __init__(self, in_features, filters, scale_level=1):
        super().__init__()
        self.scale_level = scale_level
        self.use_freq = scale_level <= 3
        self.skip = nn.Sequential(
            nn.Conv2d(in_features, filters, 1, bias=False),
            nn.BatchNorm2d(filters)
        )
        self.c1 = conv_block(in_features, filters, 3, 1)
        self.c2 = conv_block(filters, filters, 3, 1)
        self.c3 = nn.Sequential(
            nn.Conv2d(filters, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters)
        )
        self.freq_branch = FrequencyBranch(filters)
        self.dynamic_acc = DynamicACC(filters)
        self.lga = nn.Sequential(
            nn.Conv2d(filters, filters, 3, padding=1, groups=filters, bias=False),
            nn.BatchNorm2d(filters), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(filters, filters, 1), nn.Sigmoid()
        )
        gate_in = filters * (4 if self.use_freq else 3)
        gate_out = 4 if self.use_freq else 3
        self.fusion_gate = nn.Sequential(nn.Conv2d(gate_in, gate_out, 1), nn.Softmax(dim=1))
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        s1 = self.c1(x)
        s2 = self.c2(s1)
        s3 = self.c3(s2)
        acc_feat = self.dynamic_acc(s2)
        lga_feat = s3 * self.lga(s1)
        if self.use_freq:
            freq_feat = self.freq_branch(s3)
            combined = torch.cat([s3, lga_feat, acc_feat, freq_feat], dim=1)
            gates = self.fusion_gate(combined)
            g_s3, g_lga, g_acc, g_freq = gates.chunk(4, dim=1)
            fused = g_s3 * s3 + g_lga * lga_feat + g_acc * acc_feat + g_freq * freq_feat
        else:
            combined = torch.cat([s3, lga_feat, acc_feat], dim=1)
            gates = self.fusion_gate(combined)
            g_s3, g_lga, g_acc = gates.chunk(3, dim=1)
            fused = g_s3 * s3 + g_lga * lga_feat + g_acc * acc_feat
        return self.final_relu(identity + fused)


# ==================== 改进的DSTBS（核心创新：跨流交互） ====================

class DualStreamTargetBackgroundSeparation(nn.Module):
    """
    改进的双流目标-背景分离

    针对NUDT-SIRST特性的设计：
    1. 目标流：小感受野保留目标细节
    2. 背景流：大感受野平滑背景
    3. 跨流交互：让两个流相互增强
    4. 自适应权重：动态调整融合比例

    改进相比原DSTBS：
    - 添加跨流交互模块（交互而不是并行）
    - 自适应权重从2个改为3个（目标、背景、交互）
    - 更强的残差连接（0.4而非0.2）
    """
    def __init__(self, channels):
        super().__init__()

        # 目标流：小感受野（3×3）保留细节，适合小目标
        self.target_stream = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # 背景流：大感受野（5×5）平滑背景，抑制背景干扰
        self.background_stream = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(channels)
        )

        # 跨流交互模块：让目标流和背景流相互增强
        # 原理：通过融合两个流的特征，提高鲁棒性
        self.cross_interaction = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # 自适应权重预测器：三路融合权重
        # 根据输入特征自适应地调整目标、背景、交互的融合比例
        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1),
            nn.Softmax(dim=1)
        )

        # ✅ 正确的初始化方式
        for m in self.weight_predictor.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 目标流处理
        target_feat = self.target_stream(x)

        # 背景流处理
        bg_feat = self.background_stream(x)

        # 跨流交互：两个流的特征融合
        # 这样做的好处：目标和背景不再完全独立，能相互增强
        interaction_feat = self.cross_interaction(torch.cat([target_feat, bg_feat], dim=1))

        # 自适应权重预测（三路）
        weights = self.weight_predictor(x)  # [B, 3, 1, 1]
        w_target = weights[:, 0:1, :, :]
        w_bg = weights[:, 1:2, :, :]
        w_inter = weights[:, 2:3, :, :]

        # 加权融合三个特征
        fused = w_target * target_feat + w_bg * bg_feat + w_inter * interaction_feat

        # 残差连接，权重为0.4
        return x + 0.4 * fused


# ==================== LMSA（轻量级多尺度聚合） ====================

class LightweightMultiScaleAggregation(nn.Module):
    def __init__(self, channels, adjacent_channels=None):
        super().__init__()
        self.adjacent_channels = adjacent_channels if adjacent_channels is not None else channels

        if self.adjacent_channels != channels:
            self.align_channels = nn.Sequential(
                nn.Conv2d(self.adjacent_channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels)
            )
        else:
            self.align_channels = None

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 8, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 8, 1), channels, 1),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, 1, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate[0].weight, 0)
        nn.init.constant_(self.gate[0].bias, 0)

    def forward(self, x_current, x_adjacent=None):
        if x_adjacent is not None:
            if self.align_channels is not None:
                x_adjacent = self.align_channels(x_adjacent)

            if x_adjacent.shape[2:] != x_current.shape[2:]:
                x_adjacent = F.interpolate(
                    x_adjacent, size=x_current.shape[2:],
                    mode='bilinear', align_corners=True
                )

            attn = self.channel_attn(x_adjacent)
            x_adjacent = x_adjacent * attn

            gate_weight = self.gate(torch.cat([x_current, x_adjacent], dim=1))
            gate_weight = torch.clamp(gate_weight, 0.1, 0.9)
            x_fused = gate_weight * x_adjacent + (1 - gate_weight) * x_current
        else:
            x_fused = x_current

        x_refined = self.refine(x_fused)
        return x_current + 0.1 * x_refined


# ==================== 主网络 DNANet（版本B：完整改进版） ====================

class DNANet(nn.Module):
    """
    改进的DNANet - 版本B（针对NUDT-SIRST红外小目标检测）

    改进内容：
    1. UNetBlock深层细化：分离卷积+点卷积用于特征细化
    2. DSTBS加跨流交互：目标和背景流相互增强
    3. 改进的损失函数：在训练脚本中替换为SoftIoU+Dice组合
    4. 学习率调度：Warmup+CosineAnnealing
    5. 数据增强：MixUp和CutMix（在dataloader中实现）
    """
    def __init__(self, num_classes=1, input_channels=1, nb_filter=[16, 32, 64, 128, 256], deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder（FEDA）
        self.conv0_0 = FEDA(input_channels, nb_filter[0], scale_level=1)
        self.conv1_0 = FEDA(nb_filter[0], nb_filter[1], scale_level=2)
        self.conv2_0 = FEDA(nb_filter[1], nb_filter[2], scale_level=3)
        self.conv3_0 = FEDA(nb_filter[2], nb_filter[3], scale_level=4)
        self.conv4_0 = FEDA(nb_filter[3], nb_filter[4], scale_level=5)

        # self.conv0_0 = UNetBlock(input_channels, nb_filter[0])
        # self.conv1_0 = UNetBlock(nb_filter[0], nb_filter[1])
        # self.conv2_0 = UNetBlock(nb_filter[1], nb_filter[2])
        # self.conv3_0 = UNetBlock(nb_filter[2], nb_filter[3])
        # self.conv4_0 = UNetBlock(nb_filter[3], nb_filter[4])

        # 改进的DSTBS（含跨流交互）
        self.dstbs1 = DualStreamTargetBackgroundSeparation(nb_filter[1])
        self.dstbs2 = DualStreamTargetBackgroundSeparation(nb_filter[2])
        self.dstbs3 = DualStreamTargetBackgroundSeparation(nb_filter[3])
        self.dstbs4 = DualStreamTargetBackgroundSeparation(nb_filter[4])

        # LMSA
        self.lmsa1 = LightweightMultiScaleAggregation(nb_filter[1], nb_filter[2])
        self.lmsa2 = LightweightMultiScaleAggregation(nb_filter[2], nb_filter[3])
        self.lmsa3 = LightweightMultiScaleAggregation(nb_filter[3], nb_filter[4])
        self.lmsa4 = LightweightMultiScaleAggregation(nb_filter[4], None)

        # Decoder（使用改进的UNetBlock）
        # self.conv3_1 = UNetBlock(nb_filter[3] + nb_filter[4], nb_filter[3])
        # self.conv2_2 = UNetBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        # self.conv1_3 = UNetBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        # self.conv0_4 = UNetBlock(nb_filter[0] + nb_filter[1], nb_filter[0])

        self.conv3_1 = FEDA(nb_filter[3] + nb_filter[4], nb_filter[3], scale_level=4)
        self.conv2_2 = FEDA(nb_filter[2] + nb_filter[3], nb_filter[2], scale_level=3)
        self.conv1_3 = FEDA(nb_filter[1] + nb_filter[2], nb_filter[1], scale_level=2)
        self.conv0_4 = FEDA(nb_filter[0] + nb_filter[1], nb_filter[0], scale_level=1)

        # 输出头
        self.head1 = nn.Conv2d(nb_filter[1], 1, 1)
        self.head2 = nn.Conv2d(nb_filter[2], 1, 1)
        self.head3 = nn.Conv2d(nb_filter[3], 1, 1)
        self.head4 = nn.Conv2d(nb_filter[0], 1, 1)

        # ABE（边界增强）
        # self.abe1 = AdaptiveBoundaryEnhancement()
        # self.abe2 = AdaptiveBoundaryEnhancement()
        # self.abe3 = AdaptiveBoundaryEnhancement()
        # self.abe4 = AdaptiveBoundaryEnhancement()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            # 只初始化可训练的卷积层，避免覆盖 FrequencyBranch 的固定HPF
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.shape[2:]

        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # 应用改进的DSTBS
        x1_0 = self.dstbs1(x1_0)
        x2_0 = self.dstbs2(x2_0)
        x3_0 = self.dstbs3(x3_0)
        x4_0 = self.dstbs4(x4_0)

        # LMSA
        x1_0 = self.lmsa1(x1_0, x2_0)
        x2_0 = self.lmsa2(x2_0, x3_0)
        x3_0 = self.lmsa3(x3_0, x4_0)
        x4_0 = self.lmsa4(x4_0, None)

        # Decoder
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        # 输出头
        p1 = self.head1(x1_3)
        p2 = self.head2(x2_2)
        p3 = self.head3(x3_1)
        p4 = self.head4(x0_4)

        # 插值到原始尺寸
        p1 = F.interpolate(p1, size=input_size, mode='bilinear', align_corners=True)
        p2 = F.interpolate(p2, size=input_size, mode='bilinear', align_corners=True)
        p3 = F.interpolate(p3, size=input_size, mode='bilinear', align_corners=True)

        if self.deep_supervision:
            return [p1, p2, p3, p4]
        else:
            return p4