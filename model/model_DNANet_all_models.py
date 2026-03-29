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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self. conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


# ==================== FEDA模块（保持不变）====================

class FrequencyBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.hpf = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        kernel = torch.tensor([
            [-1.0, -1.0, -1.0], 
            [-1.0,  8.0, -1.0], 
            [-1.0, -1.0, -1.0]
        ]).float().view(1, 1, 3, 3) / 8.0
        with torch.no_grad():
            self.hpf.weight. copy_(kernel. repeat(channels, 1, 1, 1))
        self.hpf.weight.requires_grad = False
        self.enhance = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
    
    def forward(self, x):
        high_freq = self.hpf(x)
        attn = self.enhance(high_freq. abs())
        return x + attn * high_freq

class DynamicACC(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dilation1 = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(channels//2), nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels//2), nn.ReLU(inplace=True))
        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 8, 1), nn.ReLU(),
            nn.Conv2d(8, 2, 1), nn.Softmax(dim=1))
    
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
        self.skip = nn.Sequential(nn.Conv2d(in_features, filters, 1, bias=False), nn.BatchNorm2d(filters))
        self.c1 = conv_block(in_features, filters, 3, 1)
        self.c2 = conv_block(filters, filters, 3, 1)
        self.c3 = nn.Sequential(nn.Conv2d(filters, filters, 3, padding=1, bias=False), nn.BatchNorm2d(filters))
        self.freq_branch = FrequencyBranch(filters)
        self.dynamic_acc = DynamicACC(filters)
        self.lga = nn.Sequential(
            nn.Conv2d(filters, filters, 3, padding=1, groups=filters, bias=False),
            nn.BatchNorm2d(filters), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(filters, filters, 1), nn.Sigmoid())
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
            g_s3, g_lga, g_acc, g_freq = gates. chunk(4, dim=1)
            fused = g_s3 * s3 + g_lga * lga_feat + g_acc * acc_feat + g_freq * freq_feat
        else:
            combined = torch.cat([s3, lga_feat, acc_feat], dim=1)
            gates = self.fusion_gate(combined)
            g_s3, g_lga, g_acc = gates.chunk(3, dim=1)
            fused = g_s3 * s3 + g_lga * lga_feat + g_acc * acc_feat
        return self. final_relu(identity + fused)


# ==================== 核心创新：DSTBS ====================

class DualStreamTargetBackgroundSeparation(nn.Module):
    """
    双流目标-背景分离（Dual-Stream Target-Background Separation）
    
    【核心创新点】
    1. 显式建模目标流和背景流（独立的两套处理逻辑）
    2. 目标流：小感受野（3×3）+ 高频增强 → 适合小目标
    3. 背景流：大感受野（7×7）+ 低通抑制 → 适合背景
    4. 动态路由器：pixel-wise决定使用哪个流
    
    【与ACM的本质区别】
    - ACM:   权重不对称，但处理流程相同
    - DSTBS: 完全独立的两个处理流，针对目标/背景的不同特性
    
    【可视化】
    - 可以可视化router的输出，看哪里是目标（红色）、哪里是背景（蓝色）
    """
    def __init__(self, channels):
        super().__init__()
        
        # 【目标流】针对小、亮、高频的红外小目标
        self.target_stream = nn.Sequential(
            # 小感受野保留细节
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # 高频增强（检测边缘）
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        # 【背景流】针对大面积、平滑的背景
        self. background_stream = nn.Sequential(
            # 大感受野平滑背景
            nn.Conv2d(channels, channels, 7, padding=3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # 背景抑制因子
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn. Sigmoid()  # [0, 1]范围的抑制因子
        )
        
        # 【动态路由器】决定每个像素用哪个流
        # 基于局部统计量（均值、方差、梯度）判断是目标还是背景
        self.router = nn.Sequential(
            # 提取局部特征
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 1),
            nn.Softmax(dim=1)  # [target_weight, background_weight]
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn. Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        # 初始化router为平衡状态
        nn.init.constant_(self.router[-2].weight, 0)
        nn.init.constant_(self.router[-2].bias, 0)
        
    def forward(self, x):
        # 目标流处理
        target_feat = self.target_stream(x)
        
        # 背景流处理（带抑制）
        bg_suppression = self.background_stream(x)
        bg_feat = x * (1 - bg_suppression)  # 抑制背景
        
        # 动态路由（pixel-wise）
        weights = self.router(x)  # [B, 2, H, W]
        w_target = weights[:, 0:1, :, :]      # 目标权重
        w_bg = weights[:, 1:2, :, :]          # 背景权重
        
        # 加权融合
        routed = w_target * target_feat + w_bg * bg_feat
        
        # 最终融合
        out = self.fusion(routed)
        
        # 残差连接（较小权重保证稳定性）
        return x + 0.2 * out


# ==================== LMSA（稳定版）====================

class LightweightMultiScaleAggregation(nn.Module):
    def __init__(self, channels, adjacent_channels=None):
        super().__init__()
        self.adjacent_channels = adjacent_channels if adjacent_channels is not None else channels
        
        if self.adjacent_channels != channels:
            self.align_channels = nn.Sequential(
                nn.Conv2d(self.adjacent_channels, channels, 1, bias=False),
                nn. BatchNorm2d(channels)
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
        
        self. refine = nn.Sequential(
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
            
            if x_adjacent.shape[2: ] != x_current.shape[2:]:
                x_adjacent = F.interpolate(x_adjacent, size=x_current.shape[2:], 
                                          mode='bilinear', align_corners=True)
            
            attn = self.channel_attn(x_adjacent)
            x_adjacent = x_adjacent * attn
            
            gate_weight = self.gate(torch.cat([x_current, x_adjacent], dim=1))
            gate_weight = torch.clamp(gate_weight, 0.1, 0.9)
            x_fused = gate_weight * x_adjacent + (1 - gate_weight) * x_current
        else:
            x_fused = x_current
        
        x_refined = self. refine(x_fused)
        return x_current + 0.1 * x_refined


# ==================== ABE（稳定版）====================

class AdaptiveBoundaryEnhancement(nn.Module):
    def __init__(self):
        super().__init__()
        coarse_kernel = torch.tensor([
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [-1.0,  2.0,  2.0,  2.0, -1.0],
            [-1.0,  2.0,  8.0,  2.0, -1.0],
            [-1.0,  2.0,  2.0,  2.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0]
        ]).float().view(1, 1, 5, 5) / 16.0
        self.register_buffer('coarse_kernel', coarse_kernel)
        
        fine_kernel = torch.tensor([
            [-1.0, -1.0, -1.0],
            [-1.0,  8.0, -1.0],
            [-1.0, -1.0, -1.0]
        ]).float().view(1, 1, 3, 3) / 8.0
        self.register_buffer('fine_kernel', fine_kernel)
        
        self.enhance_net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
        for m in self.enhance_net. modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        coarse_boundary = F.conv2d(x, self.coarse_kernel, padding=2)
        fine_boundary = F.conv2d(x, self.fine_kernel, padding=1)
        
        coarse_boundary = torch.clamp(coarse_boundary, -10, 10)
        fine_boundary = torch.clamp(fine_boundary, -10, 10)
        
        combined = torch.cat([x, coarse_boundary. abs(), fine_boundary.abs()], dim=1)
        enhance_weight = self.enhance_net(combined)
        
        return x + 0.1 * enhance_weight * fine_boundary


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



# ==================== 主网络 ====================

class DNANet(nn.Module):
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

        # 【核心创新】DSTBS - 在每个尺度应用目标-背景分离
        self.dstbs1 = DualStreamTargetBackgroundSeparation(nb_filter[1])
        self.dstbs2 = DualStreamTargetBackgroundSeparation(nb_filter[2])
        self.dstbs3 = DualStreamTargetBackgroundSeparation(nb_filter[3])
        self.dstbs4 = DualStreamTargetBackgroundSeparation(nb_filter[4])

        # LMSA
        self.lmsa1 = LightweightMultiScaleAggregation(nb_filter[1], nb_filter[2])
        self.lmsa2 = LightweightMultiScaleAggregation(nb_filter[2], nb_filter[3])
        self.lmsa3 = LightweightMultiScaleAggregation(nb_filter[3], nb_filter[4])
        self.lmsa4 = LightweightMultiScaleAggregation(nb_filter[4], None)

        # Decoder
        self.conv3_1 = UNetBlock(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = UNetBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = UNetBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = UNetBlock(nb_filter[0] + nb_filter[1], nb_filter[0])

        # 输出头
        self.head1 = nn.Conv2d(nb_filter[1], 1, 1)
        self.head2 = nn.Conv2d(nb_filter[2], 1, 1)
        self.head3 = nn.Conv2d(nb_filter[3], 1, 1)
        self.head4 = nn.Conv2d(nb_filter[0], 1, 1)

        # ABE
        self.abe1 = AdaptiveBoundaryEnhancement()
        self.abe2 = AdaptiveBoundaryEnhancement()
        self.abe3 = AdaptiveBoundaryEnhancement()
        self.abe4 = AdaptiveBoundaryEnhancement()
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn. init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.shape[2:]

        # Encoder（FEDA）
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # 【应用DSTBS】目标-背景分离
        x1_0 = self.dstbs1(x1_0)
        x2_0 = self. dstbs2(x2_0)
        x3_0 = self.dstbs3(x3_0)
        x4_0 = self. dstbs4(x4_0)

        # LMSA（多尺度聚合）
        x1_0 = self.lmsa1(x1_0, x2_0)
        x2_0 = self.lmsa2(x2_0, x3_0)
        x3_0 = self.lmsa3(x3_0, x4_0)
        x4_0 = self.lmsa4(x4_0, None)

        # Decoder
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        # 输出
        p1 = self.head1(x1_3)
        p2 = self.head2(x2_2)
        p3 = self.head3(x3_1)
        p4 = self.head4(x0_4)

        # ABE（边界增强）
        p1 = self.abe1(p1)
        p2 = self.abe2(p2)
        p3 = self.abe3(p3)
        p4 = self.abe4(p4)

        # 插值
        p1 = F.interpolate(p1, size=input_size, mode='bilinear', align_corners=True)
        p2 = F.interpolate(p2, size=input_size, mode='bilinear', align_corners=True)
        p3 = F.interpolate(p3, size=input_size, mode='bilinear', align_corners=True)

        if self.deep_supervision:
            return [p1, p2, p3, p4]
        else: 
            return p4