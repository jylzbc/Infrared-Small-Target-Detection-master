import torch.nn as nn
import numpy as np
import  torch

def SoftIoULoss( pred, target):
        # Old One
        pred = torch.sigmoid(pred)
        smooth = 1

        # print("pred.shape: ", pred.shape)
        # print("target.shape: ", target.shape)

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)

        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss
# 通用的指标追踪器，在训练循环中使用。辅助统计
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np


# # ==================== 原始的 SoftIoU Loss ====================

# class SoftIoULoss(nn.Module):
#     """
#     原始SoftIoU Loss - 保持稳定性
#     IoU = |A∩B| / |A∪B|
#     """
#     def __init__(self):
#         super().__init__()
#         self.smooth = 1.0

#     def forward(self, pred, target):
#         pred = torch.sigmoid(pred)
#         intersection = pred * target
#         union = pred + target - intersection
#         loss = (intersection. sum() + self.smooth) / (union.sum() + self.smooth)
#         loss = 1 - loss. mean()
#         return loss


# # ==================== Dice Loss ====================

# class DiceLoss(nn.Module):
#     """
#     Dice Loss
#     Dice = 2|A∩B| / (|A| + |B|)
    
#     特点：
#     - 对小目标更敏感
#     - 关注精准度和召回率的平衡
#     - 特别适合NUDT-SIRST的小目标检测
#     """
#     def __init__(self, smooth=1.0):
#         super().__init__()
#         self.smooth = smooth

#     def forward(self, pred, target):
#         pred = torch.sigmoid(pred)
        
#         intersection = (pred * target).sum()
#         dice_score = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
#         dice_loss = 1.0 - dice_score
        
#         return dice_loss


# # ==================== 改进的组合Loss（推荐用于NUDT-SIRST） ====================

# class CombinedIoUDiceLoss(nn.Module):
#     """
#     改进方案：SoftIoU + Dice Loss的加权组合
    
#     【设计原理】
#     1. SoftIoU：全局的交集/并集比例
#        - 关注整体的检测准确率
#        - 对背景误检有约束
    
#     2. Dice Loss：局部的精准度和召回率
#        - 关注目标边界的清晰度
#        - 对小目标漏检更敏感
    
#     3. 组合：相互补充
#        - SoftIoU保证全局，Dice保证局部
#        - 特别适合红外小目标检测
    
#     【针对NUDT-SIRST的权重设置】
#     - soft_iou_weight=0.4：SoftIoU权重较低（不过度惩罚全局）
#     - dice_weight=0.6：Dice权重较高（强化小目标检测）
    
#     【预期效果】
#     - 原SoftIoU:  baseline
#     - 改进Loss: +0.5-1.5%
#     """
#     def __init__(self, soft_iou_weight=0.4, dice_weight=0.6, smooth=1.0):
#         super().__init__()
#         self.soft_iou_weight = soft_iou_weight
#         self.dice_weight = dice_weight
#         self.smooth = smooth

#     def forward(self, pred, target):
#         pred = torch.sigmoid(pred)
        
#         # SoftIoU部分
#         intersection = pred * target
#         union = pred + target - intersection
#         soft_iou = (intersection.sum() + self.smooth) / (union.sum() + self.smooth)
#         soft_iou_loss = 1 - soft_iou. mean()
        
#         # Dice部分
#         dice_intersection = (pred * target).sum()
#         dice_score = (2.0 * dice_intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
#         dice_loss = 1.0 - dice_score
        
#         # 加权组合
#         combined_loss = self.soft_iou_weight * soft_iou_loss + self.dice_weight * dice_loss
        
#         return combined_loss


# # ==================== 进阶Loss：焦点组合（可选） ====================

# class FocalCombinedLoss(nn.Module):
#     """
#     焦点组合Loss：针对困难样本和小目标
    
#     【适用场景】
#     - 小目标占比很小
#     - 背景很复杂
#     - 需要更强的监督信号
    
#     【改进点】
#     - 添加Focal Loss项，关注困难样本
#     - 可选使用，如果CombinedIoUDiceLoss效果已经很好，不必使用
#     """
#     def __init__(self, soft_iou_weight=0.4, dice_weight=0.5, focal_weight=0.1, 
#                  smooth=1.0, gamma=2.0):
#         super().__init__()
#         self.soft_iou_weight = soft_iou_weight
#         self.dice_weight = dice_weight
#         self. focal_weight = focal_weight
#         self.smooth = smooth
#         self.gamma = gamma

#     def forward(self, pred, target):
#         pred = torch.sigmoid(pred)
#         pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        
#         # SoftIoU
#         intersection = pred * target
#         union = pred + target - intersection
#         soft_iou = (intersection.sum() + self.smooth) / (union.sum() + self.smooth)
#         soft_iou_loss = 1 - soft_iou.mean()
        
#         # Dice
#         dice_intersection = (pred * target).sum()
#         dice_score = (2.0 * dice_intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
#         dice_loss = 1.0 - dice_score
        
#         # Focal部分（关注困难样本）
#         bce = F.binary_cross_entropy(pred, target, reduction='none')
#         pt = torch.where(target == 1, pred, 1 - pred)
#         focal = (1 - pt) ** self.gamma * bce
#         focal_loss = focal.mean()
        
#         # 加权组合
#         combined_loss = (self.soft_iou_weight * soft_iou_loss + 
#                         self.dice_weight * dice_loss + 
#                         self.focal_weight * focal_loss)
        
#         return combined_loss


# # ==================== 工具类 ====================

# class AverageMeter(object):
#     """计算和存储平均值"""
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self. count += n
#         self.avg = self.sum / self.count