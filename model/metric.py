import  numpy as np
import torch.nn as nn
import torch
from skimage import measure
import  numpy

# ROCMetric 类 — 计算 ROC 曲线与 Precision-Recall 曲线相关指标
class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):

        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)


        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])

# PD_FA 类 — 计算目标级 PD / FA 曲线
# PD：检测到的真实目标比例（目标级召回）
# FA：每幅图像中的误报率（每单位面积的错误检测比例）
class PD_FA():
    def __init__(self, nclass, bins):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins+1)
        self.PD = np.zeros(self.bins + 1)
        self.target= np.zeros(self.bins + 1)
    def update(self, preds, labels):

        for iBin in range(self.bins+1):
            score_thresh = iBin * (255/self.bins)
            predits  = np.array((preds > score_thresh).cpu()).astype('int64')
            predits  = np.reshape (predits,  (256,256))
            labelss = np.array((labels).cpu()).astype('int64') # P
            labelss = np.reshape (labelss , (256,256))

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss , connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin]    += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match   = []
            self.dismatch         = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        break

            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
            self.FA[iBin]+=np.sum(self.dismatch)
            self.PD[iBin]+=len(self.distance_match)

    def get(self,img_num):

        Final_FA =  self.FA / ((256 * 256) * img_num)
        Final_PD =  self.PD /self.target

        return Final_FA,Final_PD


    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])


class PD_FA_SingleValue(object):
    """
    计算红外小目标检测指标：
    PD（目标级探测率）= 正确检测的目标数 / 总目标数
    FA（像素级虚警率）= 虚警像素 / 总像素数
    根据文章描述修改FA计算逻辑：
    "If the centroid deviation of the target is larger than the predefined deviation threshold, 
     we consider those pixels as falsely predicted ones."
    """

    def __init__(self, centroid_thresh=3):
        """
        centroid_thresh: 判定预测目标正确的质心偏差阈值（单位：像素）
        """
        self.centroid_thresh = centroid_thresh
        self.reset()

    def reset(self):
        self.correct_targets = 0  # 正确检测的目标数
        self.total_targets = 0    # GT 总目标数
        self.FP = 0               # 虚警像素
        self.total_pixels = 0     # 总像素数

    def update(self, pred, target):
        """
        pred: 网络输出 logits 或概率, shape = [H, W] 或 [C, H, W]
        target: GT mask（0/1）
        """
        # 二值化预测结果
        pred_bin = (pred > 0.5).float()
        
        # 计算总像素数
        self.total_pixels += target.numel()

        # 找 GT 目标和预测目标
        target_np = target.cpu().numpy().astype(int)
        pred_np = pred_bin.cpu().numpy().astype(int)
        
        # 确保是2D图像
        if target_np.ndim == 3:
            target_np = target_np[0]
        if pred_np.ndim == 3:
            pred_np = pred_np[0]
            
        target_labels = measure.label(target_np, connectivity=2)
        pred_labels = measure.label(pred_np, connectivity=2)

        self.total_targets += np.max(target_labels)  # GT 目标总数
        
        # 记录哪些预测目标已被匹配
        matched_pred_targets = set()
        
        # ---- PD 计算（目标级） ----
        # 遍历 GT 目标，判断是否被预测正确
        for i in range(1, np.max(target_labels) + 1):
            gt_mask = (target_labels == i).astype(int)
            gt_props = measure.regionprops(gt_mask)
            if len(gt_props) == 0:
                continue
            gt_centroid = gt_props[0].centroid  # (row, col)

            detected = False
            for j in range(1, np.max(pred_labels) + 1):
                pred_mask = (pred_labels == j).astype(int)
                pred_props = measure.regionprops(pred_mask)
                if len(pred_props) == 0:
                    continue
                pred_centroid = pred_props[0].centroid

                # 欧氏距离判断
                dist = ((gt_centroid[0] - pred_centroid[0])**2 + 
                        (gt_centroid[1] - pred_centroid[1])**2)**0.5
                if dist <= self.centroid_thresh:
                    matched_pred_targets.add(j)
                    detected = True
                    break

            if detected:
                self.correct_targets += 1

        # ---- FA 计算（根据文章描述）----
        # "If the centroid deviation of the target is larger than the predefined deviation threshold,
        #  we consider those pixels as falsely predicted ones."
        # 意思是：只有未匹配的预测目标（质心偏差大于阈值）的像素才被视为虚警像素
        for j in range(1, np.max(pred_labels) + 1):
            if j not in matched_pred_targets:  # 未匹配的预测目标
                pred_mask = (pred_labels == j).astype(int)
                # 将整个预测目标区域的像素计为虚警像素
                self.FP += np.sum(pred_mask)

    def get(self):
        PD = self.correct_targets / (self.total_targets + 1e-6)
        FA = self.FP / (self.total_pixels + 1e-6)
        return PD, FA
    # ---------------------------------------------------
    # 辅助函数：计算 IoU
    # ---------------------------------------------------
    @staticmethod
    def _bbox_iou(boxA, boxB):
        yA1, xA1, yA2, xA2 = boxA
        yB1, xB1, yB2, xB2 = boxB

        inter_x1 = max(xA1, xB1)
        inter_y1 = max(yA1, yB1)
        inter_x2 = min(xA2, xB2)
        inter_y2 = min(yA2, yB2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        areaA = (xA2 - xA1) * (yA2 - yA1)
        areaB = (xB2 - xB1) * (yB2 - yB1)
        return inter_area / (areaA + areaB - inter_area)

# mIoU 类 — 计算像素级 mIoU 和 Pixel Accuracy
# mIoU + nIoU 类 — 返回 nIoU 和 IoU
class mIoU():

    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):

        # 原始代码保持不动
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)

        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

        # -----------------------------
        # 新增：每张图像的 IoU，用于 nIoU
        # -----------------------------
        iou_per_image = inter / (union + 1e-6)
        self.iou_list.append(iou_per_image)

    def get(self):

        # -----------------------------
        # 返回 nIoU（平均每张图 IoU）
        # -----------------------------
        nIoU = np.mean(self.iou_list)

        # -----------------------------
        # 返回 IoU（全局 IoU）
        # -----------------------------
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        IoU = IoU.mean()

        # 你原来返回 pixAcc, mIoU，现在替换为 nIoU, IoU
        return nIoU, IoU

    def reset(self):

        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        
        # 新增：保存 per-image IoU
        self.iou_list = []

# 辅助函数部分
# 计算 TP, FP, TN, FN。核心是用不同阈值二值化预测图。
def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos

# 统计像素级别的正确像素数与标注像素数。
# def batch_pix_accuracy(output, target):

#     if len(target.shape) == 3:
#         target = np.expand_dims(target.float(), axis=1)
#     elif len(target.shape) == 4:
#         target = target.float()
#     else:
#         raise ValueError("Unknown target dimension")

#     assert output.shape == target.shape, "Predict and Label Shape Don't Match"
#     predict = (output > 0).float()
#     pixel_labeled = (target > 0).float().sum()
#     pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()



#     assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
#     return pixel_correct, pixel_labeled


# 通过 numpy 直方图计算分割区域的交并比。
def batch_intersection_union(output, target, nclass):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union

def batch_pix_accuracy(output, target):
    """计算批量像素准确率"""
    # 确保输出和目标形状匹配
    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    
    # 二值化预测结果
    predict = (output > 0.5).float()
    target_binary = (target > 0.5).float()
    
    # 计算正确预测的像素数和总像素数
    pixel_correct = (predict == target_binary).float().sum()
    pixel_total = target.numel()  # 总像素数
    
    return pixel_correct, pixel_total