# Basic module
from tqdm import tqdm
from model.parse_args_test import parse_args
import scipy.io as scio
import os
import shutil
from PIL import Image
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch  # ensure torch is imported

# Torch and visulization
from torchvision import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from model.utils import *
from model.metric import *  # 确保包含 PD_FA_SingleValue 类
from model.loss import *
from model.load_param_data import load_dataset, load_param

# Model
from model.model_DNANet import  DNANet

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args = args
        self.ROC = ROCMetric(1, args.ROC_thr)
        self.PD_FA = PD_FA(1, 10)  # 原有多阈值PD/FA（保留）
        self.PDFA = PD_FA_SingleValue()  # 单值PD/FA（阈值0.5，与Mask一致）
        self.mIoU = mIoU(1)  # 你的mIoU类：nIoU和iou都是单值
        self.save_prefix = '_'.join([args.model, args.dataset])

        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir = os.path.join(args.root, args.dataset)
            train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset = TestSetLoader(dataset_dir, img_id=val_img_ids, base_size=args.base_size, crop_size=args.crop_size,
                                transform=input_transform, suffix=args.suffix)
        self.test_data = DataLoader(dataset=testset, batch_size=args.test_batch_size, num_workers=args.workers,
                                    drop_last=False)

        # Choose and load model
        if args.model == 'DNANet':
            model = DNANet(num_classes=1, input_channels=args.in_channels, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        model = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model = model

        # Evaluation metrics
        self.best_recall = [0] * 11
        self.best_precision = [0] * 11

        # Checkpoint
        checkpoint_path = os.path.join(args.model_dir)
        checkpoint = torch.load(checkpoint_path)
        
        # ============ 原有：统一测试结果保存路径（保留） ============
        self.test_result_base = "/home/shenyujie/zyn/NNNet/Infrared-Small-Target-Detection-master/test-results/NUDT_UNet"
        
        # 创建可视化结果子目录（保留）
        target_image_path = os.path.join(self.test_result_base, 'visulization_result', f'{args.st_model}_visulization_result')
        target_dir = os.path.join(self.test_result_base, 'visulization_result', f'{args.st_model}_visulization_fuse')
        make_visulization_dir(target_image_path, target_dir)
        
        self.target_image_path = target_image_path
        self.val_img_ids = val_img_ids  # 用于图像命名
        # ============================================

        # Load trained model
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()
        self.mIoU.reset()
        self.ROC.reset()
        self.PDFA.reset()
        tbar = tqdm(self.test_data, desc="Testing")
        losses = AverageMeter()
        with torch.no_grad():
            num = 0
            for i, (data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = sum([SoftIoULoss(preds[j], labels) for j in range(len(preds))]) / len(preds)
                    pred = preds[-1]
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)

                # ---------------------------
                # 原有：保存三类图 + GT 图（完全保留，无任何修改）
                # ---------------------------
                pred_np = pred.cpu().numpy().squeeze()
                labels_np = labels.cpu().numpy().squeeze()
                
                # 1️⃣ Mask 图（阈值0.5）
                mask = (pred_np > 0.5).astype(np.uint8) * 255
                mask_img = Image.fromarray(mask)
                mask_img.save(os.path.join(self.target_image_path, f'{val_img_ids[num]}_mask.png'))

                # 1.5️⃣ GT图（保存为二值 0/255）
                gt_mask = (labels_np > 0.5).astype(np.uint8) * 255
                gt_img = Image.fromarray(gt_mask)
                gt_img.save(os.path.join(self.target_image_path, f'{val_img_ids[num]}_gt.png'))

                # 2️⃣ 概率图（概率值可视化）
                prob = (np.clip(pred_np, 0, 1) * 255).astype(np.uint8)
                prob_img = Image.fromarray(prob)
                prob_img.save(os.path.join(self.target_image_path, f'{val_img_ids[num]}_prob.png'))

                # 3️⃣ 彩色叠加图（原图灰度 + 热力图）
                orig_img = data.cpu().numpy().squeeze().transpose(1, 2, 0)
                orig_img = ((orig_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406]))
                orig_img = np.clip(orig_img * 255, 0, 255).astype(np.uint8)
                if orig_img.ndim == 3 and orig_img.shape[2] == 3:
                    orig_gray = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
                else:
                    orig_gray = orig_img if orig_img.ndim == 2 else orig_img[:, :, 0]
                orig_gray = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR)
                heatmap = cv2.applyColorMap(prob, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(orig_gray, 0.6, heatmap, 0.4, 0)
                cv2.imwrite(os.path.join(self.target_image_path, f'{val_img_ids[num]}_overlay.png'), overlay)
                # ---------------------------

                num += 1

                # ---------------------------
                # Metrics update（新增单值PD/FA更新，原有指标保留）
                # ---------------------------
                losses.update(loss.item(), pred.size(0))
                self.ROC.update(pred, labels)
                self.mIoU.update(pred, labels)
                self.PD_FA.update(pred, labels)  # 原有多阈值PD/FA（保留）
                
                # 新增：单值PD/FA更新（与Mask图阈值0.5一致）
                for b in range(pred.shape[0]):
                    pred_bin = (pred[b, 0] > 0.5).float()  # 二值化
                    self.PDFA.update(pred_bin, labels[b, 0])

                # 更新进度条
                tbar.set_postfix(loss=losses.avg)

            # ---------------------------
            # 记录测试结果（完全按你的需求：所有指标都是单值）
            # ---------------------------
            # 1. 原有ROC指标（recall/precision是数组，函数支持，保留）
            ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
            
            # 2. IoU指标（你的mIoU类返回：nIoU=单值，iou=单值）
            nIoU, iou = self.mIoU.get()  # 完全按你的类逻辑，都是单值！
            
            # 3. 原有多阈值PD/FA（保留，用于保存.mat文件）
            FA_multi, PD_multi = self.PD_FA.get(len(self.val_img_ids))
            
            # 4. 新增单值PD/FA（单值）
            single_PD, single_FA = self.PDFA.get()
            
            # 5. 测试损失（单值）
            test_loss = losses.avg

            # ============ 原有：保存多阈值PD/FA（完全保留，无修改） ============
            value_result_dir = os.path.join(self.test_result_base, 'value_result')
            os.makedirs(value_result_dir, exist_ok=True)
            scio.savemat(os.path.join(value_result_dir, f'{args.st_model}_PD_FA_255.mat'),
                         {'number_record1': FA_multi, 'number_record2': PD_multi})
            # ============================================

            
            # ============================================
            
            # ============ 终端输出（只显示4个核心单值指标，无任何数组） ============
            print('\n' + '='*60)
            print(f'测试结果汇总 - 模型：{args.st_model}')
            print('='*60)
            print(f'test_loss: {test_loss:.4f}')
            print(f'iou: {iou:.4f}')          # 你的iou单值
            print(f'nIoU: {nIoU:.4f}')        # 你的公式计算的单值nIoU
            print(f'single_PD: {single_PD:.4f}')
            print(f'single_FA: {single_FA:.8f}')
            print('='*60 + '\n')
            # ============================================

            # ============ 调用 save_result_for_test（参数全是单值，完全匹配函数） ============
            save_result_for_test(
                base_dir=self.test_result_base,
                model_name=args.st_model,
                epoch=args.epochs,
                iou=iou,           # 单值
                nIoU=nIoU,         # 单值（你的公式结果）
                pd=single_PD,      # 单值
                fa=single_FA,      # 单值
                recall=recall,     # 原有数组（函数支持）
                precision=precision# 原有数组（函数支持）
            )
            # ============================================


def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)