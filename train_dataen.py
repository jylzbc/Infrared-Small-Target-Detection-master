# # torch and visulization
# from tqdm             import tqdm
# import torch.optim    as optim
# from torch.optim      import lr_scheduler
# from torchvision      import transforms
# from torch.utils.data import DataLoader
# from model.parse_args_train import  parse_args

# # metric, loss .etc
# from model.utils import *
# from model.metric import *
# from model.loss import *
# from model.load_param_data import  load_dataset, load_param

# # model
# # from model.model_DNANet import  Res_CBAM_block
# from model.model_DNANet import  DNANet

# class Trainer(object):
#     def __init__(self, args):
#         # Initial
#         self.args = args
#         self.ROC  = ROCMetric(1, 10)
#         self.mIoU = mIoU(1)
#         self.save_prefix = '_'.join([args.model, args.dataset])
#         self.save_dir    = args.save_dir
#         nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

#         # Read image index from TXT
#         if args.mode == 'TXT':
#             dataset_dir = args.root + '/' + args.dataset
#             train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

#         # Preprocess and load data
#         input_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
#         trainset        = TrainSetLoader(dataset_dir,img_id=train_img_ids,base_size=args.base_size,crop_size=args.crop_size,transform=input_transform,suffix=args.suffix)
#         testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
#         self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers,drop_last=True)
#         self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

#         # Choose and load model (this paper is finished by one GPU)
#         if args.model   == 'DNANet':
#             model       = DNANet(num_classes=1,input_channels=args.in_channels, nb_filter=nb_filter, deep_supervision=args.deep_supervision)

#         model           = model.cuda()
#         model.apply(weights_init_xavier)
#         print("Model Initializing")
#         self.model      = model

#         # Optimizer and lr scheduling
#         if args.optimizer   == 'Adam':
#             self.optimizer  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
#         elif args.optimizer == 'Adagrad':
#             self.optimizer  = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
#         if args.scheduler   == 'CosineAnnealingLR':
#             self.scheduler  = lr_scheduler.CosineAnnealingLR( self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
#         self.scheduler.step()

#         # Evaluation metrics
#         self.best_iou       = 0
#         self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
#         self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

#     # Training
#     def training(self,epoch):

#         tbar = tqdm(self.train_data)
#         self.model.train()
#         losses = AverageMeter()
#         for i, ( data, labels) in enumerate(tbar):
#             data   = data.cuda()
#             labels = labels.cuda()
#             if args.deep_supervision == 'True':
#                 preds= self.model(data)
#                 loss = 0
#                 for pred in preds:
#                     loss += SoftIoULoss(pred, labels)
#                 loss /= len(preds)
#             else:
#                pred = self.model(data)
#                loss = SoftIoULoss(pred, labels)
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#             losses.update(loss.item(), pred.size(0))
#             tbar.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))
#         self.train_loss = losses.avg

#     # Testing
#     def testing (self, epoch):
#         tbar = tqdm(self.test_data)
#         self.model.eval()
#         self.mIoU.reset()
#         losses = AverageMeter()

#         with torch.no_grad():
#             for i, ( data, labels) in enumerate(tbar):
#                 data = data.cuda()
#                 labels = labels.cuda()
#                 if args.deep_supervision == 'True':
#                     preds = self.model(data)
#                     loss = 0
#                     for pred in preds:
#                         loss += SoftIoULoss(pred, labels)
#                     loss /= len(preds)
#                     pred =preds[-1]
#                 else:
#                     pred = self.model(data)
#                     loss = SoftIoULoss(pred, labels)
#                 losses.update(loss.item(), pred.size(0))
#                 self.ROC .update(pred, labels)
#                 self.mIoU.update(pred, labels)
#                 ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
#                 nIou, IOU = self.mIoU.get()
#                 tbar.set_description('Epoch %d, test loss %.4f, IoU: %.4f, nIoU: %.4f' %
#                                  (epoch, losses.avg, IOU, nIoU))
#             test_loss=losses.avg
#         # save high-performance model
#         save_model(IOU, nIoU, self.best_iou, self.save_dir, self.save_prefix,
#                 train_loss, test_loss, recall, precision, epoch, self.model.state_dict())



# def main(args):
#     trainer = Trainer(args)
#     for epoch in range(args.start_epoch, args.epochs):
#         trainer.training(epoch)
#         trainer.testing(epoch)


# if __name__ == "__main__":
#     args = parse_args()
#     main(args)


# 上面是最原版的训练代码=========================================================
# 下面是换掉损失函数的训练代码====================================================

# # torch and visulization
# from tqdm             import tqdm
# import torch.optim    as optim
# from torch.optim      import lr_scheduler
# from torchvision      import transforms
# from torch.utils. data import DataLoader
# from model. parse_args_train import  parse_args

# # metric, loss . etc
# from model.utils import *
# from model.metric import *
# from model.loss import *
# from model.load_param_data import  load_dataset, load_param

# from model.model_DNANet import  DNANet


# # -------------------------------
# # 修改后的 Trainer 类测试部分
# # -------------------------------
# class Trainer(object):
#     def __init__(self, args):
#         self.args = args
#         self.ROC  = ROCMetric(1, 10)
#         self.mIoU = mIoU(1)
#         self.PDFA = PD_FA_SingleValue()  # PD/FA 单值指标

#         self.save_prefix = '_'.join([args.model, args.dataset])
#         self.save_dir    = args.save_dir
#         nb_filter, num_blocks = load_param(args.channel_size, args.backbone)


#         # Load dataset
#         if args.mode == 'TXT':
#             dataset_dir = args.root + '/' + args.dataset
#             train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

#         input_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([.485, .456, .406], [.229, .224, .225])
#         ])
#         trainset = TrainSetLoader(dataset_dir, img_id=train_img_ids,
#                                   base_size=args.base_size, crop_size=args.crop_size,
#                                   transform=input_transform, suffix=args.suffix)
#         testset  = TestSetLoader(dataset_dir, img_id=val_img_ids,
#                                  base_size=args.base_size, crop_size=args. crop_size,
#                                  transform=input_transform, suffix=args.suffix)

#         self.train_data = DataLoader(trainset, batch_size=args. train_batch_size,
#                                      shuffle=True, num_workers=args.workers, drop_last=True)
#         self.test_data  = DataLoader(testset, batch_size=args.test_batch_size,
#                                      num_workers=args.workers, drop_last=False)

#         # Load model
#         if args.model == 'DNANet': 
#             model = DNANet(num_classes=1, input_channels=args.in_channels,
#                            nb_filter=nb_filter, deep_supervision=args. deep_supervision)

#         model = model.cuda()
#         model.apply(weights_init_xavier)
#         self.model = model

#         # Optimizer & Scheduler
#         if args.optimizer == 'Adam':
#             self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
#         elif args.optimizer == 'Adagrad':
#             self.optimizer = torch.optim.Adagrad(filter(lambda p: p. requires_grad, model.parameters()), lr=args.lr)

#         if args.scheduler == 'CosineAnnealingLR':
#             self.scheduler = lr_scheduler.CosineAnnealingLR(self. optimizer, T_max=args. epochs, eta_min=args. min_lr)
#         self.scheduler. step()

#         # ==================== 修改：初始化损失函数 ====================
#         self.loss_fn = CombinedIoUDiceLoss(soft_iou_weight=0.4, dice_weight=0.6)

#         self.best_iou = 0

#     # -------------------------------
#     # Training
#     # -------------------------------
#     def training(self, epoch):
#         tbar = tqdm(self. train_data)
#         self.model.train()
#         losses = AverageMeter()

#         for i, (data, labels) in enumerate(tbar):
#             data   = data.cuda()
#             labels = labels.cuda()

#             if self.args.deep_supervision == 'True':
#                 preds = self.model(data)
#                 loss = 0
#                 for pred in preds:
#                     loss += self. loss_fn(pred, labels)  # 修改：使用self.loss_fn
#                 loss /= len(preds)
#             else:
#                 pred = self.model(data)
#                 loss = self.loss_fn(pred, labels)  # 修改：使用self.loss_fn

#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()

#             losses.update(loss.item(), pred.size(0))
#             tbar.set_description(f'Epoch {epoch}, training loss {losses.avg:.4f}')

#         self.train_loss = losses.avg

#     # -------------------------------
#     # Testing
#     # -------------------------------
#     def testing(self, epoch):
#         tbar = tqdm(self.test_data)
#         self.model.eval()
#         self.mIoU.reset()
#         self. PDFA.reset()
#         losses = AverageMeter()

#         with torch.no_grad():
#             for i, (data, labels) in enumerate(tbar):
#                 data = data. cuda()
#                 labels = labels. cuda()

#                 if self.args.deep_supervision == 'True':
#                     preds = self.model(data)
#                     loss = 0
#                     for pred in preds:
#                         loss += self.loss_fn(pred, labels)  # 修改：使用self.loss_fn
#                     loss /= len(preds)
#                     pred = preds[-1]
#                 else:
#                     pred = self.model(data)
#                     loss = self. loss_fn(pred, labels)  # 修改：使用self.loss_fn

#                 losses.update(loss.item(), pred.size(0))

#                 self.ROC.update(pred, labels)
#                 self.mIoU.update(pred, labels)
#                 for b in range(pred.shape[0]):
#                     self. PDFA.update(pred[b,0], labels[b,0])

#                 _, _, recall, precision = self.ROC.get()
#                 nIoU, IoU = self.mIoU.get()
#                 pd, fa = self.PDFA.get()

#                 tbar. set_description(
#                     f'Epoch {epoch} | loss {losses.avg:.4f} | IoU {IoU:.4f} | nIoU {nIoU:.4f} | PD {pd:.4f} | FA {fa:.6f}'
#                 )

#             test_loss = losses.avg

#         if IoU > self.best_iou:
#             print(f"➡️  New best IoU: {IoU:.4f} (previous {self.best_iou:.4f})")

#             save_model(
#                 mean_IOU=IoU,
#                 nIoU=nIoU,
#                 best_iou=self.best_iou,   # 传旧值（关键）
#                 save_dir=self.save_dir,
#                 save_prefix=self.save_prefix,
#                 train_loss=self.train_loss,
#                 test_loss=test_loss,
#                 recall=recall,
#                 precision=precision,
#                 epoch=epoch,
#                 net=self.model,           # 传模型本体（关键）
#                 pd=pd,
#                 fa=fa
#             )

#             self.best_iou = IoU           # 最后更新



# def main(args):
#     trainer = Trainer(args)
#     for epoch in range(args.start_epoch, args.epochs):
#         trainer.training(epoch)
#         trainer.testing(epoch)


# if __name__ == "__main__":
#     args = parse_args()
#     main(args)


# 下面的代码是包含数据增强的
# torch and visualization
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from model. parse_args_train import parse_args
import numpy as np

# metric, loss . etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import load_dataset, load_param

# model
from model.model_DNANet import DNANet

# # ==================== 改进的损失函数（版本B） ====================

# class CombinedIoUDiceLoss(torch.nn.Module):
#     """
#     改进的组合损失函数：SoftIoU + Dice Loss
#     针对NUDT-SIRST红外小目标检测优化
#     """
#     def __init__(self, soft_iou_weight=0.4, dice_weight=0.6, smooth=1.0):
#         super().__init__()
#         self.soft_iou_weight = soft_iou_weight
#         self. dice_weight = dice_weight
#         self.smooth = smooth

#     def forward(self, pred, target):
#         pred = torch.sigmoid(pred)
        
#         # SoftIoU部分
#         intersection = pred * target
#         union = pred + target - intersection
#         soft_iou = (intersection. sum() + self.smooth) / (union.sum() + self.smooth)
#         soft_iou_loss = 1 - soft_iou. mean()
        
#         # Dice部分
#         dice_intersection = (pred * target).sum()
#         dice_score = (2.0 * dice_intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
#         dice_loss = 1.0 - dice_score
        
#         # 加权组合
#         combined_loss = self.soft_iou_weight * soft_iou_loss + self.dice_weight * dice_loss
        
#         return combined_loss


# ==================== 数据增强模块（版本B） ====================

class MixupCutmixWrapper(torch.nn.Module):
    """MixUp和CutMix的数据增强"""
    
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, 
                 mixup_prob=0.5, cutmix_prob=0.5):
        super().__init__()
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob

    def mixup_data(self, x, y):
        """执行MixUp增强"""
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index, :]
        
        return mixed_x, mixed_y

    def cutmix_data(self, x, y):
        """执行CutMix增强"""
        if self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        H, W = x.size(2), x.size(3)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        
        cx = np.random.randint(0, W)
        cy = np.random. randint(0, H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
        
        mixed_y = lam * y + (1 - lam) * y[index, :]
        
        return x, mixed_y

    def forward(self, x, y):
        """随机选择增强方式"""
        rand = np.random.rand()
        
        if rand < self.mixup_prob:
            x, y = self.mixup_data(x, y)
        elif rand < self.mixup_prob + self.cutmix_prob:
            x, y = self.cutmix_data(x, y)
        
        return x, y


# ==================== Warmup + CosineAnnealing学习率调整 ====================

def adjust_learning_rate_with_warmup(optimizer, epoch, total_epochs, 
                                     base_lr, warmup_epochs=10, min_lr=1e-5):
    """
    Warmup + CosineAnnealing学习率调整
    
    前10个epoch：线性warmup（从1e-4到base_lr）
    之后的epoch：余弦退火（从base_lr到min_lr）
    """
    if epoch < warmup_epochs:
        # 线性预热
        progress = epoch / warmup_epochs
        new_lr = 1e-4 + (base_lr - 1e-4) * progress
    else:
        # 余弦退火
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        new_lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
    
    for param_group in optimizer. param_groups:
        param_group['lr'] = new_lr
    
    return new_lr


# ==================== 改进的Trainer类 ====================

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.ROC = ROCMetric(1, 10)
        self.mIoU = mIoU(1)
        self.PDFA = PD_FA_SingleValue()

        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir = args.save_dir
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Load dataset
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
        trainset = TrainSetLoader(dataset_dir, img_id=train_img_ids,
                                  base_size=args.base_size, crop_size=args.crop_size,
                                  transform=input_transform, suffix=args.suffix)
        testset = TestSetLoader(dataset_dir, img_id=val_img_ids,
                               base_size=args.base_size, crop_size=args. crop_size,
                               transform=input_transform, suffix=args.suffix)

        self.train_data = DataLoader(trainset, batch_size=args.train_batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
        self.test_data = DataLoader(testset, batch_size=args.test_batch_size,
                                    num_workers=args.workers, drop_last=False)

        # Load model
        if args.model == 'DNANet': 
            model = DNANet(num_classes=1, input_channels=args.in_channels,
                           nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        model = model. cuda()
        model.apply(weights_init_xavier)
        self.model = model

        # Optimizer & Scheduler
        if args.optimizer == 'Adam': 
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        if args.scheduler == 'CosineAnnealingLR':
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        self.scheduler.step()


        # ==================== 版本B改进：初始化数据增强 ====================
        self.augmentor = MixupCutmixWrapper(
            mixup_alpha=0.2,
            cutmix_alpha=1.0,
            mixup_prob=0.5,
            cutmix_prob=0.5
        )
        print("✓ Data Augmentation:  MixUp + CutMix enabled")
        
        # ==================== 版本B改进：Deep Supervision权重 ====================
        self.deep_supervision_weights = [0.5, 0.5, 0.7, 1.0]
        
        self.best_iou = 0

    # ==================== 改进的训练函数（加入数据增强） ====================
    def training(self, epoch):
        """
        改进的训练函数
        - 添加Warmup + CosineAnnealing学习率调整
        - 添加数据增强（MixUp + CutMix）
        - 改用CombinedIoUDiceLoss
        - 梯度裁剪
        """
        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()

        # 调整学习率（Warmup + CosineAnnealing）
        lr = adjust_learning_rate_with_warmup(
            self.optimizer, epoch, self.args.epochs,
            self.args.lr, warmup_epochs=10, min_lr=self.args.min_lr
        )

        for i, (data, labels) in enumerate(tbar):
            data = data.cuda()
            labels = labels.cuda()

            # ==================== 数据增强（版本B改进） ====================
            data, labels = self.augmentor(data, labels)

            if self.args.deep_supervision == 'True':
                preds = self.model(data)
                loss = torch.tensor(0.0, device=data.device)
                for idx, pred in enumerate(preds):
                    weight = self.deep_supervision_weights[idx]
                    loss += weight * SoftIoULoss(pred, labels)
                loss = loss / sum(self.deep_supervision_weights)
            else:
                pred = self.model(data)
                loss = SoftIoULoss(pred, labels)

            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（版本B改进）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            losses.update(loss.item(), data.size(0))
            tbar.set_description(f'Epoch {epoch}, loss {losses.avg:.4f}, lr {lr:.6e}')

        self.train_loss = losses.avg

    # ==================== 改进的测试函数（使用改进的损失��数） ====================
    def testing(self, epoch):
        """改进的测试函数（使用CombinedIoUDiceLoss）"""
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        self.PDFA.reset()
        losses = AverageMeter()

        with torch.no_grad():
            for i, (data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()

                if self.args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = torch.tensor(0.0, device=data.device)
                    for idx, pred in enumerate(preds):
                        weight = self.deep_supervision_weights[idx]
                        loss += weight * SoftIoULoss(pred, labels)
                    loss = loss / sum(self.deep_supervision_weights)
                    pred = preds[-1]
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)

                losses.update(loss.item(), pred.size(0))

                self.ROC.update(pred, labels)
                self. mIoU.update(pred, labels)
                for b in range(pred.shape[0]):
                    self.PDFA.update(pred[b, 0], labels[b, 0])

                _, _, recall, precision = self.ROC. get()
                nIoU, IoU = self.mIoU.get()
                pd, fa = self.PDFA.get()

                tbar. set_description(
                    f'Epoch {epoch} | loss {losses.avg:.4f} | IoU {IoU:.4f} | nIoU {nIoU:.4f} | PD {pd:.4f} | FA {fa:.6f}'
                )

            test_loss = losses.avg

        if IoU > self.best_iou:
            print(f"➡️  New best IoU: {IoU:.4f} (previous {self.best_iou:.4f})")

            save_model(
                mean_IOU=IoU,
                nIoU=nIoU,
                best_iou=self.best_iou,
                save_dir=self.save_dir,
                save_prefix=self.save_prefix,
                train_loss=self.train_loss,
                test_loss=test_loss,
                recall=recall,
                precision=precision,
                epoch=epoch,
                net=self.model,
                pd=pd,
                fa=fa
            )

            self.best_iou = IoU


def main(args):
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)


if __name__ == "__main__":
    args = parse_args()
    main(args)