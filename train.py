# torch and visulization
from tqdm             import tqdm
import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader
from model.parse_args_train import  parse_args

# metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

from model.model_DNANet import  DNANet


# -------------------------------
# 修改后的 Trainer 类测试部分
# -------------------------------
class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.ROC  = ROCMetric(1, 10)
        self.mIoU = mIoU(1)
        self.PDFA = PD_FA_SingleValue()  # PD/FA 单值指标

        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = args.save_dir
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
        testset  = TestSetLoader(dataset_dir, img_id=val_img_ids,
                                 base_size=args.base_size, crop_size=args.crop_size,
                                 transform=input_transform, suffix=args.suffix)

        self.train_data = DataLoader(trainset, batch_size=args.train_batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
        self.test_data  = DataLoader(testset, batch_size=args.test_batch_size,
                                     num_workers=args.workers, drop_last=False)

        # Load model
        if args.model == 'DNANet':
            model = DNANet(num_classes=1, input_channels=args.in_channels,
                           nb_filter=nb_filter, deep_supervision=args.deep_supervision)

        model = model.cuda()
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

        self.best_iou = 0

    # -------------------------------
    # Training
    # -------------------------------
    def training(self, epoch):
        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()

        for i, (data, labels) in enumerate(tbar):
            data   = data.cuda()
            labels = labels.cuda()

            if self.args.deep_supervision == 'True':
                preds = self.model(data)
                loss = 0
                for pred in preds:
                    loss += SoftIoULoss(pred, labels)
                loss /= len(preds)
            else:
                pred = self.model(data)
                loss = SoftIoULoss(pred, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), pred.size(0))
            tbar.set_description(f'Epoch {epoch}, training loss {losses.avg:.4f}')

        self.train_loss = losses.avg

    # -------------------------------
    # Testing
    # -------------------------------
    def testing(self, epoch):
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
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                    loss /= len(preds)
                    pred = preds[-1]
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)

                losses.update(loss.item(), pred.size(0))

                self.ROC.update(pred, labels)
                self.mIoU.update(pred, labels)
                for b in range(pred.shape[0]):
                    self.PDFA.update(pred[b,0], labels[b,0])

                _, _, recall, precision = self.ROC.get()
                nIoU, IoU = self.mIoU.get()
                pd, fa = self.PDFA.get()

                tbar.set_description(
                    f'Epoch {epoch} | loss {losses.avg:.4f} | IoU {IoU:.4f} | nIoU {nIoU:.4f} | PD {pd:.4f} | FA {fa:.6f}'
                )

            test_loss = losses.avg

        if IoU > self.best_iou:
            print(f"➡️  New best IoU: {IoU:.4f} (previous {self.best_iou:.4f})")

            save_model(
                mean_IOU=IoU,
                nIoU=nIoU,
                best_iou=self.best_iou,   # 传旧值（关键）
                save_dir=self.save_dir,
                save_prefix=self.save_prefix,
                train_loss=self.train_loss,
                test_loss=test_loss,
                recall=recall,
                precision=precision,
                epoch=epoch,
                net=self.model,           # 传模型本体（关键）
                pd=pd,
                fa=fa
            )

            self.best_iou = IoU           # 最后更新



def main(args):
    seed_pytorch(args.seed)
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)


if __name__ == "__main__":
    args = parse_args()
    main(args)

