import os
import random

def split_dataset(img_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                  train_txt='train.txt', val_txt='val.txt', test_txt='test.txt',
                  img_exts=('.jpg', '.png', '.jpeg', '.bmp', '.tif')):
    """
    随机划分数据集，并将图片文件名（无后缀）写入 train/val/test 文本文件。
    """

    # 获取所有图片名（去掉后缀）
    all_imgs = []
    for f in os.listdir(img_dir):
        if f.lower().endswith(img_exts):
            name = os.path.splitext(f)[0]  # 去掉后缀
            all_imgs.append(name)

    if len(all_imgs) == 0:
        print("❌ 未找到任何图片，请检查数据集路径。")
        return

    # 随机打乱
    random.shuffle(all_imgs)

    # 计算数量
    total = len(all_imgs)
    num_train = int(total * train_ratio)
    num_val = int(total * val_ratio)
    num_test = total - num_train - num_val

    train_list = all_imgs[:num_train]
    val_list = all_imgs[num_train:num_train + num_val]
    test_list = all_imgs[num_train + num_val:]

    # 写入文件
    with open(train_txt, 'w') as f:
        f.write('\n'.join(train_list))

    with open(val_txt, 'w') as f:
        f.write('\n'.join(val_list))

    with open(test_txt, 'w') as f:
        f.write('\n'.join(test_list))

    print("✔ 数据集已成功划分！")
    print(f"  训练集: {len(train_list)}")
    print(f"  验证集: {len(val_list)}")
    print(f"  测试集: {len(test_list)}")


if __name__ == "__main__":
    # 修改你的图片目录路径
    img_dir = "/home/shenyujie/zyn/NNNet/Infrared-Small-Target-Detection-master/dataset/NUDT-SIRST/images"

    split_dataset(img_dir)
