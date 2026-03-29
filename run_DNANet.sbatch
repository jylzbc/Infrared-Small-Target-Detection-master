#!/bin/bash
#SBATCH --job-name=NNNet      # 任务名称
#SBATCH --time=48:00:00             # 最长运行时间
#SBATCH --ntasks=1                  # 总任务数
#SBATCH --cpus-per-task=4           # 每任务 CPU 核心数
#SBATCH --mem=100G                  # 节点内存
#SBATCH --gres=gpu:1                # 申请 1 块 GPU
#SBATCH --nodes=1                   # 申请 1 个节点
#SBATCH --output=logs/NUDT_NEWBS/32_%j.txt      # 标准输出/错误合并文件，%j=作业ID

# 打印 GPU 状态
nvidia-smi
echo "Allocated GPU devices: $CUDA_VISIBLE_DEVICES"

# 加载 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh   # 如路径不同请自行修改
conda activate dnanet_new                       # 替换为你的虚拟环境名
echo "Conda 环境已激活：$(which python)"

# 进入工程目录
cd ~/zyn/NNNet/Infrared-Small-Target-Detection-master

# 执行训练
python train.py \
    --base_size 256 \
    --crop_size 256 \
    --epochs 1500 \
    --dataset NUDT-SIRST \
    --split_method 50_50 \
    --model DNANet \
    --backbone resnet_18 \
    --deep_supervision True \
    --train_batch_size 32 \
    --test_batch_size 32 \
    --mode TXT

echo "Job completed successfully."
  