# PyTorch 玉米基因表达量预测 (Basenji)

基于 Basenji 空洞残差卷积网络的玉米基因表达量预测模型，华中农业大学信息学院实验四。

## 简介

以玉米基因 TSS 附近 3kb DNA 序列的 one-hot 编码为输入，预测该基因的 RNA-seq 表达量（TPM）。模型复现自文献：

> Kelley DR, Reshef YA, Bileschi M, et al. *Sequential regulatory activity prediction across chromosomes with convolutional neural networks.* Genome Research, 2018.

## 网络结构

```
input(4, 3000)
  → ConvBlock (filters=8, ks=15, pool=2)          → (8, 1500)
  → ConvTower (filters 16→32, repeat=2, pool=2)   → (32, 375)
  → DilatedResidual (filters=16, repeat=2)         → (32, 375)
  → ConvBlock (1×1 conv)                           → (32, 375)
  → ConvBlock (1×1 conv)                           → (1, 375)
  → Flatten + Linear                               → (1)
```

总参数: **9,587**

## 数据集

| 项目 | 值 |
|------|-----|
| 基因数 | 37,979 |
| 序列长度 | 3,001 bp (含 N 占位符) |
| 标签 | TPM (log2 变换) |
| train/valid/test | 27,344 / 3,039 / 7,596 |

## 结果

| 数据集 | PCC | R² |
|--------|-----|-----|
| Train | 0.4880 | 0.0306 |
| Valid | 0.4868 | 0.0299 |
| Test | 0.4843 | 0.0264 |

训练硬件: NVIDIA GeForce RTX 4090D (24GB), batch_size=4096, 100 epochs

## 快速开始

```bash
git clone https://github.com/1034250765/maize-gene-expression.git
cd maize-gene-expression
pip install -r requirements.txt

# 训练
python train.py --data dataset.xlsx --epochs 100 --batch_size 4096 --lr 1e-3

# 评估
python evaluate.py --data dataset.xlsx --checkpoint checkpoints/best_model.pt
```

## 文件说明

```
maize-gene-expression/
├── model.py           # Basenji 模型 (GELU, ConvBlock, ConvTower, DilatedResidual, ...)
├── data_utils.py      # DNA one-hot 编码 + PyTorch DataLoader
├── train.py           # 训练脚本
├── evaluate.py        # 评估脚本 (PCC, R², 散点图)
├── generate_report.py # Word 实验报告生成器
├── requirements.txt   # 依赖
├── checkpoints/
│   └── best_model.pt  # 最佳模型权重 (Valid PCC=0.4868)
└── outputs/
    └── test_scatter.png
```

## 依赖

- torch >= 2.0
- numpy, pandas, scipy
- matplotlib, tqdm, einops
- scikit-learn, openpyxl
