"""
评估脚本: 在测试集上评估训练好的模型

指标:
  - PCC (Pearson 相关系数)
  - R² (决定系数)
  - 预测值 vs 真实值散点图
  - epoch 训练曲线

用法:
  python evaluate.py --data dataset.xlsx --checkpoint checkpoints/best_model.pt
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from model import BasenjiModel
from data_utils import load_data


@torch.no_grad()
def predict(model, dataloader, device):
    """对 DataLoader 做推理, 返回所有预测值和真实值"""
    model.eval()
    all_preds = []
    all_targets = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_preds.extend(outputs.flatten().tolist())
        all_targets.extend(labels.tolist())

    return np.array(all_preds), np.array(all_targets)


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """绘制预测值 vs 真实值散点图"""
    # 转回原始 TPM 空间
    true_tpm = 2 ** y_true - 1
    pred_tpm = 2 ** y_pred - 1

    pcc, _ = pearsonr(y_pred, y_true)
    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(pred_tpm, true_tpm, s=0.3, alpha=0.5, color="steelblue")
    ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], "r--", linewidth=1, label="y = x")

    ax.set_xlabel("Predicted TPM", fontsize=12)
    ax.set_ylabel("True TPM", fontsize=12)
    ax.set_title(
        f"Test Set: Predicted vs True Expression\n"
        f"PCC = {pcc:.4f},  R² = {r2:.4f}",
        fontsize=13,
    )
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Scatter plot saved to: {save_path}")


def plot_history(history: dict, save_path: str):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(history["Valid Loss"]) + 1)

    ax1.plot(epochs, history["Valid Loss"], "b-o", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Valid Loss (MSE)")
    ax1.set_title("Validation Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["Valid PCC"], "r-o", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Valid PCC")
    ax2.set_title("Validation PCC")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Training curves saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Basenji model on test set"
    )
    parser.add_argument("--data", type=str, default="dataset.xlsx",
                        help="Path to dataset.xlsx")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory for output plots")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    train_loader, valid_loader, test_loader, y_train, y_valid, y_test = load_data(
        args.data, batch_size=args.batch_size
    )

    # 加载模型
    model = BasenjiModel().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from: {args.checkpoint}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Valid PCC: {checkpoint.get('valid_pcc', '?'):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 测试集评估
    print("\n--- Test Set Evaluation ---")
    y_test_pred, y_test_true = predict(model, test_loader, device)

    pcc, _ = pearsonr(y_test_pred, y_test_true)
    r2 = r2_score(y_test_true, y_test_pred)

    print(f"PCC: {pcc:.4f}")
    print(f"R²:  {r2:.4f}")

    # 散点图
    plot_scatter(
        y_test_true, y_test_pred,
        save_path=os.path.join(args.output_dir, "test_scatter.png"),
    )

    # 训练曲线 (如果 history 存在)
    if "history" in checkpoint:
        print("\nGenerating training curves...")
        plot_history(
            checkpoint["history"],
            save_path=os.path.join(args.output_dir, "training_curves.png"),
        )

    # 也评估验证集
    print("\n--- Validation Set ---")
    y_valid_pred, y_valid_true = predict(model, valid_loader, device)
    valid_pcc, _ = pearsonr(y_valid_pred, y_valid_true)
    valid_r2 = r2_score(y_valid_true, y_valid_pred)
    print(f"Valid PCC: {valid_pcc:.4f}")
    print(f"Valid R²:  {valid_r2:.4f}")

    # 训练集评估
    print("\n--- Training Set ---")
    y_train_pred, y_train_true = predict(model, train_loader, device)
    train_pcc, _ = pearsonr(y_train_pred, y_train_true)
    train_r2 = r2_score(y_train_true, y_train_pred)
    print(f"Train PCC: {train_pcc:.4f}")
    print(f"Train R²:  {train_r2:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
