"""
训练脚本: Basenji 模型用于玉米基因表达量预测

训练流程:
  1. 加载数据 (one-hot 编码)
  2. 构建 Basenji 模型
  3. 训练循环 (MSELoss + Adam)
  4. 每个 epoch 后在验证集上计算 Loss 和 PCC
  5. 保存最佳模型

用法:
  python train.py --data dataset.xlsx --epochs 10 --batch_size 32 --lr 1e-3
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from scipy.stats import pearsonr

from model import BasenjiModel
from data_utils import load_data


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """训练一个 epoch"""
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(dataloader, unit="step")

    for step, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.reshape(labels.shape), labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(labels)
        pbar.set_description(f"Train Loss: {loss.item():.4f}")

    return epoch_loss / len(dataloader.dataset)


@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    """在验证集上评估: Loss 和 PCC"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs.reshape(labels.shape), labels)
        total_loss += loss.item() * len(labels)

        all_preds.extend(outputs.flatten().tolist())
        all_targets.extend(labels.tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    pcc, _ = pearsonr(all_preds, all_targets)
    return avg_loss, pcc, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(description="Train Basenji model for maize gene expression")
    parser.add_argument("--data", type=str, default="dataset.xlsx",
                        help="Path to dataset.xlsx")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    args = parser.parse_args()

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    train_loader, valid_loader, test_loader, y_train, y_valid, y_test = load_data(
        args.data, batch_size=args.batch_size
    )

    # 构建模型
    model = BasenjiModel().to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # 训练历史
    history = {"Valid Loss": [], "Valid PCC": []}
    best_pcc = -1.0

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

        # 验证
        valid_loss, valid_pcc, _, _ = validate(model, valid_loader, loss_fn, device)

        history["Valid Loss"].append(valid_loss)
        history["Valid PCC"].append(valid_pcc)

        print(
            f"[{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f}, "
            f"Valid Loss: {valid_loss:.4f}, "
            f"Valid PCC: {valid_pcc:.4f}"
        )

        # 保存最佳模型
        if valid_pcc > best_pcc:
            best_pcc = valid_pcc
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "valid_pcc": valid_pcc, "valid_loss": valid_loss},
                os.path.join(args.save_dir, "best_model.pt"),
            )
            print(f"  → Best model saved (PCC: {best_pcc:.4f})")

    # 保存最终模型和历史
    torch.save(
        {"epoch": args.epochs, "model_state_dict": model.state_dict(),
         "history": history},
        os.path.join(args.save_dir, "final_model.pt"),
    )
    print(f"\nTraining complete. Best Valid PCC: {best_pcc:.4f}")
    print(f"Models saved to: {args.save_dir}/")


if __name__ == "__main__":
    main()
