"""
数据预处理: DNA 序列 one-hot 编码, 数据集加载

one-hot 编码: 将每种碱基 (A/T/C/G) 映射为四维向量
  A/a → [1,0,0,0]
  C/c → [0,1,0,0]
  G/g → [0,0,1,0]
  T/t → [0,0,0,1]
  N/n → [0,0,0,0] (保留全零, 不做处理)

输入: (4, 3000) 的 one-hot 矩阵
标签: log2(TPM + 1), 来源于 RNA-seq 实验数据
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# One-hot 编码: 沿 channel axis
# ---------------------------------------------------------------------------

def one_hot_encode_along_channel_axis(sequence: str) -> np.ndarray:
    """
    将 DNA 序列沿 channel axis (axis=0) 做 one-hot 编码

    Args:
        sequence: 只含 A/T/C/G/N 的字符串

    Returns:
        np.ndarray: (4, len(sequence)) 形状的 int8 矩阵
    """
    to_return = np.zeros((4, len(sequence)), dtype=np.int8)
    _seq_to_one_hot_fill(to_return, sequence, one_hot_axis=0)
    return to_return


def _seq_to_one_hot_fill(zero_array: np.ndarray, sequence: str, one_hot_axis: int):
    """填充 one-hot 矩阵 (直接修改传入数组)"""
    assert one_hot_axis == 0 or one_hot_axis == 1
    if one_hot_axis == 0:
        assert zero_array.shape[1] == len(sequence)
    elif one_hot_axis == 1:
        assert zero_array.shape[0] == len(sequence)

    for i, char in enumerate(sequence):
        if char == "A" or char == "a":
            char_idx = 0
        elif char == "C" or char == "c":
            char_idx = 1
        elif char == "G" or char == "g":
            char_idx = 2
        elif char == "T" or char == "t":
            char_idx = 3
        elif char == "N" or char == "n":
            continue  # 保持全零
        else:
            raise RuntimeError(f"Unsupported character: {char}")

        if one_hot_axis == 0:
            zero_array[char_idx, i] = 1
        elif one_hot_axis == 1:
            zero_array[i, char_idx] = 1


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class MaizeDataset(Dataset):
    """玉米基因表达量数据集"""

    def __init__(self, inputs: np.ndarray, labels: np.ndarray):
        """
        Args:
            inputs: (N, 4, 3000) one-hot 编码矩阵
            labels: (N,) log2(TPM+1) 值
        """
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, index: int):
        return self.inputs[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)


# ---------------------------------------------------------------------------
# 主加载函数
# ---------------------------------------------------------------------------

def load_data(excel_path: str, batch_size: int = 32):
    """
    加载并预处理数据集

    Args:
        excel_path: dataset.xlsx 路径
        batch_size: DataLoader batch size

    Returns:
        train_loader, valid_loader, test_loader,
        y_train, y_valid, y_test (原始 numpy 数组, 用于最后评估)
    """
    print(f"Loading data from: {excel_path}")
    df = pd.read_excel(excel_path)
    print(f"Genes number: {df.shape[0]}")
    print(f"Columns: {df.columns.tolist()}")

    # 按 dataset 列划分
    df_train = df[df["dataset"] == "train"]
    df_valid = df[df["dataset"] == "valid"]
    df_test = df[df["dataset"] == "test"]

    print(f"train: {len(df_train)}, valid: {len(df_valid)}, test: {len(df_test)}")

    # 标签: log2(TPM + 1)
    y_train = np.log2(df_train["TPM"].values + 1).astype(np.float32)
    y_valid = np.log2(df_valid["TPM"].values + 1).astype(np.float32)
    y_test = np.log2(df_test["TPM"].values + 1).astype(np.float32)

    # One-hot 编码 DNA 序列
    print("One-hot encoding DNA sequences...")
    train_data = np.array(
        [one_hot_encode_along_channel_axis(s.strip()) for s in df_train["sequence"].values]
    )
    valid_data = np.array(
        [one_hot_encode_along_channel_axis(s.strip()) for s in df_valid["sequence"].values]
    )
    test_data = np.array(
        [one_hot_encode_along_channel_axis(s.strip()) for s in df_test["sequence"].values]
    )
    print(f"Encoded shapes: train={train_data.shape}, valid={valid_data.shape}, test={test_data.shape}")

    # 构建 DataLoader
    train_dataset = MaizeDataset(train_data, y_train)
    valid_dataset = MaizeDataset(valid_data, y_valid)
    test_dataset = MaizeDataset(test_data, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, valid_loader, test_loader, y_train, y_valid, y_test


if __name__ == "__main__":
    import sys

    excel_path = sys.argv[1] if len(sys.argv) > 1 else "dataset.xlsx"
    train_loader, valid_loader, test_loader, yt, yv, yte = load_data(excel_path, batch_size=32)

    print(f"\nTPM stats (before log2 transform):")
    for name, y in [("train", yt), ("valid", yv), ("test", yte)]:
        print(f"  {name}: mean={2**y.mean()-1:.2f}, median={2**np.median(y)-1:.2f}")
