"""
Basenji 模型实现 — 用于玉米基因表达量预测

模型结构来源于文献:
Kelley et al. "Sequential regulatory activity prediction across
chromosomes with convolutional neural networks."
Genome Research, 2018. https://genome.cshlp.org/content/28/5/739

网络由以下模块组成:
  ConvBlock → ConvTower → DilatedResidual → ConvBlock(1×1) → ConvBlock(1×1) → BasenjiFinal
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops.layers.torch import Rearrange


# ---------------------------------------------------------------------------
# 激活函数: GELU (Gaussian Error Linear Unit)
# ---------------------------------------------------------------------------

class GELU(nn.Module):
    """自定义 GELU: x * sigmoid(1.702 * x)"""

    def __init__(self):
        super().__init__()
        self.constant_param = nn.Parameter(torch.Tensor([1.702]))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(self.constant_param * x)


# ---------------------------------------------------------------------------
# MaxPool1d 包装: 支持 padding='same'
# ---------------------------------------------------------------------------

class KerasMaxPool1d(nn.Module):
    """torch MaxPool1d 没有 padding='same', 因此重新定义, kernel_size 固定为 2"""

    def __init__(self, pool_size=2, padding="valid", dilation=1,
                 return_indices=False, ceil_mode=False):
        super().__init__()
        self.padding = padding
        if pool_size != 2:
            raise NotImplementedError(
                "MaxPool1D with kernel size other than 2."
            )
        self.pool = nn.MaxPool1d(
            kernel_size=pool_size,
            padding=0,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.padding == "same" and x.shape[-1] % 2 == 1:
            x = F.pad(x, (0, 1), value=-float("inf"))
        return self.pool(x)


# ---------------------------------------------------------------------------
# 残差连接
# ---------------------------------------------------------------------------

class Residual(nn.Module):
    """残差包装: output = module(x) + x"""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x) + x


# ---------------------------------------------------------------------------
# ConvBlock: GELU → Conv1d → BatchNorm → (Dropout) → (MaxPool)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int,
        kernel_size=1,
        padding="same",
        stride=1,
        dilation_rate=1,
        pool_size=1,
        dropout=0.0,
        bn_momentum=0.1,
    ):
        super().__init__()
        block = nn.ModuleList()
        block.append(GELU())
        block.append(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=int(round(dilation_rate)),
                bias=False,
            )
        )
        block.append(nn.BatchNorm1d(filters, momentum=bn_momentum, affine=True))
        if dropout > 0:
            block.append(nn.Dropout(p=dropout))
        if pool_size > 1:
            block.append(KerasMaxPool1d(pool_size=pool_size, padding=padding))
        self.block = nn.Sequential(*block)
        self.out_channels = filters

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# ConvTower: 多个 ConvBlock 串联, filters 逐步倍增
# ---------------------------------------------------------------------------

class ConvTower(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters_init: int,
        filters_end: int = None,
        filters_mult: float = None,
        divisible_by: int = 1,
        repeat: int = 2,
        **kwargs,
    ):
        super().__init__()

        def _round(x: float) -> int:
            return int(np.round(x / divisible_by) * divisible_by)

        # 确定乘数
        if filters_mult is None:
            assert filters_end is not None
            filters_mult = np.exp(
                np.log(filters_end / filters_init) / (repeat - 1)
            )

        rep_filters = filters_init
        in_ch = in_channels
        tower = nn.ModuleList()
        for _ in range(repeat):
            tower.append(
                ConvBlock(
                    in_channels=in_ch,
                    filters=_round(rep_filters),
                    **kwargs,
                )
            )
            in_ch = _round(rep_filters)
            rep_filters *= filters_mult

        self.tower = nn.Sequential(*tower)
        self.out_channels = in_ch

    def forward(self, x: Tensor) -> Tensor:
        return self.tower(x)


# ---------------------------------------------------------------------------
# DilatedResidual: 空洞残差块, dilation rate 按 rate_mult 递增
# ---------------------------------------------------------------------------

class DilatedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int,
        kernel_size=3,
        rate_mult=2,
        dropout=0.0,
        repeat=1,
        **kwargs,
    ):
        super().__init__()
        dilation_rate = 1.0
        in_ch = in_channels
        blocks = nn.ModuleList()

        for _ in range(repeat):
            inner = nn.ModuleList()
            inner.append(
                ConvBlock(
                    in_channels=in_ch,
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation_rate=int(np.round(dilation_rate)),
                    **kwargs,
                )
            )
            inner.append(
                ConvBlock(
                    in_channels=filters,
                    filters=in_channels,
                    dropout=dropout,
                    **kwargs,
                )
            )
            blocks.append(Residual(nn.Sequential(*inner)))
            dilation_rate *= rate_mult
            dilation_rate = np.round(dilation_rate)

        self.block = nn.Sequential(*blocks)
        self.out_channels = in_channels

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# BasenjiFinal: Flatten → Linear
# ---------------------------------------------------------------------------

class BasenjiFinal(nn.Module):
    def __init__(self, in_features: int, units=1, activation="linear", **kwargs):
        super().__init__()
        block = nn.ModuleList()
        # Rearrange('b ... -> b (...)') 等价于 flatten
        block.append(Rearrange("b ... -> b (...)"))
        block.append(nn.Linear(in_features=in_features, out_features=units))
        self.block = nn.Sequential(*block)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# BasenjiModel: 完整模型
# ---------------------------------------------------------------------------

class BasenjiModel(nn.Module):
    """
    输入: (batch, 4, 3000) — one-hot 编码的 DNA 序列
    输出: (batch, 1) — 预测的 log2(TPM + 1)

    结构:
      input(4, 3000)
        → ConvBlock (filters=8, ks=15, pool=2)          → (8, 1500)
        → ConvTower (filters: 16→32, repeat=2, pool=2)   → (32, 750)
        → DilatedResidual (filters=16, repeat=2)          → (32, 750)
        → ConvBlock (1×1 conv, filters=32)                → (32, 750)
        → ConvBlock (1×1 conv, filters=1)                 → (1, 750)
        → BasenjiFinal (Linear)                           → (1)
    """

    def __init__(
        self,
        # 第一个 conv block 参数
        conv1_filters=8,
        conv1_ks=15,
        conv1_pad=7,
        conv1_pool=2,
        conv1_pdrop=0.4,
        conv1_bn_momentum=0.1,
        # conv tower 参数
        convt_filters_init=16,
        filters_end=32,
        convt_repeat=2,
        convt_ks=5,
        convt_pool=2,
        # dilated residual block 参数
        dil_in_channels=32,
        dil_filters=16,
        dil_ks=3,
        rate_mult=2,
        dil_pdrop=0.3,
        dil_repeat=2,
        # 第二个和第三个 conv block 参数
        conv2_in_channels=32,
        conv2_filters=32,
        conv3_in_channels=32,
        conv3_filters=1,
        # final block 参数 (3000 / 2^3 = 375)
        final_in_features=375,
    ):
        super().__init__()

        block = nn.ModuleList()

        # 1. 第一个 ConvBlock: (4, 3000) → (8, 1500)
        block.append(
            ConvBlock(
                in_channels=4,
                filters=conv1_filters,
                kernel_size=conv1_ks,
                padding=conv1_pad,
                pool_size=conv1_pool,
                dropout=conv1_pdrop,
                bn_momentum=conv1_bn_momentum,
            )
        )

        # 2. ConvTower: (8, 1500) → (32, 750)
        block.append(
            ConvTower(
                in_channels=conv1_filters,
                filters_init=convt_filters_init,
                filters_end=filters_end,
                repeat=convt_repeat,
                kernel_size=convt_ks,
                pool_size=convt_pool,
            )
        )

        # 3. DilatedResidual: (32, 750) → (32, 750)
        block.append(
            DilatedResidual(
                in_channels=dil_in_channels,
                filters=dil_filters,
                kernel_size=dil_ks,
                rate_mult=rate_mult,
                dropout=dil_pdrop,
                repeat=dil_repeat,
            )
        )

        # 4. 第二个 ConvBlock (1×1 conv): (32, 750) → (32, 750)
        block.append(
            ConvBlock(
                in_channels=conv2_in_channels,
                filters=conv2_filters,
                kernel_size=1,
            )
        )

        # 5. 第三个 ConvBlock (1×1 conv): (32, 750) → (1, 750)
        block.append(
            ConvBlock(
                in_channels=conv3_in_channels,
                filters=conv3_filters,
                kernel_size=1,
            )
        )

        # 6. BasenjiFinal: flatten → Linear: (375) → (1)
        block.append(BasenjiFinal(in_features=final_in_features))

        self.block = nn.Sequential(*block)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, 4, 3000) one-hot encoded DNA sequence
        Returns:
            (batch, 1) predicted log2(TPM + 1)
        """
        return self.block(x)


# ---------------------------------------------------------------------------
# 快速测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torchsummary import summary

    BATCH_SIZE = 32
    model = BasenjiModel()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 使用 torchsummary 显示每层形状
    try:
        summary(model, input_size=(4, 3000), batch_size=BATCH_SIZE, device="cpu")
    except ImportError:
        print("torchsummary not installed, skipping layer summary.")

    # 前向传播测试
    x = torch.randn(BATCH_SIZE, 4, 3000)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
