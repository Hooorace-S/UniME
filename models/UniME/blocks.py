import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    Basic Blocks with <Conv> style for modern architecture.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_affine: bool = False,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect', bias=False
            ),
            nn.InstanceNorm3d(num_features=out_channels, affine=norm_affine),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor with shape (B, C, D, H, W)
        """
        return self.block(x)


class ECABlock(nn.Module):
    """
    Constructs a ECA module.
    """
    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ECA module.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor with shape (B, C, D, H, W)
        """
        # Global average pooling on spatial dims -> (B, C, D, 1, 1)
        y = self.avg_pool(x)

        # Reshape for Conv1d: (B, C, D, 1, 1) -> (B, D, C)
        y = y.squeeze(-1).squeeze(-1).transpose(1, 2)  # [B, D, C]

        # Conv1d expects (B, 1, C) or (B, D, C)
        y = self.conv(y).transpose(1, 2)  # [B, D, C] -> [B, C, D]

        # Sigmoid activation
        y = self.sigmoid(y).unsqueeze(-1).unsqueeze(-1)  # [B, C, D, 1, 1]

        # Reweight
        return x * (2.0 * y).expand_as(x)


class FusionBlock(nn.Module):
    """
    Modal Fusion Block for multi-modal fusion.
    """
    def __init__(self, in_channels: int, num_modals: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels=in_channels * num_modals, out_channels=in_channels),
            ConvBlock(in_channels=in_channels, out_channels=in_channels),
            ECABlock(kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape (B, K*C, D, H, W)

        Returns:
            torch.Tensor: Output tensor with shape (B, C, D, H, W)
        """
        return self.block(x)


class FusionBlockVariants(nn.Module):
    """
    Modal Fusion Block for multi-modal fusion but with different number of channels.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels),
            ConvBlock(in_channels=out_channels, out_channels=out_channels),
            ECABlock(kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor with shape (B, C, D, H, W)
        """
        return self.block(x)
