import torch
from torch import nn


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss with regularization of the volume-wise mean of the learnable prior.

    <NOTE>:
        - Loss = MSE(recon, target) + regulization_rate * ||volume-wise mean (learnable_prior)||_2
    """
    def __init__(self, regulization_rate: float = 0.005, eps: float = 1e-6):
        """
        Args:
            regulization_rate (float, optional): Regularization rate for the volume-wise mean.
            eps (float, optional): Epsilon for the volume-wise mean.
        """
        super().__init__()
        self.regularization_rate = regulization_rate
        self.mse = nn.MSELoss(reduction='mean')
        self.eps = eps

    def forward(
        self, recon: torch.Tensor, learnable_prior: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            recon (torch.Tensor): Reconstruction output with shape (B, num_modalities, D, H, W)
            learnable_prior (torch.Tensor): Learnable prior output with shape (B, num_modalities, D, H, W)
            target (torch.Tensor): Target output with shape (B, num_modalities, D, H, W)

        <NOTE>:
            - learnable_prior is the learnable prior output of the model.
            - recon is assumed <WITHOUT> SoftMax or LogSoftMax.

        Returns:
            torch.Tensor: scalar loss
        """
        match self.regularization_rate:
            case 0.0:
                return self.mse(recon, target)
            case _:
                mean = learnable_prior.mean(dim=(2, 3, 4), keepdim=True)
                reg = torch.norm(learnable_prior - mean, p=2, dim=(2, 3, 4))
                reg = (reg + self.eps).mean()
                return self.mse(recon, target) + self.regularization_rate * reg
