import torch
import numpy as np

from torch import nn
from source.utils.runtime import set_model_is_training


def sliding_window_inference(
    model: nn.Module,
    inputs: torch.Tensor,
    num_classes: int,
    crop_size: int,
    overlap: float = 0.5,
    masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Sliding window inference for multi-modal segmentation.
        - Support different masks for different modalities combination.

    <IMPORTANT>:
        - ALL outputs are assumed <WITHOUT> SoftMax or LogSoftMax.
        - Model should have "is_training" attribute to get single output
        - Model forward process take the input with shape (N, num_modals, crop_size, crop_size, crop_size)

    Args:
        model (nn.Module): Model to be used for inference.
        inputs (torch.Tensor): Input tensor with shape (N, num_modals, H, W, D).
        num_classes (int): Number of classes of output.
        crop_size (int): Size of the crop for sliding window inference.
        overlap (float): Overlap ratio of the crop for sliding window inference.
        masks (torch.Tensor[bool, ...]): Masks for different modalities combination.

    Returns:
        torch.Tensor: Output tensor with shape (N, H, W, D) containing predicted class indices.
    """
    model.eval()
    set_model_is_training(model, False)

    device = inputs.device
    batch_size, _, h, w, d = inputs.size()

    mask_tensor = torch.tensor([True, True, True, True], dtype=torch.bool) if masks is None else masks
    if mask_tensor.dim() == 1:
        mask_tensor = mask_tensor.unsqueeze(0).repeat(batch_size, 1)
    masks = mask_tensor.to(device)

    stride = int(crop_size * (1 - overlap))

    h_cnt = int(np.ceil((h - crop_size) / stride)) + 1
    h_idx_list = [h_idx * stride for h_idx in range(h_cnt)]
    if h_idx_list[-1] + crop_size > h:
        h_idx_list[-1] = h - crop_size

    w_cnt = int(np.ceil((w - crop_size) / stride)) + 1
    w_idx_list = [w_idx * stride for w_idx in range(w_cnt)]
    if w_idx_list[-1] + crop_size > w:
        w_idx_list[-1] = w - crop_size

    d_cnt = int(np.ceil((d - crop_size) / stride)) + 1
    d_idx_list = [d_idx * stride for d_idx in range(d_cnt)]
    if d_idx_list[-1] + crop_size > d:
        d_idx_list[-1] = d - crop_size

    one_tensor = torch.ones(
        1, 1, crop_size, crop_size, crop_size, device=device, dtype=inputs.dtype
    )
    weight = torch.zeros(
        1, 1, h, w, d, device=device, dtype=inputs.dtype
    )

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            for d_idx in d_idx_list:
                weight[
                    :, :, h_idx:h_idx+crop_size, w_idx:w_idx+crop_size, d_idx:d_idx+crop_size
                ] += one_tensor

    weight = weight.repeat(batch_size, num_classes, 1, 1, 1)

    pred = torch.zeros(batch_size, num_classes, h, w, d, device=device, dtype=inputs.dtype)
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                for d_idx in d_idx_list:
                    x_input = inputs[
                        :, :, h_idx:h_idx+crop_size, w_idx:w_idx+crop_size, d_idx:d_idx+crop_size
                    ]
                    pred_part = model(x_input, masks)
                    pred_part = softmax(pred_part)
                    pred[
                        :, :, h_idx:h_idx+crop_size, w_idx:w_idx+crop_size, d_idx:d_idx+crop_size
                    ] += pred_part

    pred = pred / weight
    pred = torch.argmax(pred, dim=1)

    return pred
