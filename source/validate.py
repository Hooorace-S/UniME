from typing import Tuple
import csv
import os

import torch
import numpy as np
from tqdm import tqdm

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from source.core import evaluate_single_sample, sliding_window_inference
from source.config import TrainingConfig, VALID_MASKS
from source.logger import Logger
from source.utils.runtime import require_cuda, set_model_is_training


def validate_model(
    model: nn.Module,
    args: TrainingConfig,
    val_loader: DataLoader,
    epoch: int,
    writer: SummaryWriter,
    logger: Logger,
    validation_results_csv: str,
    valid_masks: torch.Tensor | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Validate the model with the validation data loader and return the average dice score

    <IMPORTANT>:
        <1> ALL outputs are assumed <WITHOUT> SoftMax or LogSoftMax.
        <2> Model should have "is_training" attribute to get single output
        <3> Model forward process take the input with shape (N, num_modals, crop_size, crop_size, crop_size)
            with num_modals = args.num_modals and crop_size = args.crop_size
        <4> In original BraTS dataset, the classes are:
                - 0: background - 1: NCR/NET - 2: Edema - 4: Enhancing tumor
            But we have converted the classes in data processing stage to:
                - 0: background - 1: NCR/NET - 2: Edema - 3: Enhancing tumor
        <5> The standard evaluation areas in BraTS are:
                - WT: (label == 1) | (label == 2) | (label == 3)
                - TC: (label == 1) | (label == 3)
                - ET: (label == 3)
            Therefore, in this function, we return both the average dice score for each class and each evaluation area
        <6> Since the validation is used to get the final model, we only validate it with the "ALL" modalities combination
            and the valid masks are set to [True, True, True, True].
        <7> The shape of return tensor of valid data loader is:
                - inputs: (N, num_modals, crop_size, crop_size, crop_size)
                - targets: (N, num_modals, crop_size, crop_size, crop_size)
                - masks: (N, num_modals)
                - names: (N, )
            where N is the batch size. So `targets = torch.argmax(targets, dim=1)` is needed.

    Args:
        model (nn.Module): Model to validate
        val_loader (DataLoader): Validation data loader
        epoch (int): Current epoch to validate
        writer (SummaryWriter): TensorBoard writer
        logger (Logger): Logger instance
        validation_results_csv (str): Path to the validation results csv file
        valid_masks (torch.Tensor): Valid masks for different modalities combination.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - dice_separate (np.ndarray): Average (across all samples) dice score for each class, with shape (3, )
            - dice_evaluation (np.ndarray): Average (across all samples) dice score for each evaluation area, with shape (4, )
            - composite_score (float): Composite score calculated as mean of all evaluation dice scores
    """
    # Set validation masks to all True for complete modality validation
    device = require_cuda("validation")

    valid_mask_tensor = (
        torch.tensor(VALID_MASKS, dtype=torch.bool)
        if valid_masks is None else
        valid_masks
    )

    # Initialize accumulators for dice scores
    all_dice_separate = []
    all_dice_evaluation = []

    # Class names for logging
    class_names = ['NCR/NET', 'Edema', 'ET']
    evaluation_names = ['WT', 'TC', 'ET', 'ET_postpro']

    # Log validation start
    logger.write("=" * 80 + "\n")
    logger.write(f"Validation at epoch {epoch}...\n")
    logger.write("=" * 80 + "\n")

    # Set model to evaluation mode
    model.eval()
    set_model_is_training(model, False)

    # Iterate through validation data with progress bar
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}", leave=False)

        for batch_idx, (inputs, targets, masks, names) in enumerate(pbar):
            # Move data to GPU
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Use complete modality masks for validation
            batch_size = inputs.shape[0]
            masks = valid_mask_tensor.unsqueeze(0).repeat(batch_size, 1).to(device, non_blocking=True)

            # Perform sliding window inference
            predictions = sliding_window_inference(
                model=model,
                inputs=inputs,
                num_classes=args.num_classes,
                crop_size=args.crop_size,
                overlap=0.5,
                masks=masks
            )

            targets = torch.argmax(targets, dim=1)

            # Evaluate each sample in the batch
            dice_separate_batch, dice_evaluation_batch = evaluate_single_sample(
                output_classes=predictions,
                target_classes=targets
            )

            # Accumulate results
            all_dice_separate.append(dice_separate_batch)
            all_dice_evaluation.append(dice_evaluation_batch)

            # Update progress bar with current batch metrics
            mean_dice = dice_evaluation_batch[:, :3].mean()
            pbar.set_postfix({'Dice': f'{mean_dice:.4f}'})

            # Log per-batch results (less verbose)
            for k, name in enumerate(names):
                # Add evaluation area scores
                eval_scores = []
                for j, eval_name in enumerate(evaluation_names[:3]):
                    score = dice_evaluation_batch[k, j]
                    eval_scores.append(f"{eval_name}: {float(score):.4f}")

                # Only log to file, not to terminal (progress bar is enough)
                msg = f"Sample [{batch_idx * val_loader.batch_size + k + 1}/{len(val_loader.dataset)}] {name}: "
                msg += ", ".join(eval_scores)
                logger.write(msg + "\n")

            # Flush logger periodically
            if (batch_idx + 1) % 10 == 0:
                logger.flush()

    # Calculate average scores across all samples
    all_dice_separate = np.concatenate(all_dice_separate, axis=0)
    all_dice_evaluation = np.concatenate(all_dice_evaluation, axis=0)

    avg_dice_separate = np.mean(all_dice_separate, axis=0)
    avg_dice_evaluation = np.mean(all_dice_evaluation, axis=0)

    # Log average results
    logger.write("-" * 80 + "\n")
    logger.write(f"Validation Results - Epoch {epoch}:\n")
    logger.write("-" * 80 + "\n")

    # Log class-wise dice scores
    logger.write("Class-wise Dice Scores:\n")
    for class_name, score in zip(class_names, avg_dice_separate):
        logger.write(f"  {class_name}: {score:.4f}\n")

    # Log evaluation area dice scores
    logger.write("\nEvaluation Area Dice Scores:\n")
    for eval_name, score in zip(evaluation_names, avg_dice_evaluation):
        logger.write(f"  {eval_name}: {score:.4f}\n")

    logger.write("=" * 80 + "\n")
    logger.flush()

    # Calculate composite score
    composite_score = avg_dice_evaluation.mean()

    # Write to TensorBoard
    writer.add_scalar('Validation/Dice_WT', avg_dice_evaluation[0], epoch)
    writer.add_scalar('Validation/Dice_TC', avg_dice_evaluation[1], epoch)
    writer.add_scalar('Validation/Dice_ET', avg_dice_evaluation[2], epoch)
    writer.add_scalar('Validation/Dice_ET_postpro', avg_dice_evaluation[3], epoch)
    writer.add_scalar('Validation/Composite_Score', composite_score, epoch)

    # Write to CSV file
    if validation_results_csv:
        # Check if CSV file exists, if not create with header
        if not os.path.exists(validation_results_csv):
            with open(validation_results_csv, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                headers = ['Epoch', 'Dice_WT', 'Dice_TC', 'Dice_ET', 'Dice_ET_postpro', 'Composite_Score']
                csv_writer.writerow(headers)

        # Append validation results
        with open(validation_results_csv, 'a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            row_data = [
                epoch,
                f"{avg_dice_evaluation[0]:.6f}",
                f"{avg_dice_evaluation[1]:.6f}",
                f"{avg_dice_evaluation[2]:.6f}",
                f"{avg_dice_evaluation[3]:.6f}",
                f"{composite_score:.6f}"
            ]
            csv_writer.writerow(row_data)

    # Set model back to training mode
    model.train()
    set_model_is_training(model, True)

    return avg_dice_separate, avg_dice_evaluation, composite_score
