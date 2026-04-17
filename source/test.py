import csv
import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from source.config import TrainingConfig
from source.logger import Logger
from source.config import MASKS, MASK_NAMES, MASK_ARRAY
from source.core import (
    sliding_window_inference,
    evaluate_single_sample
)
from source.utils import wandb_utils
from source.utils.runtime import load_model_state_dict_compat, require_cuda, set_model_is_training

TEST_MASKS_TENSOR = torch.tensor(MASKS, dtype=torch.bool)
CheckpointRecord = tuple[int, str, float]


def test_model(
    model: nn.Module,
    args: TrainingConfig,
    test_loader: DataLoader,
    epoch: int,
    logger: Logger,
    test_results_csv: str,
    test_masks: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Test the model with the test data loader across <ALL> modalities combination (15 cases, see MASK_NAMES).

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
        <6> For each modalities combination (for example, T1), we shall compute:
            - T1-WT, T1-TC, T1-ET, T1-ET_postpro
            and we finally get 15 cases, 4 evaluation areas. We shall write the results to the test_results_csv file in
            the format:
                - Epoch (1),  T1, T2, T1ce, FLAIR, T1ce-T2, T1ce-T1, ..., T1ce-T1-T2, ALL, Average
                - WT
                - TC
                - ET
                - ET_postpro
                - Epoch (2), ...
                - ...
                - Epoch (k), ...
                - ...
            where <Average> is the average of the 15 cases for each evaluation area and k is the number of checkpoints
            evaluated in the current testing phase.
        <7> The shape of return tensor of test data loader is:
                - inputs: (N, num_modals, crop_size, crop_size, crop_size)
                - targets: (N, num_modals, crop_size, crop_size, crop_size)
                - masks: (N, num_modals)
                - names: (N, )
            where N is the batch size. So `targets = torch.argmax(targets, dim=1)` is needed.

    Args:
        model (nn.Module): Model to test
        args (TrainingConfig): Training configuration
        test_loader (DataLoader): Test data loader
        logger (Logger): Logger instance
        test_results_csv (str): Path to the test results csv file
        test_masks (torch.Tensor, optional): Test masks for different modalities combinations.

    Returns:
        tuple[np.ndarray, np.ndarray, list[str]]:
            - average_results (np.ndarray): Average dice scores across all modalities, shape (4,)
            - all_modality_results (np.ndarray): Dice scores per modality, shape (num_masks, 4)
            - tested_names (list[str]): Modality combination names aligned with rows in all_modality_results
    """
    device = require_cuda("test-time evaluation")

    # Set model to evaluation mode
    model.eval()
    set_model_is_training(model, False)

    # Define evaluation area names
    evaluation_names = ['WT', 'TC', 'ET', 'ET_postpro']

    # Use MASK_ARRAY for consistency and build name mapping
    # This ensures we always use the canonical masks and names
    resolved_test_masks = TEST_MASKS_TENSOR if test_masks is None else test_masks
    if isinstance(resolved_test_masks, torch.Tensor):
        test_masks_array = (
            resolved_test_masks.numpy()
            if resolved_test_masks.is_cuda is False
            else resolved_test_masks.cpu().numpy()
        )
    else:
        test_masks_array = np.array(resolved_test_masks)

    # Build mask to name mapping for robustness
    mask_to_name = {}
    for idx, (mask, name) in enumerate(zip(MASK_ARRAY, MASK_NAMES)):
        mask_to_name[tuple(mask.tolist())] = (idx, name)

    # Validate that all test masks are recognized
    num_test_masks = len(test_masks_array)
    valid_mask_indices = []
    for test_mask in test_masks_array:
        mask_tuple = tuple(test_mask.tolist())
        if mask_tuple in mask_to_name:
            valid_mask_indices.append(mask_to_name[mask_tuple][0])
        else:
            raise ValueError(f"Unrecognized mask pattern: {test_mask}")

    # Initialize storage for all modality combinations results
    # Shape: (num_test_masks, 4 evaluation areas)
    all_modality_results = np.zeros((num_test_masks, len(evaluation_names)))

    # Log test start
    logger.write("=" * 80 + "\n")
    logger.write(f"Testing at epoch {epoch} across {num_test_masks} modality combinations...\n")
    logger.write("=" * 80 + "\n")

    # Process each modality combination
    for test_idx, (mask_idx, test_mask) in enumerate(zip(valid_mask_indices, test_masks_array)):
        mask_name = MASK_NAMES[mask_idx]
        current_mask = torch.tensor(test_mask, dtype=torch.bool)

        # Initialize accumulators for this modality combination
        dice_evaluation_list = []

        # Log current modality being tested
        logger.write(f"\nTesting modality combination: {mask_name}\n")
        logger.write("-" * 40 + "\n")

        # Process all samples with this modality combination
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Testing {mask_name}", leave=False)

            for _, (inputs, targets, _, _) in enumerate(pbar):
                # Move data to GPU
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                # Use current modality mask for all samples in batch
                batch_size = inputs.shape[0]
                masks = current_mask.unsqueeze(0).repeat(batch_size, 1).to(device, non_blocking=True)

                # Perform sliding window inference
                predictions = sliding_window_inference(
                    model=model,
                    inputs=inputs,
                    num_classes=args.num_classes,
                    crop_size=args.crop_size,
                    overlap=0.5,
                    masks=masks
                )

                # Convert targets to class indices
                targets = torch.argmax(targets, dim=1)

                # Evaluate each sample in the batch
                _, dice_evaluation_batch = evaluate_single_sample(
                    output_classes=predictions,
                    target_classes=targets
                )

                # Accumulate results for this modality
                dice_evaluation_list.append(dice_evaluation_batch)

                # Update progress bar with current batch metrics
                if len(dice_evaluation_list) > 0:
                    current_avg = np.concatenate(dice_evaluation_list, axis=0).mean(axis=0)
                    pbar.set_postfix({
                        'WT': f'{current_avg[0]:.4f}',
                        'TC': f'{current_avg[1]:.4f}',
                        'ET': f'{current_avg[2]:.4f}'
                    })

        # Calculate average scores for this modality combination
        dice_evaluation_all = np.concatenate(dice_evaluation_list, axis=0)
        avg_dice_evaluation = np.mean(dice_evaluation_all, axis=0)

        # Store results for this modality
        all_modality_results[test_idx] = avg_dice_evaluation

        # Log results for this modality combination
        logger.write(f"Results for {mask_name}:\n")
        for eval_name, score in zip(evaluation_names, avg_dice_evaluation):
            logger.write(f"  {eval_name}: {score:.6f}\n")
        logger.flush()

    # Calculate average across all modality combinations
    average_results = np.mean(all_modality_results, axis=0)

    # Log overall results
    logger.write("\n" + "=" * 80 + "\n")
    logger.write(f"Test Results Summary - Epoch {epoch}:\n")
    logger.write("=" * 80 + "\n")

    # Log individual modality results
    logger.write("\nPer-Modality Results:\n")
    logger.write("-" * 80 + "\n")
    for test_idx, mask_idx in enumerate(valid_mask_indices):
        mask_name = MASK_NAMES[mask_idx]
        logger.write(f"{mask_name:15s}: ")
        scores_str = " | ".join([f"{eval_name}: {all_modality_results[test_idx, i]:.4f}"
                                for i, eval_name in enumerate(evaluation_names)])
        logger.write(scores_str + "\n")

    # Log average results
    logger.write("\n" + "-" * 80 + "\n")
    logger.write("Average across all modalities:\n")
    for eval_name, score in zip(evaluation_names, average_results):
        logger.write(f"  {eval_name}: {score:.6f}\n")

    logger.write("=" * 80 + "\n")
    logger.flush()

    # Write results to CSV file
    if test_results_csv:
        # Check if CSV exists or needs to be created
        write_header = not os.path.exists(test_results_csv)

        with open(test_results_csv, 'a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write header if file is new
            if write_header:
                # Create header row with tested modality names
                tested_names = [MASK_NAMES[idx] for idx in valid_mask_indices]
                header = ['Metric'] + tested_names + ['Average']
                csv_writer.writerow(header)

            # Write epoch separator
            csv_writer.writerow([f'Epoch {epoch}'] + [''] * num_test_masks)

            # Write results for each evaluation area
            for eval_idx, eval_name in enumerate(evaluation_names):
                row_data = [eval_name]
                # Add scores for each tested modality
                for test_idx in range(num_test_masks):
                    row_data.append(f"{all_modality_results[test_idx, eval_idx]:.6f}")
                # Add average
                row_data.append(f"{average_results[eval_idx]:.6f}")
                csv_writer.writerow(row_data)

            # Add empty row for separation
            csv_writer.writerow([])

    # Set model back to training mode
    model.train()
    set_model_is_training(model, True)
    tested_names = [MASK_NAMES[idx] for idx in valid_mask_indices]
    return average_results, all_modality_results, tested_names


def test_top_k_models(
    model: nn.Module,
    args: TrainingConfig,
    test_loader: DataLoader,
    topk_checkpoints: list[CheckpointRecord],
    test_results_csv: str,
    logger: Logger
) -> None:
    """
    Test selected checkpoints saved during training.

    Args:
        model (nn.Module): Model instance (will be loaded with checkpoint weights)
        args (TrainingConfig): Training configuration
        test_loader (DataLoader): Test data loader
        topk_checkpoints (list): List of tuples (epoch, checkpoint_path, composite_score)
        test_results_csv (str): Path to save test results
        logger (Logger): Logger instance
    """
    _ = require_cuda("checkpoint testing")

    if not topk_checkpoints:
        logger.write("No checkpoints to test.\n")
        return

    logger.write("\n" + "=" * 80 + "\n")
    logger.write(f"Testing {len(topk_checkpoints)} checkpoints on test dataset...\n")
    logger.write("=" * 80 + "\n")

    # Sort checkpoints by composite score (best first)
    topk_checkpoints_sorted = sorted(topk_checkpoints, key=lambda x: x[2], reverse=True)

    for idx, (epoch, checkpoint_path, composite_score) in enumerate(topk_checkpoints_sorted):
        logger.write(
            f"\n[{idx + 1}/{len(topk_checkpoints)}] Testing model from epoch {epoch} "
            f"(validation composite score: {composite_score:.6f})\n"
        )

        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            logger.write(f"WARNING: Checkpoint not found: {checkpoint_path}\n")
            continue

        try:
            # Load checkpoint
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False
            )
            load_model_state_dict_compat(model, checkpoint['model_state_dict'])
            logger.write(f"Loaded checkpoint: {os.path.basename(checkpoint_path)}\n")

            # Test the model using test_model function
            average_results, all_modality_results, tested_names = test_model(
                model=model,
                args=args,
                test_loader=test_loader,
                epoch=epoch,
                logger=logger,
                test_results_csv=test_results_csv
            )
            wandb_utils.log(
                {
                    "epoch": int(epoch),
                    "test/checkpoint_epoch": int(epoch),
                    "test/avg_WT": float(average_results[0]),
                    "test/avg_TC": float(average_results[1]),
                    "test/avg_ET": float(average_results[2]),
                    "test/avg_ET_postpro": float(average_results[3]),
                },
            )
            if tested_names:
                rows = []
                for name, scores in zip(tested_names, all_modality_results):
                    rows.append([int(epoch), name, float(scores[0]), float(scores[1]), float(scores[2]), float(scores[3])])
                wandb_utils.log_table(
                    "test/per_modality",
                    columns=["epoch", "mask", "WT", "TC", "ET", "ET_postpro"],
                    data=rows,
                )
            if idx == 0:
                run = wandb_utils.run()
                if run is not None:
                    try:
                        run.summary["test_top1_epoch"] = int(epoch)
                        run.summary["test_top1_avg_WT"] = float(average_results[0])
                        run.summary["test_top1_avg_TC"] = float(average_results[1])
                        run.summary["test_top1_avg_ET"] = float(average_results[2])
                        run.summary["test_top1_avg_ET_postpro"] = float(average_results[3])
                    except Exception:
                        pass

            logger.write(f"Completed testing for epoch {epoch}\n")

        except Exception as exc:  # noqa: BLE001
            logger.write(f"ERROR loading/testing checkpoint {checkpoint_path}: {exc}\n")
            continue

    logger.write("\n" + "=" * 80 + "\n")
    logger.write("All checkpoints tested successfully!\n")
    logger.write(f"Test results saved to: {test_results_csv}\n")
    logger.write("=" * 80 + "\n")
    logger.flush()
