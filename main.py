import os
from typing import Sized, cast

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed

from models import get_model
from models.UniME.networks import resolve_uni_encoder_pretrained_path
from models.UniME.wrapper import resolve_checkpoint_path
from source.config import get_args
from source.config.setup_seed import setup_seed
from source.dataset import setup_dataloader
from source.logger import Logger, setup_logger
from source.train import train_brats
from source.test import test_top_k_models
from source.utils import wandb_utils
from source.utils.runtime import is_accelerate_launch, require_cuda


def _apply_unime_model_overrides(
    args,
    model_kwargs: dict[str, int | float | str],
) -> tuple[dict[str, int | float | str], str | None]:
    is_unime_model = args.model_name.lower().startswith("unime")
    uni_encoder_name = args.uni_encoder_name
    uni_encoder_checkpoint = args.uni_encoder_checkpoint

    if not is_unime_model:
        if uni_encoder_name or uni_encoder_checkpoint:
            raise ValueError(
                "--uni_encoder_name and --uni_encoder_checkpoint are only supported for UniME models."
            )
        return model_kwargs, None

    selected_source = None
    if uni_encoder_name:
        model_kwargs["encoder_name"] = uni_encoder_name

    if uni_encoder_checkpoint:
        resolved_checkpoint, _exists = resolve_checkpoint_path(uni_encoder_checkpoint)
        model_kwargs["pretrained_path"] = resolved_checkpoint
        selected_source = f"Uni-Encoder checkpoint: {resolved_checkpoint}"
        return model_kwargs, selected_source

    if uni_encoder_name:
        default_layout_path = resolve_uni_encoder_pretrained_path(encoder_name=uni_encoder_name)
        resolved_checkpoint, _exists = resolve_checkpoint_path(default_layout_path)
        selected_source = f"Uni-Encoder name: {uni_encoder_name} -> {resolved_checkpoint}"

    return model_kwargs, selected_source


def main():
    args = get_args()
    assert args.log_root is not None

    # Set GPU IDs
    if args.gpu_ids and not is_accelerate_launch():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    require_cuda("the finetune entrypoint")
    mixed_precision = "no"
    if args.amp:
        mixed_precision = "bf16" if args.bfloat16 else "fp16"
    accelerator = Accelerator(mixed_precision=mixed_precision)

    # Setup random seed for reproducibility
    setup_seed(args.seed)
    accelerate_set_seed(args.seed, device_specific=True)

    # Setup logger with CSV paths
    log_setup = setup_logger(
        log_root=args.log_root,
        dataset_name=args.dataset_name,
        seed=args.seed,
        model_name=args.model_name,
        split_type=args.split_type,
        enabled=accelerator.is_main_process
    )
    validation_results_csv = cast(str, log_setup["validation_results_csv"])
    test_results_csv = cast(str, log_setup["test_results_csv"])
    logger = cast(Logger, log_setup["logger"])

    # Log configuration as structured data
    config_dict = args.to_dict()
    config_dict['type'] = 'config'
    config_dict['save_path'] = os.path.join(
        args.log_root, f"{args.dataset_name}-{args.seed}-{args.split_type}", args.model_name
    )
    logger.log_config(config_dict)

    # Also write to text log for visibility
    logger.write(str(args) + "\n")

    log_dir = config_dict["save_path"]
    tags: list[str] = []
    if getattr(args, "amp", False):
        tags.append("amp")
    if getattr(args, "bfloat16", False):
        tags.append("bf16")
    if getattr(args, "compile", False):
        tags.append("compile")
    if getattr(args, "layer_decay", None) is not None:
        tags.append("lldr")
    if getattr(args, "train_ratio", None) is not None:
        tags.append(f"ratio{args.train_ratio}")
    if accelerator.is_main_process:
        wandb_utils.init_run(args, log_dir=log_dir, job_type="train", extra_config=config_dict, tags=tags)

    try:
        device = accelerator.device
        logger.write(f"Accelerate distributed type: {accelerator.distributed_type}\n")
        logger.write(f"World size: {accelerator.num_processes}\n")
        logger.write(f"Local device: {device}\n")

        # Initialize model
        model_class = get_model(args.model_name)
        model_kwargs: dict[str, int | float | str] = {
            "num_modals": 4,
            "num_classes": args.num_classes
        }
        is_unime_model = args.model_name.lower().startswith("unime")
        if args.layer_decay is not None and is_unime_model:
            model_kwargs["layer_decay"] = args.layer_decay
        model_kwargs, uni_encoder_source = _apply_unime_model_overrides(args, model_kwargs)
        model = model_class(**model_kwargs)

        if args.compile:
            model = torch.compile(model, mode='default')
        logger.write(f"Primary GPU: {torch.cuda.get_device_name(device)}\n")
        if is_unime_model:
            if uni_encoder_source is None:
                default_checkpoint = getattr(getattr(model, "uni_encoder", None), "pretrained_path", None)
                if default_checkpoint is not None:
                    uni_encoder_source = f"Uni-Encoder default checkpoint: {default_checkpoint}"
            if uni_encoder_source is not None:
                logger.write(uni_encoder_source + "\n")
        run = wandb_utils.run()
        if run is not None:
            system_info = {
                "torch_version": torch.__version__,
                "cuda_available": True,
                "cuda_version": torch.version.cuda,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_count": torch.cuda.device_count(),
                "distributed_type": str(accelerator.distributed_type),
                "world_size": accelerator.num_processes,
            }
            try:
                run.config.update({"system": system_info}, allow_val_change=True)
            except Exception:
                pass

        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())  # type: ignore[arg-type]
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # type: ignore[arg-type]
        logger.write(f"Total parameters: {total_params:,}\n")
        logger.write(f"Trainable parameters: {trainable_params:,}\n")
        wandb_utils.log(
            {
                "model/total_params": int(total_params),
                "model/trainable_params": int(trainable_params),
            }
        )

        # Load data
        train_loader, val_loader, test_loader = setup_dataloader(args)
        train_dataset = cast(Sized, train_loader.dataset)
        val_dataset = cast(Sized, val_loader.dataset)

        # Log dataset information
        logger.log_metrics({
            'type': 'dataset_info',
            'train_batches': len(train_loader),
            'val_batches': len(val_loader),
            'train_samples': len(train_dataset),  # Ignore if dataset doesn't implement __len__
            'val_samples': len(val_dataset),
            'batch_size': args.batch_size
        })
        wandb_utils.log(
            {
                "data/train_batches": len(train_loader),
                "data/val_batches": len(val_loader),
                "data/train_samples": len(train_dataset),
                "data/val_samples": len(val_dataset),
                "data/batch_size": int(args.batch_size),
            }
        )
        logger.flush()

        # Call training function and get checkpoints for testing (top-k + final epoch)
        # Static typing only: model is a compiled/module instance at runtime.
        checkpoints_for_test = train_brats(
            args=args,
            model=model,  # type: ignore[arg-type]
            train_loader=train_loader,
            val_loader=val_loader,
            validation_results_csv=validation_results_csv,
            logger=logger,
            accelerator=accelerator
        )

        logger.write("Training completed successfully!\n")

        # Test selected checkpoints on test dataset
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and checkpoints_for_test and test_loader:
            logger.write("\nProceeding to test selected checkpoints...\n")
            test_top_k_models(
                model=model,  # type: ignore[arg-type]
                args=args,
                test_loader=test_loader,
                topk_checkpoints=checkpoints_for_test,
                test_results_csv=test_results_csv,
                logger=logger
            )
        elif not checkpoints_for_test:
            logger.write("No checkpoints available for testing.\n")
        elif not test_loader:
            logger.write("Test data loader not available.\n")
        accelerator.wait_for_everyone()

        logger.write("\nAll tasks completed successfully!\n")
    finally:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            wandb_utils.finish()
        if hasattr(accelerator, "end_training"):
            accelerator.end_training()
        logger.close()


if __name__ == "__main__":
    main()
