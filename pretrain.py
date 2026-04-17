import os
from typing import cast

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed

from source.config import setup_seed
from source.logger import Logger, setup_logger
from source.pretrain import (
    get_pretrain_args,
    setup_pretrain_dataloader,
    pretrain_brats
)
from pretrain_models import get_model
from source.utils import is_accelerate_launch, wandb_utils


def main():
    """
    main script for pretraining.
    """
    args = get_pretrain_args()
    if args.gpu_ids and not is_accelerate_launch():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    mixed_precision = "no"
    if args.amp:
        mixed_precision = "bf16" if args.bfloat16 else "fp16"
    accelerator = Accelerator(mixed_precision=mixed_precision)

    setup_seed(args.seed)
    accelerate_set_seed(args.seed, device_specific=True)

    log = setup_logger(
        log_root=args.log_root,
        dataset_name=args.dataset_name,
        seed=args.seed,
        model_name=args.model_name,
        split_type=args.split_type,
        enabled=accelerator.is_main_process,
    )
    logger = cast(Logger, log["logger"])
    logger.write("=== Pretraining config ===\n")
    logger.write(str(args) + "\n")
    logger.write(f"Accelerate distributed type: {accelerator.distributed_type}\n")
    logger.write(f"World size: {accelerator.num_processes}\n")
    logger.write(f"Local device: {accelerator.device}\n")
    logger.flush()

    log_dir = os.path.join(args.log_root, f"{args.dataset_name}-{args.seed}-{args.split_type}", args.model_name)
    tags: list[str] = []
    if getattr(args, "amp", False):
        tags.append("amp")
    if getattr(args, "bfloat16", False):
        tags.append("bf16")
    if getattr(args, "compile", False):
        tags.append("compile")
    if getattr(args, "use_ema", False):
        tags.append("ema")
    if getattr(args, "train_ratio", None) is not None:
        tags.append(f"ratio{args.train_ratio}")
    if accelerator.is_main_process:
        wandb_utils.init_run(
            args,
            log_dir=log_dir,
            job_type="pretrain",
            extra_config={"save_path": log_dir},
            tags=tags,
        )

    try:
        train_loader = setup_pretrain_dataloader(args, num_processes=accelerator.num_processes)

        model = get_model(args.model_name)(
            in_channels=4,
            out_channels=4,
            pre_train=True,
            crop_size=args.crop_size,
            patch_mask_ratio=args.mask_ratio,
            original_shape=tuple(args.original_shape),
            num_mask_modalities=args.num_mask_modalities,
        )

        if accelerator.device.type == "cuda" and args.compile:
            model = torch.compile(model, mode='max-autotune')
        run = wandb_utils.run()
        if run is not None:
            system_info = {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "distributed_type": str(accelerator.distributed_type),
                "world_size": accelerator.num_processes,
                "local_device": str(accelerator.device),
            }
            if accelerator.device.type == "cuda":
                system_info.update(
                    {
                        "gpu_name": torch.cuda.get_device_name(accelerator.device),
                        "gpu_count": torch.cuda.device_count(),
                    }
                )
            try:
                run.config.update({"system": system_info}, allow_val_change=True)
            except Exception:
                pass

        total_params = sum(p.numel() for p in model.parameters())  # type: ignore[arg-type]
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # type: ignore[arg-type]
        wandb_utils.log(
            {
                "model/total_params": int(total_params),
                "model/trainable_params": int(trainable_params),
                "data/train_samples": len(train_loader.dataset),  # type: ignore[arg-type]
                "data/batch_size_per_process": int(args.batch_size),
                "data/world_size": int(accelerator.num_processes),
            }
        )

        pretrain_brats(
            args=args,
            model=model,  # type: ignore[arg-type]
            train_loader=train_loader,
            logger=logger,
            accelerator=accelerator,
        )

        logger.write("Done.\n")
    finally:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            wandb_utils.finish()
        if hasattr(accelerator, "end_training"):
            accelerator.end_training()
        logger.close()


if __name__ == "__main__":
    main()
