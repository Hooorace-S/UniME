import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_WANDB: Any | None = None
_RUN: Any | None = None
_ENABLED: bool = False


def _safe_import_wandb() -> Any | None:
    global _WANDB  # noqa: PLW0603
    if _WANDB is not None:
        return _WANDB
    try:
        import wandb  # type: ignore
    except Exception:
        _WANDB = None
        return None
    _WANDB = wandb
    return _WANDB


def _to_plain_dict(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    if is_dataclass(args):
        return asdict(args)  # type: ignore[union-attr]
    if hasattr(args, "to_dict") and callable(getattr(args, "to_dict")):
        return dict(args.to_dict())
    if hasattr(args, "__dict__"):
        return dict(vars(args))
    return {}


def _default_group(args_dict: Mapping[str, Any], job_type: str | None) -> str:
    dataset = str(args_dict.get("dataset_name", "dataset"))
    split = str(args_dict.get("split_type", "split"))
    model = str(args_dict.get("model_name", job_type or "run"))
    return f"{dataset}-{split}-{model}"


def _default_run_name(args_dict: Mapping[str, Any], job_type: str | None) -> str:
    dataset = str(args_dict.get("dataset_name", "dataset"))
    split = str(args_dict.get("split_type", "split"))
    model = str(args_dict.get("model_name", job_type or "run"))
    seed = args_dict.get("seed", None)
    train_ratio = args_dict.get("train_ratio", None)

    seed_part = f"-seed{seed}" if seed is not None else ""
    ratio_part = f"-ratio{train_ratio}" if train_ratio is not None else ""
    return f"{model}-{dataset}-{split}{seed_part}{ratio_part}"


def init_run(
    args: Any,
    log_dir: str,
    job_type: str,
    extra_config: Mapping[str, Any] | None = None,
    tags: Sequence[str] | None = None,
) -> Any | None:
    """
    Initialize a wandb run once per process.

    When args.wandb_mode == "disabled", this is a no-op.
    """
    global _RUN, _ENABLED  # noqa: PLW0603

    args_dict = _to_plain_dict(args)
    wandb_mode = str(args_dict.get("wandb_mode", "online"))
    if wandb_mode == "disabled":
        _ENABLED = False
        _RUN = None
        return None

    wandb = _safe_import_wandb()
    if wandb is None:
        _ENABLED = False
        _RUN = None
        return None

    # Reuse the active run if already initialized (e.g., sweeps).
    if getattr(wandb, "run", None) is not None:
        _RUN = wandb.run
        _ENABLED = True
        return _RUN

    project = str(args_dict.get("wandb_project", "U-Bench"))
    entity = args_dict.get("wandb_entity", None)
    group = args_dict.get("wandb_group", None) or _default_group(args_dict, job_type=job_type)
    name = args_dict.get("wandb_run_name", None) or _default_run_name(args_dict, job_type=job_type)

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            mode=wandb_mode,
            dir=log_dir,
            config=args_dict,
            job_type=job_type,
            tags=list(tags) if tags else None,
        )
    except Exception as exc:
        try:
            print(f"[wandb] init failed ({exc!s}); continuing without W&B.")
        except Exception:
            pass
        _ENABLED = False
        _RUN = None
        return None

    _RUN = run
    _ENABLED = run is not None

    if run is None:
        return None

    derived_config: dict[str, Any] = {
        "log_dir": str(log_dir),
        "cwd": os.getcwd(),
    }
    if extra_config:
        derived_config.update(dict(extra_config))
    try:
        run.config.update(derived_config, allow_val_change=True)
    except Exception:
        pass

    # Improve charts: keep train/val/test/pretrain metrics aligned to epochs.
    try:
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")
        run.define_metric("test/*", step_metric="epoch")
        run.define_metric("pretrain/*", step_metric="epoch")
        run.define_metric("val/composite_score", summary="max")
        run.define_metric("val/dice_WT", summary="max")
        run.define_metric("val/dice_TC", summary="max")
        run.define_metric("val/dice_ET", summary="max")
        run.define_metric("val/dice_ET_postpro", summary="max")
    except Exception:
        pass

    return run


def enabled() -> bool:
    return bool(_ENABLED and _RUN is not None)


def run() -> Any | None:
    return _RUN


def log(metrics: Mapping[str, Any], step: int | None = None) -> None:
    if not enabled():
        return
    if not metrics:
        return

    payload: dict[str, Any] = {}
    for k, v in metrics.items():
        if hasattr(v, "item") and callable(getattr(v, "item")):
            try:
                payload[k] = v.item()
                continue
            except Exception:
                pass
        payload[k] = v

    try:
        if step is None:
            _RUN.log(payload)  # type: ignore[union-attr]
        else:
            _RUN.log(payload, step=int(step))  # type: ignore[union-attr]
    except Exception:
        pass


def log_table(
    key: str,
    columns: Sequence[str],
    data: Sequence[Sequence[Any]],
    step: int | None = None,
) -> None:
    if not enabled():
        return
    wandb = _safe_import_wandb()
    if wandb is None:
        return
    try:
        table = wandb.Table(columns=list(columns), data=list(data))
        log({key: table}, step=step)
    except Exception:
        pass


def save(
    path_or_glob: str,
    base_path: str | None = None,
    policy: str | None = "end",
) -> None:
    if not enabled():
        return
    try:
        if base_path is None:
            _RUN.save(path_or_glob, policy=policy)  # type: ignore[union-attr]
        else:
            _RUN.save(path_or_glob, base_path=base_path, policy=policy)  # type: ignore[union-attr]
    except Exception:
        pass


def log_artifact(
    paths: str | Sequence[str],
    name: str,
    type: str,
    aliases: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    description: str | None = None,
) -> Any | None:
    if not enabled():
        return None
    wandb = _safe_import_wandb()
    if wandb is None:
        return None
    try:
        artifact = wandb.Artifact(
            name=name,
            type=type,
            metadata=dict(metadata) if metadata else None,
            description=description,
        )
        paths_list = [paths] if isinstance(paths, str) else list(paths)
        for p in paths_list:
            p_path = Path(p)
            if p_path.is_dir():
                artifact.add_dir(str(p_path))
            else:
                artifact.add_file(str(p_path))
        _RUN.log_artifact(artifact, aliases=list(aliases) if aliases else None)  # type: ignore[union-attr]
        return artifact
    except Exception:
        return None


def finish() -> None:
    global _RUN, _ENABLED  # noqa: PLW0603
    if _RUN is None:
        _ENABLED = False
        return
    try:
        _RUN.finish()
    except Exception:
        pass
    finally:
        _RUN = None
        _ENABLED = False
