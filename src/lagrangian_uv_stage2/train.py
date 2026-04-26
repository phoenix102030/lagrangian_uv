from __future__ import annotations

import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import load_config
from .data import build_data_bundle, serialize_scalers
from .models.state_space import Stage2LagrangianStateSpaceModel


LOSS_METRIC_KEYS = (
    "loss",
    "negative_log_likelihood",
    "normalized_negative_log_likelihood",
    "one_step_forecast_loss",
    "one_step_mae",
    "one_step_rmse",
    "rollout_forecast_loss",
    "rollout_mae",
    "rollout_rmse",
    "kernel_one_step_loss",
    "kernel_one_step_mae",
    "kernel_one_step_rmse",
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int


def _is_main_process(context: DistributedContext) -> bool:
    return context.rank == 0


def _init_distributed(config: dict[str, Any]) -> DistributedContext:
    use_ddp = bool(config["training"].get("use_ddp", True))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if use_ddp and world_size > 1:
        backend = str(config["training"].get("ddp_backend", "nccl"))
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        return DistributedContext(enabled=True, rank=rank, world_size=world_size, local_rank=local_rank)

    return DistributedContext(enabled=False, rank=0, world_size=1, local_rank=0)


def _cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _distributed_any(flag: bool, device: torch.device, context: DistributedContext) -> bool:
    if not context.enabled:
        return flag
    tensor = torch.tensor(1 if flag else 0, device=device, dtype=torch.int64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(tensor.item())


def _distributed_mean(sum_value: float, count_value: int, device: torch.device, context: DistributedContext) -> float:
    sum_tensor = torch.tensor(sum_value, device=device, dtype=torch.float64)
    count_tensor = torch.tensor(count_value, device=device, dtype=torch.float64)
    if context.enabled:
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    if float(count_tensor.item()) <= 0.0:
        return float("nan")
    return float((sum_tensor / count_tensor).item())


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    non_blocking = bool(device.type == "cuda")
    obs = batch["obs"].to(device, non_blocking=non_blocking)
    nwp_u = batch["nwp_u"].to(device, non_blocking=non_blocking)
    nwp_v = batch["nwp_v"].to(device, non_blocking=non_blocking)
    return obs, nwp_u, nwp_v


def _autocast_context(device: torch.device, enabled: bool, dtype_name: str):
    if not enabled:
        return nullcontext()
    dtype_lookup = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_lookup.get(dtype_name.lower())
    if dtype is None:
        raise ValueError("training.amp_dtype must be one of: float16, fp16, bfloat16, bf16")
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def _make_grad_scaler(device: torch.device, enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler(device.type, enabled=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.cuda.amp.GradScaler(enabled=True)


def _loader_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    training_cfg = config["training"]
    num_workers = int(training_cfg["num_workers"])
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(training_cfg.get("pin_memory", False)),
        "persistent_workers": bool(training_cfg.get("persistent_workers", False)) and num_workers > 0,
    }
    if num_workers > 0 and "prefetch_factor" in training_cfg:
        kwargs["prefetch_factor"] = int(training_cfg.get("prefetch_factor", 1))
    return kwargs


def _compute_batch_loss(
    model: Stage2LagrangianStateSpaceModel,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    loss, _ = _compute_batch_loss_and_metrics(model, batch, device)
    return loss


def _compute_batch_loss_and_metrics(
    model: Stage2LagrangianStateSpaceModel,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    observations, nwp_u, nwp_v = _move_batch_to_device(batch, device)
    outputs = model(observations, nwp_u, nwp_v)
    loss = outputs["loss"]
    loss = loss.mean() if loss.ndim > 0 else loss
    metrics = {"loss": loss}
    for key in LOSS_METRIC_KEYS:
        if key in outputs:
            value = outputs[key]
            metrics[key] = value.mean() if value.ndim > 0 else value
    return loss, metrics


def _add_loss_metrics(
    sums: dict[str, float],
    counts: dict[str, int],
    metrics: dict[str, torch.Tensor],
) -> None:
    for key, value in metrics.items():
        if not torch.isfinite(value):
            continue
        sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())
        counts[key] = counts.get(key, 0) + 1


def _distributed_metric_means(
    sums: dict[str, float],
    counts: dict[str, int],
    device: torch.device,
    context: DistributedContext,
) -> dict[str, float]:
    keys = sorted(set(sums) | set(counts))
    return {key: _distributed_mean(sums.get(key, 0.0), counts.get(key, 0), device, context) for key in keys}


def _unwrap_model(model: Stage2LagrangianStateSpaceModel) -> Stage2LagrangianStateSpaceModel:
    return model.module if hasattr(model, "module") else model


def _scheduled_sampling_ratio(config: dict[str, Any], epoch: int) -> float:
    training_cfg = config["training"]
    start = float(training_cfg.get("scheduled_sampling_start", 0.0))
    end = float(training_cfg.get("scheduled_sampling_end", start))
    decay_epochs = max(1, int(training_cfg.get("scheduled_sampling_decay_epochs", 1)))
    progress = min(max((epoch - 1) / decay_epochs, 0.0), 1.0)
    return start + (end - start) * progress


def _first_bad_parameter(model: Stage2LagrangianStateSpaceModel) -> str | None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            return name
    return None


def _first_bad_gradient(model: Stage2LagrangianStateSpaceModel) -> str | None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return name
    return None


def _evaluate_loader(
    model: Stage2LagrangianStateSpaceModel,
    loader: DataLoader,
    device: torch.device,
    context: DistributedContext,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
) -> dict[str, float]:
    model.eval()
    local_metric_sums: dict[str, float] = {}
    local_metric_counts: dict[str, int] = {}
    with torch.inference_mode():
        for batch in loader:
            with _autocast_context(device, amp_enabled, amp_dtype):
                loss, metrics = _compute_batch_loss_and_metrics(model, batch, device)
            if torch.isfinite(loss):
                _add_loss_metrics(local_metric_sums, local_metric_counts, metrics)
    return _distributed_metric_means(local_metric_sums, local_metric_counts, device, context)


def train(config: dict[str, Any]) -> Path:
    context = _init_distributed(config)
    _set_seed(int(config["project"]["seed"]) + context.rank)
    bundle = build_data_bundle(config, include_online=False)
    per_gpu_batch_size = int(config["training"]["batch_size"])
    accumulation_steps = max(1, int(config["training"].get("gradient_accumulation_steps", 1)))
    global_batch_size = per_gpu_batch_size * context.world_size
    effective_global_batch_size = global_batch_size * accumulation_steps

    configured_device = str(config["training"]["device"])
    if context.enabled and configured_device.startswith("cuda"):
        torch.cuda.set_device(context.local_rank)
        device = torch.device("cuda", context.local_rank)
    else:
        device = torch.device(configured_device)

    if device.type == "cuda":
        if bool(config["training"].get("allow_tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if bool(config["training"].get("cudnn_benchmark", True)):
            torch.backends.cudnn.benchmark = True

    amp_enabled = bool(config["training"].get("use_amp", False)) and device.type == "cuda"
    amp_dtype = str(config["training"].get("amp_dtype", "float16"))
    grad_scaler = _make_grad_scaler(device, amp_enabled)

    model = Stage2LagrangianStateSpaceModel(config, site_coords=bundle.site_coords).to(device)
    if context.enabled:
        ddp_kwargs = {"find_unused_parameters": bool(config["training"].get("find_unused_parameters", False))}
        if device.type == "cuda":
            model = DDP(model, device_ids=[context.local_rank], output_device=context.local_rank, **ddp_kwargs)
        else:
            model = DDP(model, **ddp_kwargs)
    elif (
        device.type == "cuda"
        and bool(config["training"].get("use_data_parallel", False))
        and torch.cuda.device_count() > 1
    ):
        if _is_main_process(context):
            print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    if _is_main_process(context):
        print(
            "Data bundle ready "
            f"(train_windows={len(bundle.train_dataset)}, val_windows={len(bundle.val_dataset)}, "
            f"online_loaded={bundle.online_sequence is not None})"
        )
        if context.enabled:
            backend = str(config["training"].get("ddp_backend", "nccl"))
            print(
                "DDP enabled "
                f"(backend={backend}, world_size={context.world_size}, per_gpu_batch_size={per_gpu_batch_size}, "
                f"global_batch_size={global_batch_size}, effective_global_batch_size={effective_global_batch_size}, "
                f"grad_accumulation_steps={accumulation_steps}, device={device.type})"
            )
        else:
            print(
                "Single-process training "
                f"(per_gpu_batch_size={per_gpu_batch_size}, global_batch_size={global_batch_size}, "
                f"effective_global_batch_size={effective_global_batch_size}, grad_accumulation_steps={accumulation_steps}, "
                f"device={device.type})"
            )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        eps=float(config["training"].get("adam_eps", 1.0e-8)),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    train_sampler = (
        DistributedSampler(
            bundle.train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
        )
        if context.enabled
        else None
    )
    val_sampler = (
        DistributedSampler(
            bundle.val_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=False,
        )
        if context.enabled and len(bundle.val_dataset) > 0
        else None
    )

    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **_loader_kwargs(config),
    )
    val_loader = DataLoader(
        bundle.val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        sampler=val_sampler,
        **_loader_kwargs(config),
    )

    output_dir = Path(config["project"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    try:
        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            model.train()
            _unwrap_model(model).set_scheduled_sampling_ratio(_scheduled_sampling_ratio(config, epoch))
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            local_metric_sums: dict[str, float] = {}
            local_metric_counts: dict[str, int] = {}
            local_grad_norm_sum = 0.0
            local_grad_norm_max = 0.0
            local_grad_norm_count = 0
            local_optimizer_steps = 0
            bad_batch_count = 0
            optimizer.zero_grad(set_to_none=True)

            for batch_index, batch in enumerate(train_loader, start=1):
                should_step = (batch_index % accumulation_steps == 0) or (batch_index == len(train_loader))
                sync_context = (
                    model.no_sync()
                    if context.enabled and accumulation_steps > 1 and not should_step
                    else nullcontext()
                )

                with sync_context:
                    with _autocast_context(device, amp_enabled, amp_dtype):
                        loss, metrics = _compute_batch_loss_and_metrics(model, batch, device)

                    skip_loss = _distributed_any(not torch.isfinite(loss), device, context)
                    if skip_loss:
                        bad_batch_count += 1
                        if _is_main_process(context):
                            start_index = batch["start_index"][0] if isinstance(batch["start_index"], torch.Tensor) else batch["start_index"]
                            print(f"Skipping non-finite loss at epoch={epoch} start_index={int(start_index)}")
                        optimizer.zero_grad(set_to_none=True)
                        if bad_batch_count > int(config["training"].get("max_bad_batches_per_epoch", 100)):
                            raise RuntimeError("Too many non-finite batches in one epoch.")
                        continue

                    scaled_loss = loss / accumulation_steps
                    if grad_scaler is not None:
                        grad_scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                if should_step:
                    if grad_scaler is not None:
                        grad_scaler.unscale_(optimizer)

                    bad_grad_name = _first_bad_gradient(_unwrap_model(model))
                    skip_grad = _distributed_any(bad_grad_name is not None, device, context)
                    if skip_grad:
                        bad_batch_count += 1
                        if _is_main_process(context):
                            print("Skipping optimizer step due to non-finite gradient on at least one rank.")
                        optimizer.zero_grad(set_to_none=True)
                        if grad_scaler is not None:
                            grad_scaler.update()
                        if bad_batch_count > int(config["training"].get("max_bad_batches_per_epoch", 100)):
                            raise RuntimeError("Too many non-finite batches in one epoch.")
                        continue

                    grad_norm = clip_grad_norm_(
                        model.parameters(),
                        max_norm=float(config["training"]["gradient_clip_norm"]),
                        error_if_nonfinite=False,
                    )
                    if torch.isfinite(grad_norm):
                        grad_norm_value = float(grad_norm.detach().cpu())
                        local_grad_norm_sum += grad_norm_value
                        local_grad_norm_max = max(local_grad_norm_max, grad_norm_value)
                        local_grad_norm_count += 1
                    if grad_scaler is not None:
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        optimizer.step()
                    local_optimizer_steps += 1

                    bad_param_name = _first_bad_parameter(_unwrap_model(model))
                    skip_param = _distributed_any(bad_param_name is not None, device, context)
                    if skip_param:
                        bad_batch_count += 1
                        if _is_main_process(context):
                            print("Skipping corrupted optimizer state because at least one parameter became non-finite.")
                        optimizer.zero_grad(set_to_none=True)
                        if bad_batch_count > int(config["training"].get("max_bad_batches_per_epoch", 100)):
                            raise RuntimeError("Too many non-finite batches in one epoch.")
                        continue
                    optimizer.zero_grad(set_to_none=True)

                _add_loss_metrics(local_metric_sums, local_metric_counts, metrics)

            train_metrics = _distributed_metric_means(local_metric_sums, local_metric_counts, device, context)
            train_loss = train_metrics.get("loss", float("nan"))
            val_metrics = (
                _evaluate_loader(model, val_loader, device, context, amp_enabled, amp_dtype)
                if len(bundle.val_dataset) > 0
                else {"loss": float("nan")}
            )
            val_loss = val_metrics.get("loss", float("nan"))
            grad_norm_mean = _distributed_mean(local_grad_norm_sum, local_grad_norm_count, device, context)
            grad_norm_max = local_grad_norm_max
            if context.enabled:
                grad_norm_max_tensor = torch.tensor(grad_norm_max, device=device, dtype=torch.float64)
                dist.all_reduce(grad_norm_max_tensor, op=dist.ReduceOp.MAX)
                grad_norm_max = float(grad_norm_max_tensor.item())

            if _is_main_process(context):
                history_entry = {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "optimizer_steps": float(local_optimizer_steps),
                    "grad_norm_mean": grad_norm_mean,
                    "grad_norm_max": grad_norm_max,
                }
                for key, value in train_metrics.items():
                    history_entry[f"train_{key}"] = value
                for key, value in val_metrics.items():
                    history_entry[f"val_{key}"] = value
                history.append(history_entry)

                if epoch % int(config["training"]["log_every"]) == 0:
                    train_nll = train_metrics.get("negative_log_likelihood", float("nan"))
                    train_one = train_metrics.get("one_step_forecast_loss", float("nan"))
                    train_roll = train_metrics.get("rollout_forecast_loss", float("nan"))
                    train_kernel = train_metrics.get("kernel_one_step_loss", float("nan"))
                    train_one_rmse = train_metrics.get("one_step_rmse", float("nan"))
                    train_roll_rmse = train_metrics.get("rollout_rmse", float("nan"))
                    val_nll = val_metrics.get("negative_log_likelihood", float("nan"))
                    val_one_rmse = val_metrics.get("one_step_rmse", float("nan"))
                    val_roll_rmse = val_metrics.get("rollout_rmse", float("nan"))
                    print(
                        f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                        f"steps={local_optimizer_steps} grad_norm={grad_norm_mean:.3e}/{grad_norm_max:.3e} "
                        f"train_nll={train_nll:.6f} train_one={train_one:.6f} "
                        f"train_roll={train_roll:.6f} train_kernel={train_kernel:.6f} "
                        f"train_rmse={train_one_rmse:.6f}/{train_roll_rmse:.6f} "
                        f"val_nll={val_nll:.6f} val_rmse={val_one_rmse:.6f}/{val_roll_rmse:.6f}"
                    )

                checkpoint = {
                    "model_state": _unwrap_model(model).state_dict(),
                    "config": config,
                    "site_coords": bundle.site_coords.tolist(),
                    "feature_names": bundle.feature_names,
                    "scalers": serialize_scalers(bundle),
                    "history": history,
                    "split_index": bundle.split_index,
                }
                torch.save(checkpoint, last_path)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(checkpoint, best_path)

            if device.type == "cuda" and bool(config["training"].get("empty_cache_each_epoch", True)):
                torch.cuda.empty_cache()

        if _is_main_process(context):
            print(f"Best checkpoint saved to {best_path}")
        return best_path
    finally:
        _cleanup_distributed(context)


def train_from_config(config_path: str | Path) -> Path:
    config = load_config(config_path)
    return train(config)
