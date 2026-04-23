from __future__ import annotations

import os
import random
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


def _compute_batch_loss(
    model: Stage2LagrangianStateSpaceModel,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    observations, nwp_u, nwp_v = _move_batch_to_device(batch, device)
    outputs = model(observations, nwp_u, nwp_v)
    loss = outputs["loss"]
    return loss.mean() if loss.ndim > 0 else loss


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
) -> float:
    model.eval()
    local_loss_sum = 0.0
    local_loss_count = 0
    with torch.no_grad():
        for batch in loader:
            loss = _compute_batch_loss(model, batch, device)
            if torch.isfinite(loss):
                local_loss_sum += float(loss.detach().cpu())
                local_loss_count += 1
    return _distributed_mean(local_loss_sum, local_loss_count, device, context)


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
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=bool(config["training"].get("pin_memory", False)),
        persistent_workers=bool(config["training"].get("persistent_workers", False))
        and int(config["training"]["num_workers"]) > 0,
    )
    val_loader = DataLoader(
        bundle.val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        sampler=val_sampler,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=bool(config["training"].get("pin_memory", False)),
        persistent_workers=bool(config["training"].get("persistent_workers", False))
        and int(config["training"]["num_workers"]) > 0,
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

            local_loss_sum = 0.0
            local_loss_count = 0
            bad_batch_count = 0
            optimizer.zero_grad(set_to_none=True)

            for batch_index, batch in enumerate(train_loader, start=1):
                loss = _compute_batch_loss(model, batch, device)

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

                (loss / accumulation_steps).backward()

                bad_grad_name = _first_bad_gradient(_unwrap_model(model))
                skip_grad = _distributed_any(bad_grad_name is not None, device, context)
                if skip_grad:
                    bad_batch_count += 1
                    if _is_main_process(context):
                        print("Skipping optimizer step due to non-finite gradient on at least one rank.")
                    optimizer.zero_grad(set_to_none=True)
                    if bad_batch_count > int(config["training"].get("max_bad_batches_per_epoch", 100)):
                        raise RuntimeError("Too many non-finite batches in one epoch.")
                    continue

                should_step = (batch_index % accumulation_steps == 0) or (batch_index == len(train_loader))
                if should_step:
                    clip_grad_norm_(
                        model.parameters(),
                        max_norm=float(config["training"]["gradient_clip_norm"]),
                        error_if_nonfinite=False,
                    )
                    optimizer.step()

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

                local_loss_sum += float(loss.detach().cpu())
                local_loss_count += 1

            train_loss = _distributed_mean(local_loss_sum, local_loss_count, device, context)
            val_loss = (
                _evaluate_loader(model, val_loader, device, context)
                if len(bundle.val_dataset) > 0
                else float("nan")
            )

            if _is_main_process(context):
                history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

                if epoch % int(config["training"]["log_every"]) == 0:
                    print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

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
