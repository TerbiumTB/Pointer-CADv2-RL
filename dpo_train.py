import argparse
import copy
import datetime
import math
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

from models.pointercad import PointerCAD
from models.processor import Text2CADProcessor
from rl.dpo import PointerCADDPO
from rl.dpo_data import PointerCADDataset, collate_dpo_pairs
from rl.likelihood import ScoringTemperatures


def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[str(name).lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model dtype {name!r}; expected {sorted(mapping)}."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        default="./config/dpo_train.yaml",
        help="DPO training YAML configuration.",
    )
    return parser.parse_args()


def load_yaml(path: str) -> Dict:
    with Path(path).open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML object in {path}.")
    return value


def load_model_checkpoint(model, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    missing = model.load_state_dict(state_dict, strict=False)
    if missing.missing_keys:
        logger.warning(
            "Missing checkpoint keys: {}",
            sorted(set(key.split(".")[0] for key in missing.missing_keys)),
        )
    if missing.unexpected_keys:
        logger.warning(
            "Unexpected checkpoint keys: {}", missing.unexpected_keys
        )


def set_deterministic_likelihood_mode(model) -> None:
    """Disable dropout and running-stat updates without disabling gradients."""
    model.eval()


def save_checkpoint(
    accelerator: Accelerator,
    model,
    optimizer,
    scheduler,
    output_dir: Path,
    global_step: int,
    epoch: int,
    config: Dict,
) -> None:
    accelerator.wait_for_everyone()
    model_state = accelerator.get_state_dict(model)
    if not accelerator.is_main_process:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"model_step_{global_step:08d}.pth"
    torch.save(
        {
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "config": config,
        },
        checkpoint_path,
    )
    logger.info("Saved DPO checkpoint to {}", checkpoint_path)


def reduce_metrics(
    accelerator: Accelerator, metrics: Dict[str, torch.Tensor], batch_size: int
) -> Dict[str, float]:
    names = sorted(metrics)
    weighted = torch.stack(
        [metrics[name].detach().float() * batch_size for name in names]
        + [torch.tensor(float(batch_size), device=accelerator.device)]
    )
    reduced = accelerator.reduce(weighted, reduction="sum")
    denominator = max(float(reduced[-1].item()), 1.0)
    return {
        name: float(reduced[index].item()) / denominator
        for index, name in enumerate(names)
    }


def make_dataloader(config: Dict, split: str, cached_reference_hash):
    dataset_config = config["dataset"]
    dataset = PointerCADDataset(
        episodes_root=dataset_config["episodes_root"],
        rollout_root=dataset_config["rollout_root"],
        preferences_root=dataset_config["preferences_root"],
        split=split,
        reference_checkpoint_hash=cached_reference_hash,
    )
    loader_config = config["training"]
    num_workers = int(loader_config.get("num_workers", 0))
    kwargs = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(
            loader_config.get("prefetch_factor", 2)
        )
    return DataLoader(
        dataset,
        batch_size=int(loader_config["batch_size"]),
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=bool(loader_config.get("pin_memory", False)),
        collate_fn=collate_dpo_pairs,
        **kwargs,
    )


@torch.no_grad()
def evaluate(
    accelerator: Accelerator,
    policy,
    objective: PointerCADDPO,
    dataloader,
) -> Dict[str, float]:
    policy.eval()
    metric_names = None
    local_sums = None
    for pairs in dataloader:
        with accelerator.autocast():
            output = objective.compute(
                policy_model=policy,
                pairs=pairs,
                device=accelerator.device,
            )
        if metric_names is None:
            metric_names = sorted(output.metrics)
            local_sums = torch.zeros(
                len(metric_names) + 1,
                dtype=torch.float32,
                device=accelerator.device,
            )
        batch_size = len(pairs)
        for index, name in enumerate(metric_names):
            local_sums[index] += (
                output.metrics[name].detach().float() * batch_size
            )
        local_sums[-1] += batch_size

    if local_sums is None:
        return {}
    reduced = accelerator.reduce(local_sums, reduction="sum")
    denominator = max(float(reduced[-1].item()), 1.0)
    return {
        name: float(reduced[index].item()) / denominator
        for index, name in enumerate(metric_names)
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    training = config["training"]
    accelerator = Accelerator(
        gradient_accumulation_steps=int(
            training.get("gradient_accumulation_steps", 1)
        ),
        mixed_precision=training.get("mixed_precision", "bf16"),
    )
    cuda_device_count = torch.cuda.device_count()
    if cuda_device_count == 0:
        raise RuntimeError(
            "DPO training requires CUDA, but torch.cuda.device_count() returned 0."
        )
    local_rank = accelerator.local_process_index % cuda_device_count
    torch.cuda.set_device(local_rank)
    logger.info(
        "Process {} uses CUDA device {} of {}: {}",
        accelerator.process_index,
        local_rank,
        cuda_device_count,
        torch.cuda.get_device_properties(local_rank),
    )
    set_seed(int(training.get("seed", 0)), device_specific=True)

    reference_config = config["reference"]
    reference_mode = reference_config.get("mode", "model")
    if reference_mode not in {"model", "cached"}:
        raise ValueError("reference.mode must be 'model' or 'cached'.")
    cached_reference_hash = (
        reference_config["checkpoint_hash"]
        if reference_mode == "cached"
        else None
    )
    train_loader = make_dataloader(config, "train", cached_reference_hash)
    validation_loader = make_dataloader(
        config, "validation", cached_reference_hash
    )
    if len(train_loader.dataset) == 0:
        raise ValueError("DPO training preference view is empty.")
    if len(validation_loader.dataset) == 0:
        logger.warning(
            "DPO validation preference view is empty; validation metrics will "
            "not be available."
        )

    model_config = config["model"]
    model_dtype = torch_dtype(model_config.get("dtype", "bfloat16"))
    logger.info(
        "Loading policy and reference weights with model dtype {} and "
        "mixed precision {}",
        model_dtype,
        accelerator.mixed_precision,
    )
    policy = PointerCAD(
        qwen_model=model_config["base_model"],
        dtype=model_dtype,
    )
    load_model_checkpoint(policy, model_config["checkpoint_path"])

    reference_model = None
    if reference_mode == "model":
        reference_path = reference_config.get(
            "checkpoint_path", model_config["checkpoint_path"]
        )
        if os.path.abspath(reference_path) == os.path.abspath(
            model_config["checkpoint_path"]
        ):
            reference_model = copy.deepcopy(policy)
        else:
            reference_model = PointerCAD(
                qwen_model=model_config["base_model"],
                dtype=model_dtype,
            )
            load_model_checkpoint(reference_model, reference_path)
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad = False

    processor = Text2CADProcessor.from_pretrained(
        pretrained_model_name_or_path=model_config["base_model"],
        padding_side="left",
    )
    optimizer = torch.optim.AdamW(
        policy.get_param_groups(
            base_lr=float(training["learning_rate"]),
            tau_lr=float(
                training.get("tau_learning_rate", training["learning_rate"])
            ),
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
    )
    updates_per_epoch = math.ceil(
        len(train_loader)
        / int(training.get("gradient_accumulation_steps", 1))
    )
    total_updates = max(updates_per_epoch * int(training["num_epochs"]), 1)
    scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=float(training.get("final_lr_factor", 0.0)),
        total_iters=total_updates,
    )
    (
        policy,
        optimizer,
        train_loader,
        validation_loader,
        scheduler,
    ) = accelerator.prepare(
        policy, optimizer, train_loader, validation_loader, scheduler
    )
    if reference_model is not None:
        reference_model = accelerator.prepare_model(
            reference_model,
            evaluation_mode=True,
        )
        set_deterministic_likelihood_mode(reference_model)
    set_deterministic_likelihood_mode(policy)
    logger.info(
        "DPO policy and reference likelihoods use eval mode to disable dropout "
        "and BatchNorm running-stat updates; policy gradients remain enabled."
    )

    temperatures = ScoringTemperatures(**config.get("scoring_temperatures", {}))
    objective = PointerCADDPO(
        processor=processor,
        rollout_root=config["dataset"]["rollout_root"],
        max_length=int(model_config.get("max_length", 3072)),
        beta=float(training["beta"]),
        label_smoothing=float(training.get("label_smoothing", 0.0)),
        temperatures=temperatures,
        reference_model=reference_model,
    )

    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(training["output_dir"]) / run_name
    log_interval = int(training.get("log_interval", 10))
    checkpoint_interval = int(training.get("checkpoint_interval", 100))
    validation_interval = int(training.get("validation_interval", 1))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(int(training["num_epochs"])):
        set_deterministic_likelihood_mode(policy)
        for batch_index, pairs in enumerate(train_loader):
            with accelerator.accumulate(policy):
                with accelerator.autocast():
                    output = objective.compute(
                        policy_model=policy,
                        pairs=pairs,
                        device=accelerator.device,
                    )
                accelerator.backward(output.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue
            global_step += 1
            accelerator.unwrap_model(policy).clamp_parameters()
            if global_step % log_interval == 0:
                metrics = reduce_metrics(
                    accelerator, output.metrics, len(pairs)
                )
                if accelerator.is_main_process:
                    summary = " ".join(
                        f"{name}={value:.4f}"
                        for name, value in sorted(metrics.items())
                    )
                    logger.info(
                        "epoch={} step={} lr={:.3e} {}",
                        epoch + 1,
                        global_step,
                        optimizer.param_groups[0]["lr"],
                        summary,
                    )
            if global_step % checkpoint_interval == 0:
                save_checkpoint(
                    accelerator,
                    policy,
                    optimizer,
                    scheduler,
                    output_dir,
                    global_step,
                    epoch,
                    config,
                )

        if (epoch + 1) % validation_interval == 0:
            validation_metrics = evaluate(
                accelerator,
                policy,
                objective,
                validation_loader,
            )
            if accelerator.is_main_process:
                if validation_metrics:
                    summary = " ".join(
                        f"{name}={value:.4f}"
                        for name, value in sorted(validation_metrics.items())
                    )
                    logger.info("validation epoch={} {}", epoch + 1, summary)
                else:
                    logger.warning(
                        "validation epoch={} skipped: no preference pairs",
                        epoch + 1,
                    )

    save_checkpoint(
        accelerator,
        policy,
        optimizer,
        scheduler,
        output_dir,
        global_step,
        int(training["num_epochs"]) - 1,
        config,
    )


if __name__ == "__main__":
    main()
