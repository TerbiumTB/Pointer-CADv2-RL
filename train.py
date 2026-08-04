import os
import sys
import math
import wandb
import torch
import argparse
import datetime
import gc
import argparse
import yaml
import shutil
from loguru import logger
from torch.optim.lr_scheduler import LinearLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from models.pointercad import PointerCAD
from dataset.dataset import get_dataloaders
from metrics.criterion import Text2CADCriterion
from metrics.metrics import LabelAccuracyCalculator, ParameterAccuracyCalculator, PointerAccuracyCalculator, SmoothedMetric
from misc import get_progress_bar
from models.processor import Text2CADProcessor



def setup(local_rank, global_rank, world_size, master_addr, master_port):
    # Initialize the distributed process group
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=global_rank
    )
    torch.cuda.set_device(local_rank)


def cleanup():
    dist.destroy_process_group()


def parse_config_file(config_file):
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data


def save_yaml_file(yaml_data, filename, output_dir):
    with open(os.path.join(output_dir, filename), "w+") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)


@torch.no_grad()
def all_reduce(var):
    if isinstance(var, torch.Tensor):
        if var.device.type != 'cuda':
            var = var.cuda()
        dist.all_reduce(var, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        return var / world_size
    elif isinstance(var, float):
        var_tensor = torch.tensor(var, dtype=torch.float32, device='cuda')
        dist.all_reduce(var_tensor, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        return var_tensor.item() / world_size
    else:
        raise TypeError(f"Unsupported type for all_reduce: {type(var)}. Only torch.Tensor and float are supported.")


@logger.catch()
def main(local_rank, rank_offset, world_size, master_addr, master_port):
    global_rank = local_rank + rank_offset
    setup(local_rank, global_rank, world_size, master_addr, master_port)

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, default="./config/train.yaml")
    args = parser.parse_args()
    config = parse_config_file(args.config_path)
    device = torch.device(f'cuda:{local_rank}')
    logger.info(f"Current RANK {global_rank} with PID {os.getpid()}")
    logger.info(f"Current Device {torch.cuda.get_device_properties(device)}")

    # -------------------------------- Load Model and Processor -------------------------------- #
    text2cad = PointerCAD(qwen_model=config["model"]["base_model"]).to(device)
    processor: Text2CADProcessor = Text2CADProcessor.from_pretrained(pretrained_model_name_or_path=config["model"]["base_model"], padding_side="left")

    if local_rank == 0: text2cad.model.print_trainable_parameters()

    # --------------------------- Prepare Log Directory -------------------------- #
    now = datetime.datetime.now()
    time_str = now.strftime("%H:%M")
    date_str = datetime.date.today()
    log_dir = os.path.join(config["train"]["log_dir"], f"{date_str}/{time_str}")

    if local_rank == 0: logger.info(f"Current Date {date_str} Time {time_str}\n")

    if global_rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        shutil.copy(args.config_path, log_dir)

    # Create the dataloader for train
    train_loader, val_loader = get_dataloaders(
        dataset_dir=config["dataset"]["dataset_dir"],
        split_filepath=config["dataset"]["split_filepath"],
        subsets=["train", "validation"],
        batch_sizes=[config["train"]["batch_size"], config["val"]["batch_size"]],
        num_workers=config["train"]["num_workers"],
        pin_memory=False,
        shuffle=True,
        prefetch_factor=config["train"]["prefetch_factor"],
        prompt_choices=config["dataset"]["prompt"]
    )

    # ---------------------- Resume Training from checkpoint --------------------- #
    checkpoint_file = os.path.join(log_dir, f"model.pth")
    checkpoint_file_load = config["train"]["checkpoint_path"]

    if checkpoint_file_load is not None and os.path.exists(checkpoint_file_load):
        if local_rank == 0: logger.info(f"Using saved checkpoint at {checkpoint_file_load}")
        checkpoint = torch.load(checkpoint_file_load, map_location=device)
        missing_keys_info = text2cad.load_state_dict(checkpoint["model"], strict=False)
        if local_rank == 0 and len(missing_keys_info.missing_keys) > 0:
            logger.warning(f"Missing keys in the checkpoint: {missing_keys_info.missing_keys}")
        if not config["train"]["force_retrain"]:
            start_epoch = checkpoint["epoch"] + 1
        else:
            start_epoch = 1
    else:
        start_epoch = 1
    if local_rank == 0: logger.info(f"Saving checkpoint at {checkpoint_file}")
    config["train"]["log_absolute_path"] = os.path.abspath(log_dir)


    # -------------------------------- Prepare DDP ------------------------------- #
    model = DDP(text2cad, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(
        model.module.get_param_groups(
            base_lr = config["train"]["lr"],
            tau_lr = config["train"]["tau_lr"],
            weight_decay = config["train"]["weight_decay"],
        )
    )
    scheduler = LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0,
        total_iters=config["train"]["num_epochs"] * math.ceil(len(train_loader) / config["train"].get("gradient_accumulation_steps", 1))  
    )
    criterion = Text2CADCriterion(
        weight_logits=config["criterion"]["weights"]["logits"],
        weight_label=config["criterion"]["weights"]["label"],
        weight_parameter=config["criterion"]["weights"]["parameter"],
        weight_pointer=config["criterion"]["weights"]["pointer"]
    )

    # Load optimizer state if resuming from checkpoint
    wandb_run_id = None
    if checkpoint_file_load is not None and os.path.exists(checkpoint_file_load) and not config["train"]["force_retrain"]:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        wandb_run_id = checkpoint.get("wandb", None)
        if local_rank == 0: logger.info(f"Resuming from epoch {start_epoch}")
    else:
        if local_rank == 0: logger.info(f"Training from scratch")
    
    torch.cuda.synchronize()
    dist.barrier()

    # -------------------------------- Train Model ------------------------------- #
    train_model(
        model=model,
        processor=processor,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=(train_loader, val_loader),
        epochs=(start_epoch, config["train"]["num_epochs"]),
        checkpoint_file=checkpoint_file,
        config=config,
        wandb_run_id=wandb_run_id
    )


def train_model(
    model,
    processor,
    criterion: Text2CADCriterion,
    optimizer,
    scheduler,
    dataloader,
    epochs,
    checkpoint_file,
    config,
    wandb_run_id=None,
):
    global_rank = dist.get_rank()
    local_rank = global_rank % torch.cuda.device_count()
    # world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')

    train_loader, val_loader = dataloader
    start_epoch, num_epochs = epochs

    if global_rank == 0:
        if wandb_run_id is not None:
            wandb.init(
                project="CADv3",
                name=f"{os.path.basename(os.path.dirname(os.path.abspath(__file__)))}",
                tags=["SFT"],
                config=config,
                resume="allow",
                id=wandb_run_id
            )
        else:
            wandb.init(
                project="CADv3",
                name=f"{os.path.basename(os.path.dirname(os.path.abspath(__file__)))}",
                tags=["SFT"],
                config=config
            )

    # val_loader.sampler.set_epoch(0)
    # validation_one_epoch(
    #     val_loader=val_loader,
    #     model=model,
    #     processor=processor,
    #     epoch=start_epoch - 1,
    #     total_batch=2,
    #     bit=config["model"]["bit"],
    # )

    # ---------------------------------- Training ---------------------------------- #
    for epoch in range(start_epoch, num_epochs + 1):
        # ------------------------------- Single Epoch ------------------------------- #
        model.train()
        optimizer.zero_grad()

        train_loader.sampler.set_epoch(epoch - 1)
        val_loader.sampler.set_epoch(epoch)

        # Train for one epoch
        label_accuracy_calc, parameter_accuracy_calc, pointer_accuracy_calc = LabelAccuracyCalculator(), ParameterAccuracyCalculator(), PointerAccuracyCalculator()
        label_accuracy_avg, parameter_accuracy_avg, pointer_accuracy_avg = SmoothedMetric(), SmoothedMetric(50), SmoothedMetric(50)

        with get_progress_bar(
            train_loader,
            ascii=True,
            desc=f"\033[94mText2CAD\033[0m: Epoch [{epoch}/{num_epochs}]✨",
            dynamic_ncols=True
        ) as pbar:
            for step, iter_dict in enumerate(pbar):
                messages = []
                for prompt, plan in zip(iter_dict["prompt"], iter_dict["plan"]):
                    message = [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": "You are an expert mechanical engineer. Based on the user's text requirements, generate the corresponding CAD model design."},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "brep"},
                                {"type": "text", "text": prompt},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": plan},
                                {"type": "cad"},
                            ],
                        },
                    ]
                    messages.append(message)
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                breps = iter_dict["graph"]
                gt_parameter_maps = iter_dict["parameter_map"]
                gt_labels = iter_dict["label"]
                gt_parameters = iter_dict["parameter"]
                gt_pointers = iter_dict["pointer"]
                gt_pointer_sources = iter_dict["pointer_source"]
                inputs = processor(text=text, breps=breps, parameter_maps=gt_parameter_maps, labels=gt_labels, parameters=gt_parameters, pointers=gt_pointers, max_length=3072)

                inputs = inputs.to(device)
                gt_logits = inputs["input_ids"]
                gt_labels = inputs["labels"]
                gt_parameters = inputs["parameters"]
                gt_pointers = inputs["pointers"]

                # ------------------ Forward pass ------------------ #
                pred_logits, pred_labels, pred_parameters, pred_pointers, param_embeds, param_tau, ref_pointer_crv, ref_pointer_srf, pointer_tau, standard_plane_pointer = model(**inputs)

                # ----------------------------- Loss Calculation ----------------------------- #
                loss, loss_dict = criterion(pred_logits, gt_logits, pred_labels, gt_labels, pred_parameters, gt_parameters, param_embeds, param_tau, pred_pointers, gt_pointers, gt_pointer_sources, ref_pointer_crv, ref_pointer_srf, pointer_tau, standard_plane_pointer)

                # -------------------------- Gradient Accumulation --------------------------- #
                gradient_accumulation_steps = config["train"].get("gradient_accumulation_steps", 1)
                loss = loss / gradient_accumulation_steps
                loss.backward()

                # ------------------------------- Backward pass ------------------------------ #
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(
                        parameters=model.parameters(), max_norm=0.9, norm_type=2.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    model.module.clamp_parameters()
                    scheduler.step()

                # --------------------------- Log loss and accuracy -------------------------- #
                # Compute accuracy
                with torch.no_grad():
                    label_accuracy_avg.update(label_accuracy_calc.calculateFromProbability2D(pred_labels, gt_labels))
                    parameter_accuracy_avg.update(parameter_accuracy_calc.calculateFromParameter2D(pred_parameters, gt_parameters, gt_labels, param_embeds))
                    pointer_accuracy_avg.update(pointer_accuracy_calc.calculateFromPointer2D(pred_pointers, gt_pointer_sources, gt_labels, ref_pointer_crv, ref_pointer_srf, standard_plane_pointer.detach()))
                    label_accuracy = label_accuracy_avg.average()
                    parameter_accuracy = parameter_accuracy_avg.average()
                    pointer_accuracy = pointer_accuracy_avg.average()

                    label_accuracy = all_reduce(label_accuracy)
                    parameter_accuracy = all_reduce(parameter_accuracy)
                    pointer_accuracy = all_reduce(pointer_accuracy)

                updated_dict = {
                    "loss": f"{loss.item():.4f}",
                    "lacc": f"{label_accuracy:.2f}",
                    "pacc": f"{parameter_accuracy:.2f}",
                    "poacc": f"{pointer_accuracy:.2f}",
                }
                # Update the progress bar
                pbar.set_postfix(updated_dict)

                # ---------------------------- Add to WandB ---------------------------- #
                if global_rank == 0 and wandb.run is not None:
                    wandb.log({
                        "epoch": epoch + pbar.n / pbar.total,
                        "train/loss": loss.item(),
                        "train/loss_logit": loss_dict["logit"],
                        "train/loss_label": loss_dict["label"],
                        "train/loss_parameter": loss_dict["parameter"],
                        "train/loss_pointer": loss_dict["pointer"],
                        "train/label_accuracy": label_accuracy,
                        "train/parameter_accuracy": parameter_accuracy,
                        "train/pointer_accuracy": pointer_accuracy,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    })

        # # ---------------- Perform Validation ---------------- #
        # if epoch % config["val"]["interval"] == 0:
        #     validation_one_epoch(
        #         val_loader=val_loader,
        #         model=model,
        #         processor=processor,
        #         epoch=epoch,
        #         total_batch=config["val"]["val_batch"],
        #         bit=config["model"]["bit"],
        #     )

        # ---------------- Save the model weights and optimizer state ---------------- #
        if epoch % config["train"]["checkpoint_interval"] == 0 and global_rank == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "wandb": wandb.run.id if wandb.run is not None else None,
                },
                checkpoint_file,
            )

            if not config["train"].get("checkpoint_overwrite", False):
                log_dir = os.path.dirname(checkpoint_file)
                step_checkpoint_file = os.path.join(log_dir, f"model_step_{epoch}.pth")
                shutil.copyfile(checkpoint_file, step_checkpoint_file)

    cleanup()

    # Close the tensorboard summary writer
    wandb.finish()
    if global_rank == 0: logger.info("Training Finished.")


@torch.no_grad()
def validation_one_epoch(
    val_loader,
    model,
    processor,
    epoch=0,
    topk=1,
    total_batch=5,
    bit=8
):
    """
    Perform one validation epoch on the given validation loader.

    Args:
        val_loader (torch.utils.data.DataLoader): DataLoader for validation dataset.
        model (torch.nn.Module): The model to be validated.
        epoch (int, required): Current epoch number. Defaults to 0.
        writer (SummaryWriter, optional): TensorBoard SummaryWriter for logging. Defaults to None.
        topk (int, optional): Hybrid Sampling. Defaults to 5. Set to 1 for top-1
        config (dict, optional): Additional configuration parameters. Defaults to None.

    Returns:
        tuple: Mean Sequence Token Accuracy (mean_seq_token_acc)
    """
    global_rank = dist.get_rank()
    local_rank = global_rank % torch.cuda.device_count()
    # world_size = dist.get_world_size()

    device = torch.device(f'cuda:{local_rank}')
    model.eval()

    value_accuracy_calc, pointer_accuracy_calc, scale_accuracy_calc = LabelAccuracyCalculator(bit=bit), PointerAccuracyCalculator(), ScaleAccuracyCalculator()
    with torch.no_grad():
        with get_progress_bar(
            total=total_batch * topk,
            ascii=True,
            desc=f"Validation✨",
            dynamic_ncols=True
        ) as pbar:
            val_iter = iter(val_loader)
            value_accuracy_avg_t, value_accuracy_avg_k, value_accuracy_avg_v, pointer_accuracy_avg, scale_accuracy_avg = SmoothedMetric(total_batch), SmoothedMetric(total_batch), SmoothedMetric(total_batch), SmoothedMetric(total_batch), SmoothedMetric(total_batch)
            for _ in range(total_batch):
                iter_dict = next(val_iter)
                messages = []
                for prompt in iter_dict["prompt"]:
                    message = [
                        {"role": "system", "content": "You are an expert mechanical engineer. Based on the user's text requirements, generate the corresponding CAD model design."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "brep"},
                                {"type": "text", "text": prompt},
                            ],
                        },
                    ]
                    messages.append(message)
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                breps = iter_dict["graph"]
                gt_value_sources = iter_dict["value_source"]
                gt_pointer_sources = iter_dict["pointer_source"]
                gt_scales = iter_dict["scale"]
                inputs = processor(text=text, breps=breps, values=None, pointers=None, max_length=3072)

                inputs = inputs.to(device)

                # Create a copy of the sequence dictionaries, and take only the start token
                value_accuracy_avg_t_topk, value_accuracy_avg_k_topk, value_accuracy_avg_v_topk, pointer_accuracy_avg_topk, scale_accuracy_avg_topk = SmoothedMetric(topk), SmoothedMetric(topk), SmoothedMetric(topk), SmoothedMetric(topk), SmoothedMetric(topk)
                for topk_index in range(1, topk + 1):
                    # Autoregressive Prediction (topk outputs per sample)
                    generated_value, generated_pointer, generated_scale = model.module.predict(**inputs)
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Sync
                    torch.cuda.synchronize()
                    dist.barrier()

                    # Calculate accuracies
                    vt, vk, vv = value_accuracy_calc.calculateFromLabel2D([f.cpu() for f in generated_value], gt_value_sources)
                    pt = pointer_accuracy_calc.calculateFromLabel2D(generated_pointer, gt_pointer_sources)
                    st = scale_accuracy_calc.calculateFromBatch(generated_scale.cpu(), gt_scales)
                    value_accuracy_avg_t_topk.update(vt)
                    value_accuracy_avg_k_topk.update(vk)
                    value_accuracy_avg_v_topk.update(vv)
                    pointer_accuracy_avg_topk.update(pt)
                    scale_accuracy_avg_topk.update(st)

                    # Update progress bar with current accuracy information
                    pbar.set_postfix({
                        "vacc": f"{vt:.2f}",
                        "pacc": f"{pt:.2f}",
                        "sacc": f"{st:.2f}%"
                    })
                    pbar.update(1)

                value_accuracy_avg_t.update(value_accuracy_avg_t_topk.max())
                value_accuracy_avg_k.update(value_accuracy_avg_k_topk.max())
                value_accuracy_avg_v.update(value_accuracy_avg_v_topk.max())
                pointer_accuracy_avg.update(pointer_accuracy_avg_topk.max())
                scale_accuracy_avg.update(scale_accuracy_avg_topk.max())

        value_accuracy_t = all_reduce(value_accuracy_avg_t.average())
        value_accuracy_k = all_reduce(value_accuracy_avg_k.average())
        value_accuracy_v = all_reduce(value_accuracy_avg_v.average())
        pointer_accuracy = all_reduce(pointer_accuracy_avg.average())
        scale_accuracy = all_reduce(scale_accuracy_avg.average())

        if local_rank == 0: logger.success("Validation CAD Value Accuracy: {}", value_accuracy_t)
        if local_rank == 0: logger.success("Validation CAD Value Accuracy - Special Token: {}", value_accuracy_k)
        if local_rank == 0: logger.success("Validation CAD Value Accuracy - Value Token: {}", value_accuracy_v)
        if local_rank == 0: logger.success("Validation Pointer Accuracy: {}", pointer_accuracy)
        if local_rank == 0: logger.success("Validation Scale Accuracy: {}%", scale_accuracy)

        if global_rank == 0 and wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "validation/value_accuracy": value_accuracy_t,
                "validation/value_accuracy/special_token": value_accuracy_k,
                "validation/value_accuracy/value_token": value_accuracy_v,
                "validation/pointer_accuracy": pointer_accuracy,
                "validation/scale_accuracy": scale_accuracy,
            })

        # Calculate mean accuracies for sketches and extrusions
        gc.collect()
        torch.cuda.empty_cache()

        return value_accuracy_t, value_accuracy_k, value_accuracy_v, pointer_accuracy, scale_accuracy


if __name__ == "__main__":
    num_procs = torch.cuda.device_count()
    world_size = int(os.getenv("WORLD_SIZE", num_procs))
    master_addr = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "32501")
    rank_offset = int(os.getenv("GLOBAL_RANK_OFFSET", 0))

    try:
        mp.spawn(main, args=(rank_offset, world_size, master_addr, master_port,), nprocs=num_procs)
    except KeyboardInterrupt:
        print("Ctrl+C received, cleaning up...")
        cleanup()
        sys.exit(0)
