import os
import gc
import dgl
import copy
import yaml
import wandb
import shutil
import argparse
import datetime
import warnings
import numpy as np
import logging.config
from loguru import logger

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LinearLR
from torch.nn.parallel import DistributedDataParallel as DDP

from misc import TOKEN, STANDARD_PLANES
from models.pointercad import PointerCAD
from dataset.dataset import get_dataloaders
from metrics.criterion import MAPELoss
from metrics.metrics import LabelAccuracyCalculator, PointerAccuracyCalculator, SmoothedMetric
from misc import get_progress_bar
from models.processor import Text2CADProcessor
from metrics.rewards import Text2CADReward



# ---------------------------------------------------------------------------- #
#                            Text2CAD Training Code                            #
# ---------------------------------------------------------------------------- #



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
    elif isinstance(var, float) or isinstance(var, int):
        var_tensor = torch.tensor(var, dtype=torch.float32, device='cuda')
        dist.all_reduce(var_tensor, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        return var_tensor.item() / world_size
    else:
        raise TypeError(f"Unsupported type for all_reduce: {type(var)}. Only torch.Tensor and float are supported.")


def selective_log_softmax(pred_logits, input_ids, pred_labels, gt_labels, pred_parameters, gt_parameters, param_embeds, param_tau, 
                          pred_pointers, gt_pointers, ref_pointer_crv, ref_pointer_srf, pointer_tau, standard_plane_pointers,
                          temperature_plan, temperature_label, temperature_parameter, temperature_pointer):
    """
    Compute the log probabilities for the tokens specified in input_ids using a selective log-softmax.
    """
    selected_log_probs = []
    ids_masks = (input_ids != 151667) & (input_ids != 151670) & (input_ids != 151673)
    for logits, ids, ids_mask, pred_label, gt_label, pred_param, gt_param, param_embed, pred_pointer, gt_pointer, crv_pointer, srf_pointer in zip(pred_logits, input_ids, ids_masks, pred_labels, gt_labels, pred_parameters, gt_parameters, param_embeds, pred_pointers, gt_pointers, ref_pointer_crv, ref_pointer_srf):
        log_prob = None
        command_idx = None

        ##################### Logits Prob #####################
        ids_masked = ids[ids_mask]
        gen_idx = (ids_masked == 151644).nonzero(as_tuple=True)[0]
        gen_idx = gen_idx[gen_idx + 3 < len(ids_masked)]  # 防止越界
        gen_idx = (gen_idx[(ids_masked[gen_idx + 1] == 77091) & (ids_masked[gen_idx + 2] == 198)] + 3).tolist()
        if len(gen_idx):
            ids_masked = ids_masked[gen_idx[0]:]
            ids_log_probs = F.log_softmax(logits[gen_idx[0]:] / temperature_plan, dim=-1)  # Shape: (seq_len, vocab_size)
            ids_selected_log_probs = ids_log_probs.gather(dim=-1, index=ids_masked[:len(ids_log_probs)].unsqueeze(-1))
            log_prob = ids_selected_log_probs.squeeze(-1)
            command_idx = (ids_masked == 151671).nonzero(as_tuple=True)[0].tolist()
            if len(command_idx):
                command_idx = command_idx[0]
            else:
                selected_log_probs.append(log_prob)
                continue
        else:
            selected_log_probs.append(torch.tensor([], device=logits.device, dtype=logits.dtype))
            continue

        ##################### Label Prob #####################
        label_log_probs = F.log_softmax(pred_label / temperature_label, dim=-1)  # Shape: (seq_len, len(TOKEN))
        label_selected_log_probs = label_log_probs.gather(dim=-1, index=gt_label[:len(label_log_probs)].unsqueeze(-1))
        command_log_prob = label_selected_log_probs.squeeze(-1)

        for idx, label in zip(range(command_log_prob.shape[0]), gt_label):
            ##################### Parameter Prob #####################
            if label == TOKEN.index("<|length_value|>"):
                assert gt_param[idx] > 0, f"Parameter index out of range: {gt_param[idx]}"
                pred = pred_param[idx].expand(param_embed["length"].size(0), -1)
                cos_sim = F.cosine_similarity(pred, param_embed["length"], dim=1) * param_tau
                param_log_probs = F.log_softmax(cos_sim / temperature_parameter, dim=-1)
                param_selected_log_prob = param_log_probs[gt_param[idx] - 1]
                command_log_prob[idx] += param_selected_log_prob
            elif label == TOKEN.index("<|angle_value|>"):
                assert gt_param[idx] > 0, f"Parameter index out of range: {gt_param[idx]}"
                pred = pred_param[idx].expand(param_embed["angle"].size(0), -1)
                cos_sim = F.cosine_similarity(pred, param_embed["angle"], dim=1) * param_tau
                param_log_probs = F.log_softmax(cos_sim / temperature_parameter, dim=-1)
                param_selected_log_prob = param_log_probs[gt_param[idx] - 1]
                command_log_prob[idx] += param_selected_log_prob

            ##################### Pointer Prob #####################
            if label == TOKEN.index("<|pointer_enable|>"):
                assert gt_pointer[idx] >= -len(STANDARD_PLANES), f"Pointer index out of range: {gt_pointer[idx]}"

                if gt_label[idx - 1] == TOKEN.index("<|sketch_start|>"):
                    pointer = torch.vstack([standard_plane_pointers, srf_pointer])
                    pred = pred_pointer[idx].expand(pointer.size(0), -1)
                    cos_sim = F.cosine_similarity(pred, pointer, dim=1) * pointer_tau
                    pointer_log_probs = F.log_softmax(cos_sim / temperature_pointer, dim=-1)
                    pointer_selected_log_prob = pointer_log_probs[gt_pointer[idx] + len(STANDARD_PLANES)]
                else:
                    pred = pred_pointer[idx].expand(crv_pointer.size(0), -1)
                    cos_sim = F.cosine_similarity(pred, crv_pointer, dim=1) * pointer_tau
                    pointer_log_probs = F.log_softmax(cos_sim / temperature_pointer, dim=-1)
                    pointer_selected_log_prob = pointer_log_probs[gt_pointer[idx]]
                command_log_prob[idx] += pointer_selected_log_prob

        selected_log_probs.append(torch.cat([log_prob[:command_idx+1], command_log_prob, log_prob[command_idx+1:]], dim=0))

    return selected_log_probs


def compute_log_probs_and_aux(model, inputs, temperature_plan, temperature_label, temperature_parameter, temperature_pointer, dummy_loss=False):
    """
    Compute per-token log probabilities for a subset of tokens (typically the completion tokens).
    """
    # Run the model forward pass and obtain logits.
    pred_logits, pred_labels, pred_parameters, pred_pointers, param_embeds, param_tau, ref_pointer_crv, ref_pointer_srf, pointer_tau, standard_plane_pointer = model(**inputs)  # Shape: (batch_size, total_seq_len, vocab_size)

    # Compute the log probabilities for the selected tokens.
    log_probs = selective_log_softmax(pred_logits, inputs["input_ids"][:, 1:], pred_labels, inputs["labels"], pred_parameters, inputs["parameters"], param_embeds, 
                                      param_tau, pred_pointers, inputs["pointers"], ref_pointer_crv, ref_pointer_srf, pointer_tau, standard_plane_pointer, 
                                      temperature_plan, temperature_label, temperature_parameter, temperature_pointer)

    if dummy_loss:
        return log_probs, pointer_tau
    else:
        return log_probs


def set_tqdm_desc(pbar, desc):
    local_rank = dist.get_rank() % torch.cuda.device_count()
    
    if local_rank != 0:
        return

    pbar.desc = f"\033[94mPointerCAD\033[0m✨ ({desc})"
    pbar.refresh()


def compute_means_without_invalid_data(metric):
    if len(metric) == 0:
        return 0
    
    np_metric = np.array(metric)
    valid_metric = np_metric[np_metric >= 0]
    return valid_metric.mean() if len(valid_metric) > 0 else 0


@logger.catch()
def main(local_rank, rank_offset, world_size, master_addr, master_port):
    global_rank = local_rank + rank_offset
    setup(local_rank, global_rank, world_size, master_addr, master_port)

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, default="./config/rl_train.yaml")
    args = parser.parse_args()
    config = parse_config_file(args.config_path)
    device = torch.device(f'cuda:{local_rank}')
    logger.info(f"Current RANK {global_rank} with PID {os.getpid()}")
    logger.info(f"Current Device {torch.cuda.get_device_properties(device)}")

    assert config["train"]["num_generations"] % config["train"]["grad_accum_steps"] == 0, "num_generations must be divisible by grad_accum_steps."

    # -------------------------------- Load Model -------------------------------- #
    text2cad = PointerCAD(qwen_model=config["model"]["base_model"]).to(device)

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
        batch_sizes=[1, config["val"]["batch_size"]],
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
            logger.warning(f"Missing keys in the checkpoint: {[key.split('.')[0] for key in missing_keys_info.missing_keys]}")
        if "epoch" in checkpoint:
            config["train"]["checkpoint_epoch"] = checkpoint["epoch"]
    else:
        logger.warning(f"No checkpoint found at {checkpoint_file_load}, starting from scratch.")
    if local_rank == 0: logger.info(f"Saving checkpoint at {checkpoint_file}")


    # -------------------------------- Prepare DDP ------------------------------- #
    model = DDP(text2cad, device_ids=[local_rank], find_unused_parameters=True)
    processor: Text2CADProcessor = Text2CADProcessor.from_pretrained(pretrained_model_name_or_path=config["model"]["base_model"], padding_side="left")
    
    torch.cuda.synchronize()
    dist.barrier()

    # -------------------------------- Train Model ------------------------------- #
    train_model(
        model=model,
        processor=processor,
        dataloader=(train_loader, val_loader),
        checkpoint_file=checkpoint_file,
        config=config,
    )


def train_model(
    model: torch.nn.Module,
    processor: Text2CADProcessor,
    dataloader,
    checkpoint_file,
    config,
):
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    local_rank = global_rank % torch.cuda.device_count()
    device = torch.device(f'cuda:{local_rank}')

    train_loader, val_loader = dataloader

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.module.get_param_groups(
            base_lr = config["train"]["lr"],
            tau_lr = config["train"]["tau_lr"],
            weight_decay = config["train"]["weight_decay"],
        )
    )
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=config["train"]["num_iterations"] // world_size)
    reward_evaluator = Text2CADReward(
        valid=config["train"]["reward_weight"]["valid"], 
        chamfer_distance=config["train"]["reward_weight"]["chamfer_distance"], 
        length=config["train"]["reward_weight"]["length"],
        accuracy=config["train"]["reward_weight"]["accuracy"],
        accuracy_v=config["train"]["reward_weight"]["accuracy_weight"]["vertex"],
        accuracy_e=config["train"]["reward_weight"]["accuracy_weight"]["edge"],
        accuracy_f=config["train"]["reward_weight"]["accuracy_weight"]["face"],
    )

    if global_rank == 0:
        wandb.init(
            project="CADv3", 
            name=f"{os.path.basename(os.path.dirname(os.path.abspath(__file__)))}",
            tags=["RL"],
            config=config,
        )
        wandb.log({
            "base_model": config["model"]["base_model"].split("/")[-1],
            "num_iterations": config["train"]["num_iterations"] // world_size,
        })

    # val_loader.sampler.set_epoch(0)
    # validation_one_epoch(
    #     val_loader=val_loader,
    #     model=model,
    #     processor=processor,
    #     iteration=0,
    #     total_batch=config["val"]["val_batch"],
    # )

    # ---------------------------------- RL FINETUNING (GRPO) ---------------------------------- #
    dist.barrier()
    if local_rank == 0:
        logger.info(f"Starting RL finetuning using GRPO...")

    with get_progress_bar(total=(config["train"]["num_iterations"] // world_size) * config["train"]["batch_size"] * config["train"]["num_generations"] // config["train"]["grad_accum_steps"],
                          ascii=True, desc=f"\033[94mPointerCAD\033[0m✨", dynamic_ncols=True) as pbar:
        reference_model = None
        train_dataset_length = len(train_loader) - (len(train_loader) % config["train"]["batch_size"])
        train_dataset_iter = None
        for iteration in range(config["train"]["num_iterations"] // world_size):
            if iteration % train_dataset_length == 0:
                train_loader.sampler.set_epoch(iteration // train_dataset_length)
                train_dataset_iter = iter(train_loader)

            if iteration % config["train"]["reference_update_interval"] == 0:
                # Create reference model for KL constraint
                reference_model: torch.nn.Module = copy.deepcopy(model.module)
                reference_model.eval()
                reference_model = reference_model.to(device)
                for param in reference_model.parameters():
                    param.requires_grad = False

            train_iteration(
                model=model,
                reference_model=reference_model,
                processor=processor,
                optimizer=optimizer,
                scheduler=scheduler,
                reward_evaluator=reward_evaluator,
                train_batch=[next(train_dataset_iter) for _ in range(config["train"]["batch_size"])],
                pbar=pbar,
                iteration=iteration,
                config=config,
            )

            # ---------------- Save the model weights and optimizer state ---------------- #
            if global_rank == 0 and (((iteration + 1) % config["train"]["checkpoint_interval"] == 0) or (iteration + 1 == (config["train"]["num_iterations"] // world_size))):
                torch.save(
                    {
                        "iteration": iteration,
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "wandb": wandb.run.id if wandb.run is not None else None,
                    },
                    checkpoint_file,
                )

            # # ---------------- Perform Validation ---------------- #
            # val_loader.sampler.set_epoch(iteration + 1)
            # if iteration % config["val"]["interval"] == 0:
            #     validation_one_epoch(
            #         val_loader=val_loader,
            #         model=model,
            #         processor=processor,
            #         iteration=iteration,
            #         total_batch=config["val"]["val_batch"],
            #     )

    cleanup()

    # Close the wandb summary writer
    wandb.log({"iteration": config["train"]["num_iterations"] // world_size})
    wandb.finish()
    if global_rank == 0: logger.info("Training Finished.")


def train_iteration(
    model,
    reference_model,
    processor,
    optimizer,
    scheduler,
    reward_evaluator,
    train_batch,
    pbar,
    iteration,
    config,
):
    global_rank = dist.get_rank()
    local_rank = global_rank % torch.cuda.device_count()
    device = torch.device(f'cuda:{local_rank}')

    optimizer.zero_grad()

    for idx, item in enumerate(train_batch):
        with torch.no_grad():
            set_tqdm_desc(pbar, "Generating [A]")
            # argmax sample
            messages = [[
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
                        {"type": "text", "text": item["prompt"][0]},
                    ],
                },
            ]]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            breps = item["graph"].clone()
            inputs = processor(text=text, breps=breps, max_length=3072).to(device)
            pred_ids, pred_parameter_map, pred_label, pred_parameter, pred_pointer = model.module.predict(mode="argmax", tokenizer=processor.tokenizer, **inputs)

            gt_response_length = len(processor.tokenizer.tokenize(item["plan"][0])) + item["label"][0].shape[0]
            gt_json = item["json"][0].copy()
            generated_ids = [pred_ids.squeeze(0)]
            generated_parameter_map = pred_parameter_map
            generated_label = pred_label
            generated_parameter = pred_parameter
            generated_pointer = pred_pointer

            set_tqdm_desc(pbar, "Generating [S]")
            # sample for another num_generations-1
            messages, breps = [], []
            for _ in range(config["train"]["num_generations"] - 1):
                messages.append([
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
                            {"type": "text", "text": item["prompt"][0]},
                        ],
                    },
                ])
                breps.append(dgl.unbatch(item["graph"])[0].clone())
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, breps=dgl.batch(breps), max_length=3072).to(device)
            pred_ids, pred_parameter_map, pred_label, pred_parameter, pred_pointer = model.module.predict(mode="sample", tokenizer=processor.tokenizer,
                                                                                                          temperature_lm = config["train"]["temperature"]["plan"],
                                                                                                          temperature_label = config["train"]["temperature"]["label"],
                                                                                                          temperature_parameter = config["train"]["temperature"]["parameter"],
                                                                                                          temperature_pointer = config["train"]["temperature"]["pointer"], **inputs)

            generated_ids.extend([t for t in pred_ids])
            generated_parameter_map.extend(pred_parameter_map)
            generated_label.extend(pred_label)
            generated_parameter.extend(pred_parameter)
            generated_pointer.extend(pred_pointer)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        dist.barrier()

        set_tqdm_desc(pbar, "Calculating")
        assert len(generated_ids) == len(generated_parameter_map) == len(generated_label) == len(generated_parameter) == len(generated_pointer)

        with torch.no_grad():
            # Calculate rewards and advantages
            reward_dict = reward_evaluator(
                generated_ids,
                generated_parameter_map,
                generated_label,
                generated_parameter,
                generated_pointer,
                gt_response_length,
                gt_json
            )

            rewards = torch.tensor(reward_dict["rewards"], dtype=torch.float32, device=device)
            mean = rewards.mean()
            std = rewards.std()
            advantages = (rewards - mean) / (std + 1e-4)

            valid_mean = sum(reward_dict["valid"]) / len(reward_dict["valid"])
            cd_mean = sum(reward_dict["chamfer_distance"]) / len(reward_dict["chamfer_distance"])
            acc_mean = sum(reward_dict["accuracy"]) / len(reward_dict["accuracy"])
            length_mean = sum(reward_dict["length"]) / len(reward_dict["length"])
            valid_metric_mean = compute_means_without_invalid_data(reward_dict["valid_metric"])
            cd_metric_mean = compute_means_without_invalid_data(reward_dict["chamfer_distance_metric"])
            acc_v_metric_mean = compute_means_without_invalid_data(reward_dict["accuracy_metric"]["vertex"])
            acc_e_metric_mean = compute_means_without_invalid_data(reward_dict["accuracy_metric"]["edge"])
            acc_f_metric_mean = compute_means_without_invalid_data(reward_dict["accuracy_metric"]["face"])
            length_metric_mean = compute_means_without_invalid_data(reward_dict["length_metric"])
            rewards_mean_argmax = reward_dict["rewards"][0]
            valid_metric_argmax = reward_dict["valid_metric"][0]
            cd_metric_argmax = reward_dict["chamfer_distance_metric"][0]
            acc_v_metric_argmax = reward_dict["accuracy_metric"]["vertex"][0]
            acc_e_metric_argmax = reward_dict["accuracy_metric"]["edge"][0]
            acc_f_metric_argmax = reward_dict["accuracy_metric"]["face"][0]
            length_metric_argmax = reward_dict["length_metric"][0]

            # all reduce
            all_reduce_rewards_mean = all_reduce(mean)
            all_reduce_rewards_std = all_reduce(std)
            all_reduce_valid_mean = all_reduce(valid_mean)
            all_reduce_cd_mean = all_reduce(cd_mean)
            all_reduce_acc_mean = all_reduce(acc_mean)
            all_reduce_length_mean = all_reduce(length_mean)
            all_reduce_valid_metric_mean = all_reduce(valid_metric_mean)
            all_reduce_cd_metric_mean = all_reduce(cd_metric_mean)
            all_reduce_acc_v_metric_mean = all_reduce(acc_v_metric_mean)
            all_reduce_acc_e_metric_mean = all_reduce(acc_e_metric_mean)
            all_reduce_acc_f_metric_mean = all_reduce(acc_f_metric_mean)
            all_reduce_length_metric_mean = all_reduce(length_metric_mean)
            all_reduce_rewards_mean_argmax = all_reduce(rewards_mean_argmax)
            all_reduce_valid_metric_argmax = all_reduce(valid_metric_argmax)
            all_reduce_cd_metric_argmax = all_reduce(cd_metric_argmax)
            all_reduce_acc_v_metric_argmax = all_reduce(acc_v_metric_argmax)
            all_reduce_acc_e_metric_argmax = all_reduce(acc_e_metric_argmax)
            all_reduce_acc_f_metric_argmax = all_reduce(acc_f_metric_argmax)
            all_reduce_length_metric_argmax = all_reduce(length_metric_argmax)

        set_tqdm_desc(pbar, "Optimizing")
        for s in [slice(i * config["train"]["grad_accum_steps"], (i + 1) * config["train"]["grad_accum_steps"]) for i in range(config["train"]["num_generations"] // config["train"]["grad_accum_steps"])]:
            loss, loss_surrogate, loss_kl = train_step(
                model=model, reference_model=reference_model, processor=processor, prompt=item["prompt"][0], brep=dgl.unbatch(item["graph"])[0],
                generated_plan=processor.tokenizer.batch_decode(generated_ids[s], skip_special_tokens=True), generated_parameter_map=generated_parameter_map[s], generated_label=generated_label[s],
                generated_parameter=generated_parameter[s], generated_pointer=generated_pointer[s], advantages=advantages[s], config=config
            )

            # Update the progress bar
            updated_dict = {
                "loss": f"{loss:.4f}",
                "rewards": f"{all_reduce_rewards_mean:.2f}/{all_reduce_rewards_mean_argmax:.2f}",
                "valid": f"{all_reduce_valid_metric_mean:.2f}/{all_reduce_valid_metric_argmax:.2f}",
                "cd": f"{all_reduce_cd_metric_mean:.2f}/{all_reduce_cd_metric_argmax:.2f}",
                "acc": f"{all_reduce_acc_v_metric_mean:.1%}/{all_reduce_acc_v_metric_argmax:.1%}," +
                       f"{all_reduce_acc_e_metric_mean:.1%}/{all_reduce_acc_e_metric_argmax:.1%}," +
                       f"{all_reduce_acc_f_metric_mean:.1%}/{all_reduce_acc_f_metric_argmax:.1%}",
                "length": f"{all_reduce_length_metric_mean:.2f}/{all_reduce_length_metric_argmax:.2f}",
            }
            pbar.set_postfix(updated_dict)
            pbar.update(1)

        # ---------------------------- Add to WandB ---------------------------- #
        if global_rank == 0 and wandb.run is not None:
            wandb.log({
                "iteration": iteration + idx / config["train"]["batch_size"],
                "train/loss": loss,
                "train/loss/surrogate": loss_surrogate,
                "train/loss/kl_divergence": loss_kl,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/rewards/mean": all_reduce_rewards_mean,
                "train/rewards/std": all_reduce_rewards_std,
                "train/rewards/valid": all_reduce_valid_mean,
                "train/rewards/chamfer_distance": all_reduce_cd_mean,
                "train/rewards/accuracy": all_reduce_acc_mean,
                "train/rewards/length": all_reduce_length_mean,
                "train/metrics/valid": all_reduce_valid_metric_mean,
                "train/metrics/chamfer_distance": all_reduce_cd_metric_mean,
                "train/metrics/accuracy/vertex": all_reduce_acc_v_metric_mean,
                "train/metrics/accuracy/edge": all_reduce_acc_e_metric_mean,
                "train/metrics/accuracy/face": all_reduce_acc_f_metric_mean,
                "train/metrics/length": all_reduce_length_metric_mean,
                "train/argmax/rewards": all_reduce_rewards_mean_argmax,
                "train/argmax/metrics/valid": all_reduce_valid_metric_argmax,
                "train/argmax/metrics/chamfer_distance": all_reduce_cd_metric_argmax,
                "train/argmax/metrics/accuracy/vertex": all_reduce_acc_v_metric_argmax,
                "train/argmax/metrics/accuracy/edge": all_reduce_acc_e_metric_argmax,
                "train/argmax/metrics/accuracy/face": all_reduce_acc_f_metric_argmax,
                "train/argmax/metrics/length": all_reduce_length_metric_argmax,
            })

    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.9, norm_type=2.0)
    optimizer.step()
    optimizer.zero_grad()
    model.module.clamp_parameters()
    scheduler.step()


def train_step(
    model,
    reference_model,
    processor,
    prompt,
    brep,
    generated_plan,
    generated_parameter_map,
    generated_label,
    generated_parameter,
    generated_pointer,
    advantages,
    config,
):
    global_rank = dist.get_rank()
    local_rank = global_rank % torch.cuda.device_count()
    device = torch.device(f'cuda:{local_rank}')

    breps = []
    messages = []
    for plan in generated_plan:
        breps.append(brep.clone())
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

    with torch.no_grad():
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        generated_inputs = processor(text=text, breps=dgl.batch(breps), parameter_maps=generated_parameter_map, labels=generated_label, parameters=generated_parameter, pointers=generated_pointer, max_length=3072).to(device)
        ref_log_probs = compute_log_probs_and_aux(reference_model, generated_inputs, temperature_plan=config["train"]["temperature"]["plan"], temperature_label=config["train"]["temperature"]["label"], 
                                                  temperature_parameter=config["train"]["temperature"]["parameter"], temperature_pointer=config["train"]["temperature"]["pointer"])

    log_probs, dummy_loss = compute_log_probs_and_aux(model, generated_inputs, temperature_plan=config["train"]["temperature"]["plan"], temperature_label=config["train"]["temperature"]["label"],
                                                      temperature_parameter=config["train"]["temperature"]["parameter"], temperature_pointer=config["train"]["temperature"]["pointer"], dummy_loss=True)

    loss, loss_surrogate_report, loss_kl_report, loss_num = 0, 0, 0, 0
    for clogp, rlogp, adv in zip(log_probs, ref_log_probs, advantages):
        # Compute policy ratio
        ratio = torch.exp(clogp - clogp.detach())

        # Compute KL divergence penalty
        delta = (rlogp.detach() - clogp)  # clamp(-5.0, 5.0)
        kl_div = torch.exp(delta) - delta - 1

        surrogate_loss = ratio * adv
        loss_s = (surrogate_loss - config["train"]["beta"] * kl_div).mean()
        if torch.isfinite(loss_s):
            loss -= loss_s
            loss_surrogate_report -= surrogate_loss.detach().mean().item()
            loss_kl_report += kl_div.detach().mean().item()
            loss_num += 1

    if loss_num > 0:
        loss /= loss_num
        loss_surrogate_report /= loss_num
        loss_kl_report /= loss_num
    if not torch.is_tensor(loss):
        loss = dummy_loss.sum() * 0.0  # dummy loss

    loss = loss / (config["train"]["batch_size"] * config["train"]["num_generations"] // config["train"]["grad_accum_steps"])
    loss.backward()

    return loss.item(), loss_surrogate_report, loss_kl_report


@torch.no_grad()
def validation_one_epoch(
    val_loader,
    model,
    processor,
    iteration=0,
    topk=1,
    total_batch=5,
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

    value_accuracy_calc, pointer_accuracy_calc, scale_accuracy_calc = LabelAccuracyCalculator(), PointerAccuracyCalculator(), ScaleAccuracyCalculator()
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
                    "iteration": iteration,
                    "validation/value_accuracy": value_accuracy_t,
                    "validation/value_accuracy/special_token": value_accuracy_k,
                    "validation/value_accuracy/value_token": value_accuracy_v,
                    "validation/pointer_accuracy": pointer_accuracy,
                    "validation/scale_accuracy": scale_accuracy,
                })

            gc.collect()
            torch.cuda.empty_cache()

            return value_accuracy_t, value_accuracy_k, value_accuracy_v, pointer_accuracy, scale_accuracy


if __name__ == "__main__":
    num_procs = torch.cuda.device_count()
    world_size = int(os.getenv("WORLD_SIZE", num_procs))
    master_addr = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "32501")
    rank_offset = int(os.getenv("GLOBAL_RANK_OFFSET", 0))
    
    mp.spawn(main, args=(rank_offset, world_size, master_addr, master_port), nprocs=num_procs)