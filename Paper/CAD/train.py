import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))

import random
import numpy as np
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.logger import CLGLogger
from Cad_VLM.models.text2cad import Text2CAD
from Cad_VLM.models.loss import CELoss
from Cad_VLM.models.metrics import AccuracyCalculator
from Cad_VLM.models.utils import print_with_separator
from Cad_VLM.dataprep.t2c_dataset import get_dataloaders
from loguru import logger
import torch
import argparse
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from torch.optim.lr_scheduler import ExponentialLR
import datetime
import gc
import argparse
import yaml
import warnings
import logging.config
from transformers import BertTokenizer, BertModel
from pathlib import Path

warnings.filterwarnings("ignore")
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)

t2clogger = CLGLogger().configure_logger(verbose=True).logger


# ---------------------------------------------------------------------------- #
#                            Text2CAD Training Code                            #
# ---------------------------------------------------------------------------- #

def load_tokenizer(model_name, cache_dir=None):
    """加载分词器，增加错误处理和日志记录"""
    try:
        # 确保缓存目录存在
        if cache_dir:
            cache_path = Path(cache_dir)
            if not cache_path.exists():
                cache_path.mkdir(parents=True, exist_ok=True)
                t2clogger.info(f"Created cache directory: {cache_path}")

            # 检查必需文件是否存在
            required_files = ["vocab.txt", "tokenizer.json", "tokenizer_config.json"]
            missing_files = [f for f in required_files if not (cache_path / f).exists()]

            if missing_files:
                t2clogger.warning(f"Missing files in cache: {missing_files}")
                # 尝试自动下载缺失文件
                try:
                    from huggingface_hub import hf_hub_download
                    for file in missing_files:
                        hf_hub_download(
                            repo_id=model_name,
                            filename=file,
                            cache_dir=cache_dir
                        )
                    t2clogger.info("Missing files downloaded successfully")
                except Exception as e:
                    t2clogger.error(f"Failed to download missing files: {e}")
                    cache_dir = None

        # 尝试从缓存目录加载
        if cache_dir:
            t2clogger.info(f"Loading tokenizer from cache: {cache_dir}")
            tokenizer = BertTokenizer.from_pretrained(cache_dir)
            return tokenizer
    except Exception as e:
        t2clogger.error(f"Error loading tokenizer from cache: {e}")

    # 回退到默认加载方式
    t2clogger.info(f"Loading tokenizer from model name: {model_name}")
    return BertTokenizer.from_pretrained(model_name)


def parse_config_file(config_file):
    """解析配置文件，增加错误处理"""
    try:
        with open(config_file, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        t2clogger.error(f"Error parsing config file: {e}")
        raise


def save_yaml_file(yaml_data, filename, output_dir):
    """保存YAML文件"""
    try:
        with open(os.path.join(output_dir, filename), "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
    except Exception as e:
        t2clogger.error(f"Error saving YAML file: {e}")


@logger.catch()
def main():
    print_with_separator("😊 Text2CAD Training 😊")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="config/trainer.yaml",
    )
    args = parser.parse_args()

    try:
        config = parse_config_file(args.config_path)
    except Exception as e:
        t2clogger.critical(f"Failed to load config file: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t2clogger.info(
        "Current Device {}",
        torch.cuda.get_device_properties(device) if device == "cuda" else device,
    )

    # -------------------------------- Load Model -------------------------------- #
    cad_config = config["cad_decoder"]
    cad_config["cad_seq_len"] = MAX_CAD_SEQUENCE_LENGTH

    # 确保文本嵌入器配置存在
    if "text_encoder" not in config or "text_embedder" not in config["text_encoder"]:
        t2clogger.error("Missing text_encoder or text_embedder configuration")
        return

    text_embedder_config = config["text_encoder"]["text_embedder"]

    # 加载分词器
    tokenizer = load_tokenizer(
        model_name=text_embedder_config["model_name"],
        cache_dir=text_embedder_config.get("cache_dir")
    )

    # 创建模型
    try:
        text2cad = Text2CAD(
            text_config=config["text_encoder"],
            cad_config=cad_config,
            tokenizer=tokenizer  # 直接传入加载好的分词器
        ).to(device)
    except Exception as e:
        t2clogger.critical(f"Failed to create model: {e}")
        return

    # Freeze the base text embedder
    for param in text2cad.base_text_embedder.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(text2cad.parameters(), lr=config["train"]["lr"])
    scheduler = ExponentialLR(optimizer, gamma=0.999)
    criterion = CELoss(device=device)

    lr = config["train"]["lr"]
    dim = config["cad_decoder"]["cdim"]
    nlayers = config["cad_decoder"]["num_layers"]
    batch = config["train"]["batch_size"]
    ca_level_start = config["cad_decoder"]["ca_level_start"]

    # --------------------------- Prepare Log Directory -------------------------- #
    now = datetime.datetime.now()
    time_str = now.strftime("%H:%M")
    date_str = datetime.date.today()
    log_dir = os.path.join(
        config["train"]["log_dir"],
        f"{date_str}/{time_str}_d{dim}_nl{nlayers}_ca{ca_level_start}",
    )
    t2clogger.info(
        "Current Date {date_str} Time {time_str}\n",
        date_str=date_str,
        time_str=time_str,
    )

    # Create the log dir if it doesn't exist
    if not config.get("debug", False):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        try:
            save_yaml_file(
                config,
                filename=os.path.basename(args.config_path),
                output_dir=log_dir
            )
        except Exception as e:
            t2clogger.error(f"Failed to save config file: {e}")

    # -------------------------------- Train Model ------------------------------- #
    try:
        train_model(
            model=text2cad,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            log_dir=log_dir,
            num_epochs=config["train"]["num_epochs"],
            checkpoint_name=f"lr{lr}_d{dim}_nl{nlayers}_b{batch}_ca{ca_level_start}",
            config=config,
        )
    except Exception as e:
        t2clogger.critical(f"Training failed: {e}")
        raise


def train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        log_dir,
        num_epochs,
        checkpoint_name,
        config,
):
    """
    Trains a deep learning model.
    """
    # Create the dataloader for train
    try:
        train_loader, val_loader = get_dataloaders(
            cad_seq_dir=config["train_data"]["cad_seq_dir"],
            prompt_path=config["train_data"]["prompt_path"],
            split_filepath=config["train_data"]["split_filepath"],
            subsets=["train", "validation"],
            batch_size=config["train"]["batch_size"],
            num_workers=min(config["train"]["num_workers"], os.cpu_count()),
            pin_memory=True,
            shuffle=False,
            prefetch_factor=config["train"]["prefetch_factor"],
        )
    except Exception as e:
        t2clogger.critical(f"Failed to create data loaders: {e}")
        return

    tensorboard_dir = os.path.join(log_dir, f"summary")

    # ---------------------- Resume Training from checkpoint --------------------- #
    checkpoint_file = os.path.join(log_dir, f"t2c_{checkpoint_name}.pth")
    checkpoint_only_model_file = os.path.join(
        log_dir, f"t2c_{checkpoint_name}_model.pth"
    )

    if config["train"].get("checkpoint_path") is None:
        old_checkpoint_file = checkpoint_file
    else:
        old_checkpoint_file = config["train"]["checkpoint_path"]

    start_epoch = 1
    step = 0

    if os.path.exists(old_checkpoint_file):
        t2clogger.info("Using saved checkpoint at {}", old_checkpoint_file)
        try:
            checkpoint = torch.load(old_checkpoint_file, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            step = checkpoint.get("step", 0)
            t2clogger.info(f"Resuming from epoch {start_epoch}, step {step}")
        except Exception as e:
            t2clogger.error(f"Failed to load checkpoint: {e}")
            start_epoch = 1
            step = 0
    else:
        t2clogger.info("No checkpoint found, starting from scratch")

    t2clogger.info("Saving checkpoint at {}", checkpoint_file)

    # Create the tensorboard summary writer
    try:
        writer = SummaryWriter(log_dir=tensorboard_dir, comment=f"{checkpoint_name}")
    except Exception as e:
        t2clogger.error(f"Failed to create SummaryWriter: {e}")
        writer = None

    # 使用DataParallel进行并行处理
    if torch.cuda.device_count() > 1:
        t2clogger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    else:
        t2clogger.info("Using single GPU")

    # ---------------------------------- Training ---------------------------------- #
    if start_epoch > config["train"].get("curriculum_learning_epoch", 0):
        t2clogger.warning("MIXED LEARNING...")
        random.shuffle(train_loader.dataset.uid_pair)
    else:
        t2clogger.info("CURRICULUM LEARNING...")

    # Start training
    model.train()
    for epoch in range(start_epoch, num_epochs + 1):
        # ------------------------------- Single Epoch ------------------------------- #
        # Shuffle the data when curriculum learning stops
        if epoch == config["train"].get("curriculum_learning_epoch", 0):
            t2clogger.info("MIXED LEARNING...")
            optimizer = optim.AdamW(model.parameters(), lr=config["train"]["lr"])
            scheduler = ExponentialLR(optimizer, gamma=0.99)

        if epoch >= config["train"].get("curriculum_learning_epoch", 0):
            random.shuffle(train_loader.dataset.uid_pair)

        # Train for one epoch
        train_loss = []
        train_loss_seq = {"seq": []}
        train_accuracy_seq = {"seq": []}
        val_accuracy_seq = {"seq": []}

        try:
            with tqdm(
                    train_loader,
                    ascii=True,
                    desc=f"\033[94mText2CAD\033[0m: Epoch [{epoch}/{num_epochs + 1}]✨",
            ) as pbar:
                for _, vec_dict, prompt, mask_cad_dict in pbar:
                    step += 1

                    for key, value in vec_dict.items():
                        vec_dict[key] = value.to(device)

                    for key, value in mask_cad_dict.items():
                        mask_cad_dict[key] = value.to(device)

                    # Padding mask for predicted Cad Sequence
                    shifted_key_padding_mask = mask_cad_dict["key_padding_mask"][:, 1:]
                    # Create Label for Training
                    cad_vec_target = vec_dict["cad_vec"][:, 1:].clone()

                    # Create training input by removing the last token
                    for key, value in vec_dict.items():
                        vec_dict[key] = value[:, :-1]

                    # Padding mask for input Cad Sequence
                    mask_cad_dict["key_padding_mask"] = mask_cad_dict["key_padding_mask"][
                                                        :, :-1
                                                        ]

                    # Output from the model
                    cad_vec_pred, _ = model(
                        vec_dict=vec_dict,
                        texts=prompt,
                        mask_cad_dict=mask_cad_dict,
                        metadata=False,
                    )

                    # ----------------------------- Loss Calculation ----------------------------- #
                    loss, loss_sep_dict = criterion(
                        {
                            "pred": cad_vec_pred,
                            "target": cad_vec_target,
                            "key_padding_mask": ~shifted_key_padding_mask,
                        }
                    )

                    # ------------------------------- Backward pass ------------------------------ #
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        parameters=model.parameters(), max_norm=0.9, norm_type=2.0
                    )
                    optimizer.step()

                    # --------------------------- Log loss and accuracy -------------------------- #
                    train_loss.append(loss.item())
                    train_loss_seq["seq"].append(loss_sep_dict["loss_seq"])

                    # Compute accuracy
                    cad_accuracy = AccuracyCalculator(
                        discard_token=len(END_TOKEN)
                    ).calculateAccMulti2DFromProbability(cad_vec_pred, cad_vec_target)

                    # Add Accuracy report
                    train_accuracy_seq["seq"].append(cad_accuracy)
                    pbar_keys = ["Loss", "seq"]

                    updated_dict = {
                        key: (
                            np.round(train_loss[-1], decimals=2)
                            if key == "Loss"
                            else np.round(train_accuracy_seq[key.lower()][-1], decimals=2)
                        )
                        for key in pbar_keys
                    }
                    # Update the progress bar
                    pbar.set_postfix(updated_dict)

                    # ---------------------------- Add to Tensorboard ---------------------------- #
                    if writer and not config.get("debug", False):
                        # Add Losses
                        writer.add_scalar(
                            "Seq Loss (Train)",
                            np.mean(train_loss_seq["seq"]),
                            step,
                        )

                        # Add Accuracies
                        writer.add_scalar(
                            "Seq Accuracy (Train)",
                            np.mean(train_accuracy_seq["seq"]),
                            step,
                        )

                        writer.add_scalar(
                            "Total Train Loss", np.mean(train_loss), step
                        )
        except Exception as e:
            t2clogger.error(f"Error during training epoch {epoch}: {e}")
            continue

        # Perform Validation
        try:
            val_cad_acc = validation_one_epoch(
                val_loader=val_loader,
                model=model,
                epoch=epoch,
                num_epochs=num_epochs,
                writer=writer,
                config=config,
                total_batch=config["val"].get("val_batch", 5),
            )
            val_accuracy_seq["seq"].append(val_cad_acc)
        except Exception as e:
            t2clogger.error(f"Error during validation epoch {epoch}: {e}")
            val_accuracy_seq["seq"].append(0)

        # ---------------- Save the model weights and optimizer state ---------------- #
        if not config.get("debug", False):
            # Save checkpoints
            if epoch % config["train"].get("checkpoint_interval", 10) == 0:
                checkpoint_file = os.path.join(
                    log_dir, f"t2c_{checkpoint_name}_{epoch}.pth"
                )

                try:
                    # 获取模型状态字典（处理DataParallel情况）
                    model_state_dict = model.module.state_dict() if isinstance(model,
                                                                               torch.nn.DataParallel) else model.state_dict()

                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model_state_dict,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "step": step,
                        },
                        checkpoint_file,
                    )
                    t2clogger.info(f"Saved checkpoint to {checkpoint_file}")
                except Exception as e:
                    t2clogger.error(f"Failed to save checkpoint: {e}")

        try:
            scheduler.step()
        except Exception as e:
            t2clogger.error(f"Error updating scheduler: {e}")

        # Print epoch summary
        logger.info(
            f"Epoch [{epoch}/{num_epochs + 1}]✅,"
            f" Train Loss: {np.round(np.mean(train_loss), decimals=2)},"
            f" Train Seq Acc: {np.round(np.mean(train_accuracy_seq['seq']), decimals=2)},"
            f" Val Seq Acc: {np.round(np.mean(val_accuracy_seq['seq']), decimals=2)}",
        )

    # Close the tensorboard summary writer
    if writer:
        writer.close()
    t2clogger.success("Training Finished.")


def validation_one_epoch(
        val_loader,
        model,
        epoch=0,
        num_epochs=0,
        writer=None,
        topk=5,
        config=None,
        total_batch=5,
):
    """
    Perform one validation epoch.
    """
    seq_acc_all = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取模型（处理DataParallel情况）
    val_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    val_model = val_model.to(device)
    val_model.eval()

    cur_batch = 0
    with torch.no_grad():
        try:
            with tqdm(
                    val_loader, ascii=True, desc=f"Epoch [{epoch}/{num_epochs + 1}] Validation✨"
            ) as pbar:
                for _, vec_dict, prompt, mask_cad_dict in pbar:
                    if cur_batch == total_batch:
                        break
                    cur_batch += 1

                    for key, val in vec_dict.items():
                        vec_dict[key] = val.to(device)

                    for key, val in mask_cad_dict.items():
                        mask_cad_dict[key] = val.to(device)

                    # Create a copy of the sequence dictionaries, and take only the start token
                    sec_topk_acc = []
                    for topk_index in range(1, topk + 1):
                        new_cad_seq_dict = vec_dict.copy()

                        for key, value in new_cad_seq_dict.items():
                            new_cad_seq_dict[key] = value[:, :1]

                        # Autoregressive Prediction
                        pred_cad_seq_dict = val_model.test_decode(
                            texts=prompt,
                            maxlen=MAX_CAD_SEQUENCE_LENGTH,
                            nucleus_prob=0,
                            topk_index=topk_index,
                            device=device,
                        )
                        gc.collect()
                        torch.cuda.empty_cache()

                        # Calculate accuracies
                        try:
                            cad_seq_acc = AccuracyCalculator(
                                discard_token=len(END_TOKEN)
                            ).calculateAccMulti2DFromLabel(
                                pred_cad_seq_dict["cad_vec"].cpu(),
                                vec_dict["cad_vec"].cpu(),
                            )
                        except Exception as e:
                            t2clogger.error(f"Error calculating accuracy: {e}")
                            cad_seq_acc = 0

                        pbar.set_postfix({"Seq": np.round(cad_seq_acc, decimals=2)})
                        sec_topk_acc.append(cad_seq_acc)

                    seq_acc_all.append(np.max(sec_topk_acc))
                    sec_topk_acc = []

            # Calculate mean accuracies
            mean_seq_acc = np.mean(seq_acc_all) if seq_acc_all else 0
            gc.collect()
            torch.cuda.empty_cache()

            # Log to TensorBoard
            if writer:
                writer.add_scalar(
                    "Seq Accuracy (Val)",
                    np.round(mean_seq_acc, decimals=2),
                    epoch,
                )

            return mean_seq_acc
        except Exception as e:
            t2clogger.error(f"Error during validation: {e}")
            return 0


if __name__ == "__main__":
    main()