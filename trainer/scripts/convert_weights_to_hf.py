import json
import os
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from torch import nn

from trainer.accelerators.base_accelerator import BaseAccelerator
from trainer.configs.configs import TrainerConfig, instantiate_with_cfg

from hydra import compose, initialize

logger = get_logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_dataloaders(cfg: DictConfig) -> Any:
    dataloaders = {}
    for split in [cfg.train_split_name, cfg.valid_split_name]:
        dataset = instantiate_with_cfg(cfg, split=split)
        should_shuffle = split == cfg.train_split_name
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            shuffle=should_shuffle,
            batch_size=cfg.batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=cfg.num_workers
        )
    return dataloaders


def load_optimizer(cfg: DictConfig, model: nn.Module):
    optimizer = instantiate(cfg, model=model)
    return optimizer


def load_lr_scheduler(cfg: DictConfig, optimizer):
    scheduler = instantiate_with_cfg(cfg, optimizer=optimizer)
    return scheduler


def load_task(cfg: DictConfig, accelerator: BaseAccelerator):
    task = instantiate_with_cfg(cfg, accelerator=accelerator)
    return task


def verify_or_write_config(cfg: TrainerConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    yaml_path = os.path.join(cfg.output_dir, "config.yaml")
    if not os.path.exists(yaml_path):
        OmegaConf.save(cfg, yaml_path, resolve=True)
    with open(yaml_path) as f:
        existing_config = f.read()
    if existing_config != OmegaConf.to_yaml(cfg, resolve=True):
        # TODO: replace with something that tells you which keys changed
        # load dict from yaml
        existing_config = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
        # find different keys and values between existing config and cfg
        diff = {k: v for k, v in cfg.items() if existing_config.get(k) != v}
        # warn instead of raise
        import warnings
        warnings.warn(f"Saved config at {yaml_path} does not match given config - {diff}")
        if cfg.config_overwrite:
            logger.info(f"Overwriting config at {yaml_path}")
            OmegaConf.save(cfg, yaml_path, resolve=True)
        else:
            raise ValueError(f"Config was not saved correctly - {yaml_path}")
            
    logger.info(f"Config can be found in {yaml_path}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: TrainerConfig) -> None:
    # print(cfg.accelerator)
    for k, v in cfg.accelerator.items():
        # print(k, type(v))
        if k in ('mixed_precision', 'log_with', 'dynamo_backend'):
            cfg.accelerator[k] = v.lower()
    # config_path = cfg.output_dir
    # relative_config_path = os.path.relpath(config_path, start=os.getcwd())

    # with initialize(config_path=relative_config_path):
    #     loaded_cfg = compose(config_name="config.yaml")
    # print(loaded_cfg)
    
    accelerator = instantiate_with_cfg(cfg.accelerator)
    # ckpt_path = './outputs/1seq_rand_real/checkpoint-gstep25300'
    # cfg.output_dir = ckpt_path # TODO make this cmd arg # TODO load cfg for ckpt instead of changing manually 
    # TODO remove need to fill in seq length in dataset file!!!

    # if cfg.debug.activate and accelerator.is_main_process:
    #     import pydevd_pycharm
    #     pydevd_pycharm.settrace('localhost', port=cfg.debug.port, stdoutToServer=True, stderrToServer=True)

    # if accelerator.is_main_process:
    #     verify_or_write_config(cfg)

    logger.info(f"Loading task")
    task = load_task(cfg.task, accelerator)
    # logger.info(f"Loading RNG generator")
    logger.info(f"Loading model")
    model = instantiate_with_cfg(cfg.model)
    # logger.info(f"Loading criterion")
    # criterion = instantiate_with_cfg(cfg.criterion)
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format


    # if args.gradient_checkpointing:
    #     unet.enable_gradient_checkpointing()
    
    
    
    logger.info(f"Loading optimizer")
    optimizer = load_optimizer(cfg.optimizer, model)
    logger.info(f"Loading lr scheduler")
    lr_scheduler = load_lr_scheduler(cfg.lr_scheduler, optimizer)
    logger.info(f"Loading dataloaders")
    split2dataloader = load_dataloaders(cfg.dataset)

    dataloaders = list(split2dataloader.values())
    model, optimizer, lr_scheduler, *dataloaders = accelerator.prepare(model, optimizer, lr_scheduler, *dataloaders)
    model.generator = torch.Generator(accelerator.device).manual_seed(accelerator.cfg.seed) # TODO see if there's a cleaner way to do this... device issues from accelerate if not here
    
    split2dataloader = dict(zip(split2dataloader.keys(), dataloaders))
    
    try:
        accelerator.accelerator.load_state(cfg.output_dir, strict=True)
    except RuntimeError as e:
        # print the error and try again with strict=False
        logger.warning(f"Error loading checkpoint with strict=True: {e}")
        logger.warning("Trying again with strict=False")
        accelerator.accelerator.load_state(cfg.output_dir, strict=False) # TODO figure out how to load unet.pos correctly...
    
    # output_dir = './output/'
    unet_output_dir = os.path.join(cfg.output_dir, "unet")
    logger.info(f"Saving unet checkpoint to {unet_output_dir}")
    model.unet.save_pretrained(unet_output_dir)
    logger.info(f"Saved unet checkpoint to {unet_output_dir}")
    
    # ema_output_dir = os.path.join(cfg.output_dir, "unet_ema")
    # logger.info(f"Saving unet ema checkpoint to {ema_output_dir}")
    # model.ema_unet.save_pretrained(ema_output_dir)
    # logger.info(f"Saved unet ema checkpoint to {ema_output_dir}")
    
    

if __name__ == '__main__':
    main()
