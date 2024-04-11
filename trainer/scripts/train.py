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

logger = get_logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_dataloaders(cfg: DictConfig) -> Any:
    dataloaders = {}
    datasets = {}
    for split in [cfg.train_split_name, cfg.valid_split_name]:
        cfg.gold_set = split == cfg.valid_split_name
        dataset = instantiate_with_cfg(cfg, split=split)
        should_shuffle = split == cfg.train_split_name
        datasets[split] = dataset
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            shuffle=should_shuffle,
            batch_size=cfg.batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=cfg.num_workers
        )
    # cfg.use_vicuna = True
    # cfg.gold_set = True
    # dataset = instantiate_with_cfg(cfg, split=cfg.valid_split_name)
    # datasets['vicuna'] = dataset
    # dataloaders['vicuna'] = torch.utils.data.DataLoader(
    #     dataset,
    #     shuffle=False, 
    #     batch_size=cfg.batch_size,
    #     collate_fn=dataset.collate_fn,
    #     num_workers=cfg.num_workers
    # )
    return dataloaders, datasets


def load_optimizer(cfg: DictConfig, model: nn.Module):
    optimizer = instantiate(cfg, model=model)
    return optimizer


def load_lr_scheduler(cfg: DictConfig, optimizer):
    scheduler = instantiate_with_cfg(cfg, optimizer=optimizer)
    return scheduler


def load_task(cfg: DictConfig, accelerator: BaseAccelerator):
    task = instantiate_with_cfg(cfg, accelerator=accelerator)
    return task

def process_config(cfg: TrainerConfig):
    # for values that depend on each other; ideally should be factored into Task
    if cfg.dataset.prepend_goal_to_steps:
        cfg.dataset.text_sequence_length = cfg.dataset.sequence_length
    else:
        cfg.dataset.text_sequence_length = cfg.dataset.sequence_length + 1
    cfg.model.text_sequence_length = cfg.dataset.text_sequence_length
    # cfg.dataset.text_sequence_length = cfg.dataset.sequence_length + 1
    return cfg
    

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
    accelerator = instantiate_with_cfg(cfg.accelerator)

    # if cfg.debug.activate and accelerator.is_main_process:
    #     import pydevd_pycharm
    #     pydevd_pycharm.settrace('localhost', port=cfg.debug.port, stdoutToServer=True, stderrToServer=True)

    if accelerator.is_main_process:
        # cfg = process_config(cfg)
        verify_or_write_config(cfg)

    logger.info(f"Loading task")
    task = load_task(cfg.task, accelerator)
    # logger.info(f"Loading RNG generator")
    logger.info(f"Loading model")
    model = instantiate_with_cfg(cfg.model)
    for n, p in model.named_parameters():
        p.data = p.data.contiguous()
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
    split2dataloader, datasets = load_dataloaders(cfg.dataset)

    dataloaders = list(split2dataloader.values())
    
    model, optimizer, lr_scheduler, *dataloaders = accelerator.prepare(model, optimizer, lr_scheduler, *dataloaders)
    model.generator = torch.Generator(accelerator.device).manual_seed(accelerator.cfg.seed) # TODO see if there's a cleaner way to do this... device issues from accelerate if not here
    
    split2dataloader = dict(zip(split2dataloader.keys(), dataloaders))

    accelerator.load_state_if_needed()

    accelerator.recalc_train_length_after_prepare(len(split2dataloader[cfg.dataset.train_split_name]))

    accelerator.init_training(cfg)

    def evaluate():
        model.eval()
        end_of_train_dataloader = accelerator.gradient_state.end_of_dataloader
        logger.info(f"*** Evaluating {cfg.dataset.valid_split_name} ***")
        metrics = task.evaluate(model, datasets[cfg.dataset.valid_split_name], split2dataloader[cfg.dataset.valid_split_name])
        # logger.info(f"*** Evaluating Vicuna {cfg.dataset.valid_split_name} ***")
        # metrics2 = task.evaluate(model, datasets['vicuna'], split2dataloader['vicuna'])
        # accelerator.update_metrics(metrics)
        # accelerator.gradient_state.end_of_dataloader = end_of_train_dataloader

    logger.info(f"task: {task.__class__.__name__}")
    logger.info(f"model: {model.__class__.__name__}")
    logger.info(f"num. model params: {int(sum(p.numel() for p in model.parameters()) // 1e6)}M")
    logger.info(
        f"num. model trainable params: {int(sum(p.numel() for p in model.parameters() if p.requires_grad) // 1e6)}M")
    # logger.info(f"criterion: {criterion.__class__.__name__}")
    logger.info(f"num. train examples: {len(split2dataloader[cfg.dataset.train_split_name].dataset)}")
    logger.info(f"num. valid examples: {len(split2dataloader[cfg.dataset.valid_split_name].dataset)}")
    # logger.info(f"num. test examples: {len(split2dataloader[cfg.dataset.test_split_name].dataset)}")

    for epoch in range(accelerator.cfg.num_epochs):
        train_loss, lr = 0.0, 0.0
        for step, batch in enumerate(split2dataloader[cfg.dataset.train_split_name]):
            if accelerator.should_skip(epoch, step):
                accelerator.update_progbar_step()
                continue

            if accelerator.should_eval(): # TODO add eval back
                evaluate()

            if accelerator.should_save():
                accelerator.save_checkpoint()

            model.unet.train() # TODO confirm the intended params alone are trainable

            with accelerator.accumulate(model):
                loss = task.train_step(model, batch)
                avg_loss = accelerator.gather(loss).mean().item()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters())

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            train_loss += avg_loss / accelerator.cfg.gradient_accumulation_steps

            if accelerator.sync_gradients:
                accelerator.update_global_step(train_loss)
                train_loss = 0.0

            if accelerator.global_step > 0:
                lr = lr_scheduler.get_last_lr()[0]

            accelerator.update_step(avg_loss, lr)
            

            if accelerator.should_end():
                evaluate()
                accelerator.save_checkpoint()
                break

        if accelerator.should_end():
            break

        accelerator.update_epoch()

    accelerator.wait_for_everyone()
    accelerator.load_best_checkpoint()
    logger.info(f"*** Evaluating {cfg.dataset.valid_split_name} ***")
    metrics = task.evaluate(model, datasets[cfg.dataset.valid_split_name], split2dataloader[cfg.dataset.valid_split_name])
    accelerator.update_metrics(metrics)
    # logger.info(f"*** Evaluating {cfg.dataset.test_split_name} ***")
    # metrics = task.evaluate(model, split2dataloader[cfg.dataset.test_split_name])
    # metrics = {f"{cfg.dataset.test_split_name}_{k}": v for k, v in metrics.items()}
    # accelerator.update_metrics(metrics)
    accelerator.unwrap_and_save(model)
    accelerator.end_training()


if __name__ == '__main__':
    main()
