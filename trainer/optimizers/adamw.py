from dataclasses import dataclass


@dataclass
class AdamWOptimizerConfig:
    _target_: str = "torch.optim.adamw.AdamW" # TODO: allow replacement by bnb.optim.AdamW8bit
    lr: float = 1e-6
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.0
    eps: float = 1e-8
