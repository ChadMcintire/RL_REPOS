import math
from torch.optim.lr_scheduler import StepLR, LambdaLR, CosineAnnealingLR, OneCycleLR

def make_lr_scheduler(optimizer, sched_cfg, config):
    print("max total steps abababab", config.max_total_steps)
    kind = sched_cfg.type.lower()
    if kind == "step":
        return StepLR(optimizer,
                      step_size=sched_cfg.step_size,
                      gamma=sched_cfg.gamma)
    elif kind == "lambda":
        return LambdaLR(optimizer,
                        lr_lambda=lambda step: sched_cfg.lambda_factor ** step)
    elif kind == "cosine":
        return CosineAnnealingLR(optimizer,
                                 T_max=sched_cfg.step_size,
                                 eta_min=getattr(sched_cfg, "eta_min", 0.0))
    elif kind == "onecycle":
        return OneCycleLR(optimizer, total_steps=config.max_total_steps, max_lr=.01, three_phase=True)
    else:
        raise ValueError(f"Unknown scheduler type `{sched_cfg.type}`")
