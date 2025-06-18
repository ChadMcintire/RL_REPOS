from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)

def make_env(cfg, device):
    base_env = GymEnv(cfg.env.gym_env, device=device, render_mode=cfg.env.render_mode)
    env = TransformedEnv(
        base_env,
        Compose(
            ObservationNorm(in_keys=[str(k) for k in cfg.model.actor.in_keys]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    return env
