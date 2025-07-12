import math

def linear_eps(step: int,
               cfg,
               ) -> float:

    initial_eps = cfg.agent.initial_eps
    min_eps = cfg.agent.min_exp_eps
    decay_steps = cfg.agent.eps_decay_steps

    # compute how many raw frames before we start decaying
    warmup = cfg.init_mem_size_to_train * cfg.train_interval

    # only start counting *after* warmup frames
    adjusted = max(0, step - warmup)

    frac = min(1.0, step / decay_steps)
    return initial_eps * (1 - frac) + min_eps * frac

def exp_eps(step: int,
            cfg) -> float:

    initial_eps = cfg.agent.initial_eps
    min_eps = cfg.agent.min_exp_eps
    decay_rate = cfg.agent.decay_rate

    warmup = cfg.init_mem_size_to_train * cfg.train_interval
    adjusted = max(0, step - warmup)
    return min_eps + (initial_eps - min_eps) * math.exp(-decay_rate * step)


def make_epsilon_fn(cfg):
    """
    cfg should have:
      - cfg.agent.eps_type: "linear"|"exponential"
      - cfg.agent.initial_eps
      - cfg.agent.min_eps
      - cfg.agent.decay_steps
      - cfg.agent.decay_rate
    """
    if cfg.agent.eps_type == "linear":
        return lambda step: linear_eps(step,
                                       cfg)
    elif cfg.agent.eps_type == "exponential":
        return lambda step: exp_eps(step,
                                    cfg)
    else:
        raise ValueError(f"Unknown eps_type {cfg.agent.eps_type}")
