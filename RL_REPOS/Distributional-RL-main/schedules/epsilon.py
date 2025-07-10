import math

def linear_eps(step: int,
               initial_eps: float,
               min_eps: float,
               decay_steps: int) -> float:
    frac = min(1.0, step / decay_steps)
    return initial_eps * (1 - frac) + min_eps * frac

def exp_eps(step: int,
            initial_eps: float,
            min_eps: float,
            decay_rate: float) -> float:
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
                                       cfg.agent.initial_eps,
                                       cfg.agent.min_exp_eps,
                                       cfg.agent.eps_decay_steps)
    elif cfg.agent.eps_type == "exponential":
        return lambda step: exp_eps(step,
                                    cfg.agent.initial_eps,
                                    cfg.agent.min_exp_eps,
                                    cfg.agent.decay_rate)
    else:
        raise ValueError(f"Unknown eps_type {cfg.agent.eps_type}")
