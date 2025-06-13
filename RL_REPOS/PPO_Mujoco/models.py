import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from tensordict.nn.distributions import NormalParamExtractor

def make_mlp(output_dim, hidden_dim, activation_fn, n_layers=3, device="cpu"):
    layers = []
    for _ in range(n_layers):
        layers += [nn.LazyLinear(hidden_dim, device=device), activation_fn]
    layers += [nn.LazyLinear(output_dim, device=device)]
    return nn.Sequential(*layers)

def build_policy_module(cfg, env, device):
    actor_net = make_mlp(
        output_dim=2 * env.action_spec.shape[-1],
        hidden_dim=cfg.model.num_cells,
        activation_fn={"tanh": nn.Tanh(), "relu": nn.ReLU()}[cfg.model.hidden_activation],
        n_layers=cfg.model.actor.n_layers,
        device=device,
    )
    actor_net.append(NormalParamExtractor())

    td_module = TensorDictModule(
        actor_net,
        in_keys=[str(k) for k in cfg.model.actor.in_keys],
        out_keys=[str(k) for k in cfg.model.actor.out_keys],
    )

    return ProbabilisticActor(
        module=td_module,
        spec=env.action_spec,
        in_keys=[str(k) for k in cfg.model.actor.out_keys],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
        return_log_prob=True,
    )

def build_value_module(cfg, env, device):
    value_net = make_mlp(
        output_dim=1,
        hidden_dim=cfg.model.num_cells,
        activation_fn={"tanh": nn.Tanh(), "relu": nn.ReLU()}[cfg.model.hidden_activation],
        n_layers=cfg.model.critic.n_layers,
        device=device,
    )
    return ValueOperator(
        module=value_net,
        in_keys=[str(k) for k in cfg.model.critic.in_keys],
    )
