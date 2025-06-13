import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing

import imageio
import numpy as np



from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from tensordict import TensorDict




import torch
from torchrl.envs.libs.gym import GymEnv
import matplotlib.pyplot as plt
import matplotlib
import hydra
from omegaconf import DictConfig

import os
os.environ["MUJOCO_GL"] = "egl"

from record import record_current_model

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    num_cells=num_cells = cfg.model.num_cells
    lr = cfg.training.lr
    max_grad_norm = cfg.training.max_grad_norm
    frames_per_batch = cfg.training.frames_per_batch 

    # For a complete training, bring the number of frames up to 1M
    total_frames = cfg.training.total_frames

    sub_batch_size = cfg.training.sub_batch_size # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = cfg.training.num_epochs # optimization steps per batch of data collected

    clip_epsilon = (
        cfg.algo.ppo.clip_epsilon  # clip value for PPO loss: see the equation in the intro for more context.
    )

    gamma = cfg.training.gamma
    lmbda = cfg.training.lmbda
    entropy_eps = cfg.algo.ppo.entropy_eps
    activation_fn = {"tanh": nn.Tanh(), "relu": nn.ReLU()}[cfg.model.hidden_activation]





    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    
    
    
    
    
    base_env = GymEnv(cfg.env.gym_env, device=device, render_mode=cfg.env.render_mode)
    
    
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys= [str(k) for k in cfg.model.actor.in_keys]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    
    
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    
    
    print("normalization constant shape:", env.transform[0].loc.shape)
    
    print("observation_spec:", env.observation_spec)
    print("reward_spec:", env.reward_spec)
    print("input_spec:", env.input_spec)
    print("action_spec (as defined by input_spec):", env.action_spec)
    
    check_env_specs(env)
    
    rollout = env.rollout(3)
    print("rollout of three steps:", rollout)
    print("Shape of the rollout TensorDict:", rollout.batch_size)

    
    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        activation_fn,
        nn.LazyLinear(num_cells, device=device),
        activation_fn,
        nn.LazyLinear(num_cells, device=device),
        activation_fn,
        nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
        NormalParamExtractor(),
    )
    
    
    policy_module = TensorDictModule(
        actor_net, 
        in_keys=[str(k) for k in cfg.model.actor.in_keys], 
        out_keys=[str(k) for k in cfg.model.actor.out_keys]
    )
    
    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=[str(k) for k in cfg.model.actor.out_keys], #in keys need to match the actor_net out keys
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )
    
    
    value_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        activation_fn,
        nn.LazyLinear(num_cells, device=device),
        activation_fn,
        nn.LazyLinear(num_cells, device=device),
        activation_fn,
        nn.LazyLinear(1, device=device),
    )
    
    value_module = ValueOperator(
        module=value_net,
        in_keys=[str(k) for k in cfg.model.critic.in_keys],
    )
    
    
    print("Running policy:", policy_module(env.reset()))
    print("Running value:", value_module(env.reset()))
    
    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=cfg.algo.ppo.split_trajs,
        device=device,
    )
    
    
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )
    
    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, 
        value_network=value_module, 
        average_gae=cfg.algo.ppo.average_gae, 
        device=device,
    )
    
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        # these keys match by default but we set this for completeness
        critic_coef=cfg.algo.ppo.critic_coef,
        loss_critic_type=cfg.algo.ppo.loss_critic_type,
    )
    
    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )
    
    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""
    
    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in enumerate(collector):
        # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(num_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
    
                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()
    
        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our ``env`` horizon).
            # The ``rollout`` method of the ``env`` can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(cfg.eval.rollout_steps, policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
    
        current_step = (i + 1) * frames_per_batch
        if current_step % cfg.render.step_number == 0:
            record_current_model(env, cfg, current_step,policy_module)
                    
    
        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()
    
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (test)")
    plt.show()

if __name__ == "__main__":
    main()
