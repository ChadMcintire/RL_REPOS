from dataclasses import dataclass
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
import torch

@dataclass
class PPOComponents:
    collector: SyncDataCollector
    replay_buffer: ReplayBuffer
    advantage_module: GAE
    loss_module: ClipPPOLoss
    optim: Optimizer
    scheduler: _LRScheduler


def build_ppo_algorithm(cfg, env, policy_module, value_module) -> PPOComponents:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=cfg.training.frames_per_batch,
        total_frames=cfg.training.total_frames,
        split_trajs=cfg.algo.ppo.split_trajs,
        device=device,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.training.frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    advantage_module = GAE(
        gamma=cfg.training.gamma,
        lmbda=cfg.training.lmbda,
        value_network=value_module,
        average_gae=cfg.algo.ppo.average_gae,
        device=device,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=cfg.algo.ppo.clip_epsilon,
        entropy_bonus=bool(cfg.algo.ppo.entropy_eps),
        entropy_coef=cfg.algo.ppo.entropy_eps,
        critic_coef=cfg.algo.ppo.critic_coef,
        loss_critic_type=cfg.algo.ppo.loss_critic_type,
    )

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.training.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.total_frames // cfg.training.frames_per_batch,
        eta_min=0.0,
    )

    return PPOComponents(
        collector=collector,
        replay_buffer=replay_buffer,
        advantage_module=advantage_module,
        loss_module=loss_module,
        optim=optimizer,
        scheduler=scheduler,
    )
