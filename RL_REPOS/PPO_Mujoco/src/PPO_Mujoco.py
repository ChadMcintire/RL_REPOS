import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing

import numpy as np
from collections import defaultdict

import torch
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from tqdm import tqdm
from tensordict import TensorDict


import hydra
from omegaconf import DictConfig

import os
os.environ["MUJOCO_GL"] = "egl"

from utils.record import record_current_model
from utils.plotting import plot_training_logs
from utils.create_env import make_env
from models.mlp import build_policy_module, build_value_module
from components.ppo_components import build_ppo_algorithm
from eval.evaluator import evaluate_policy
from trainer.train_ppo import run_training_loop


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    from hydra.core.hydra_config import HydraConfig
    print("Working dir:", HydraConfig.get().run.dir)

    # --- Training config ---
    total_frames = cfg.training.total_frames

    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    
    env = make_env(cfg, device)
    
    print("normalization constant shape:", env.transform[0].loc.shape)
    print("observation_spec:", env.observation_spec)
    print("reward_spec:", env.reward_spec)
    print("input_spec:", env.input_spec)
    print("action_spec (as defined by input_spec):", env.action_spec)
    
    check_env_specs(env)
    
    rollout = env.rollout(3)
    print("rollout of three steps:", rollout)
    print("Shape of the rollout TensorDict:", rollout.batch_size)

    policy_module = build_policy_module(cfg, env, device)
    value_module = build_value_module(cfg, env, device)
    
    print("Running policy:", policy_module(env.reset()))
    print("Running value:", value_module(env.reset()))
    
    ppo = build_ppo_algorithm(cfg, env, policy_module, value_module)
    
    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""
    
    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:

    # --- Training Loop ---
    for i, cum_reward_str, stepcount_str, lr_str in run_training_loop(
        ppo, env, cfg, logs, pbar, policy_module, device
    ):

        # evaluate
        if i % 10 == 0:
            eval_str = evaluate_policy(env, policy_module, cfg, logs)


        # record video of what happened
        current_step = (i + 1) * cfg.training.frames_per_batch
        if current_step % cfg.render.step_number == 0:
            record_current_model(env, cfg, current_step,policy_module)
                    
        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
    
        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        ppo.scheduler.step()
    
    plot_training_logs(logs)

if __name__ == "__main__":
    main()
