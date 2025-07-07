import time
import numpy as np
import psutil
import torch
import os
import datetime
import glob
from collections import deque
from threading import Thread
import wandb
from pathlib import Path
from omegaconf import OmegaConf


class Logger:
    def __init__(self, agent, config):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")#datetime.datetime.now().strftime("%Y%m%d_%H%M")
        agent_name = config.agent.name.upper()
        env_name = config.env.env_name.replace("NoFrameskip-v4", "") \
                           .replace("-", "") \
                           .capitalize()
        self.run_name = f"run-{agent_name}-{env_name}-{timestamp}"
        self.log_dir = self.run_name 

        self.config = config
        self.agent = agent
        self.start_time = 0
        self.duration = 0
        self.running_reward = 0
        self.running_loss = 0
        self.max_episode_reward = -np.inf
        self.moving_avg_window = 10
        self.moving_weights = np.repeat(1.0, self.moving_avg_window) / self.moving_avg_window
        self.last_10_ep_rewards = deque(maxlen=self.moving_avg_window)
        self.thread = Thread()
        self.to_gb = lambda b: b / 1024 / 1024 / 1024
        self.wandb_active = False
        flat_config = OmegaConf.to_container(config, resolve=True)


        try:
            wandb.init(project=agent_name,
                       config=flat_config,
                       job_type="train",
                       name=self.run_name,
                       id=self.run_name,
                       resume="allow"
                       )
        except Exception as e:
            print(f"[wandb] Init failed: {e}")
            self.wandb_active = False
        else:
            self.wandb_active = True

        if not self.config.get("do_test", False) and self.config.get("save_weights", True):
            self.create_weights_folder()


    def create_weights_folder(self):
        weights_path = Path("runs") / self.log_dir / "weights"
        weights_path.mkdir(parents=True, exist_ok=True)  # Create the full directory path if it doesn't exist

    def on(self):
        self.start_time = time.time()

    def off(self):
        self.duration = time.time() - self.start_time

    def log(self, **kwargs):
        required_keys = ["episode", "reward/episode_reward", "loss", "step", "e_len"]
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required log key: '{key}'")

        episode = kwargs["episode"]
        episode_reward = kwargs["reward/episode_reward"]
        loss = kwargs["loss"]
        step = kwargs["step"]
        e_len = kwargs["e_len"]

        self.max_episode_reward = max(self.max_episode_reward, episode_reward)

        if self.running_reward == 0:
            self.running_reward = episode_reward
            self.running_loss = loss

        else:
            self.running_reward = 0.99 * self.running_reward + 0.01 * episode_reward
            self.running_loss = 0.9 * self.running_loss + 0.1 * loss

        self.last_10_ep_rewards.append(episode_reward)
        last_10_avg = (
            np.convolve(self.last_10_ep_rewards, self.moving_weights, 'valid')[0]
            if len(self.last_10_ep_rewards) == self.moving_avg_window else 0 # It is not correct but does not matter.
        )

        memory = psutil.virtual_memory()
        assert self.to_gb(memory.used) < 0.99 * self.to_gb(memory.total)

        if episode % (self.config.interval // 3) == 0:
            self.save_weights(step)

        if episode % self.config.interval == 0:
            print(
                f"E: {episode} | "
                f"E_Reward: {episode_reward:.1f} | "
                f"E_Running_Reward: {self.running_reward:.2f} | "
                f"Mem_Len: {len(self.agent.memory)} | "
                f"Mean_steps_time: {self.duration / e_len:.2f} | "
                f"RAM Usage: {self.to_gb(memory.used):.1f} / {self.to_gb(memory.total):.1f} GB"
                f"eps: {self.agent.exp_eps:.2f} | "
                f"Time: {datetime.datetime.now().strftime('%H:%M:%S')} | "
                f"Step: {step}"
            )

        metrics = {
            "Running episode reward": self.running_reward,
            "Max episode reward": self.max_episode_reward,
            "Moving last 10 episode rewards": last_10_avg,
            "Running Loss": self.running_loss,
            "Episode": episode,
            "Episode Length": e_len,
            "Total Steps": step
        }

        # Merge in any extra metrics from kwargs
        for k, v in kwargs.items():
            if k not in metrics:
                metrics[k] = v


        if self.wandb_active:
            if self.thread.is_alive():
                self.thread.join(timeout=1)
            self.thread = Thread(target=self.log_metrics, args=(metrics,))
            self.thread.start()


    @staticmethod
    def log_metrics(metrics):
        try:
            wandb.log(metrics)
        except Exception as e:
            print(f"[wandb] Log failed: {e}")


    def save_weights(self, step):
        # Define the path: runs/<log_dir>/weights/params.pth
        weights_path = Path("runs") / self.log_dir / "weights"
        weights_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        path = weights_path / f"params_step{step}.pth"
        print("path", path)
        torch.save({"online_model_state_dict": self.agent.online_model.state_dict()}, path)

    def load_weights(self):
        weights_path = Path("runs") / self.log_dir / "weights"
        checkpoint_files = list(weights_path.glob("params_step*.pth"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {weights_path}")

        # pick checkpoint with max step
        def extract_step(f):
            import re
            match = re.search(r"params_step(\d+)\.pth", f.name)
            return int(match.group(1)) if match else -1

        checkpoint_files.sort(key=extract_step)
        latest_checkpoint = checkpoint_files[-1]

        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        return checkpoint

