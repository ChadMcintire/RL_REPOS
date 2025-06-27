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


class Logger:
    def __init__(self, agent, **config):
        self.config = config
        self.agent = agent
        self.log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
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


        try:
            wandb.init(project=self.config["agent_name"],
                       config=config,
                       job_type="train",
                       name=self.log_dir)
        except Exception as e:
            print(f"[wandb] Init failed: {e}")
            self.wandb_active = False
        else:
            self.wandb_active = True

        if not self.config.get("do_test", False) and self.config.get("save_weights", True):
            self.create_weights_folder(self.log_dir)


    @staticmethod
    def create_weights_folder(dir_name):
        weights_dir = Path("weights")
        weights_dir.mkdir(exist_ok=True)
        (weights_dir / dir_name).mkdir(exist_ok=True)

    def on(self):
        self.start_time = time.time()

    def off(self):
        self.duration = time.time() - self.start_time

    def log(self, **kwargs):
        required_keys = ["episode", "episode_reward", "loss", "step", "e_len"]
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required log key: '{key}'")

        episode = kwargs["episode"]
        episode_reward = kwargs["episode_reward"]
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

        if episode % (self.config["interval"] // 3) == 0:
            self.save_weights()

        if episode % self.config["interval"] == 0:
            print(
                f"E: {episode} | "
                f"E_Reward: {episode_reward:.1f} | "
                f"E_Running_Reward: {self.running_reward:.2f} | "
                f"Mem_Len: {len(self.agent.memory)} | "
                f"Mean_steps_time: {self.duration / e_len:.2f} | "
                f"{self.to_gb(memory.used):.1f}/{self.to_gb(memory.total):.1f} GB RAM | "
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

        if self.thread.is_alive():
            self.thread.join(timeout=1)
        self.thread = Thread(target=self.log_metrics, args=(metrics,))
        self.thread.start()

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

    def save_weights(self):
        torch.save({"online_model_state_dict": self.agent.online_model.state_dict()},
                   "weights/" + self.log_dir + "/params.pth")


    def load_weights(self):
        model_dir = sorted(glob.glob("weights/*"))[-1]
        checkpoint = torch.load(os.path.join(model_dir, "params.pth"))
        self.log_dir = os.path.basename(model_dir)
        return checkpoint
