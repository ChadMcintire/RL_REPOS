import gymnasium as gym
import numpy as np
from .atari_wrappers import make_atari
import time
from gymnasium.wrappers import RecordVideo
from pathlib import Path
import datetime


class Evaluator:
    def __init__(self, agent, config, max_episodes=3):
        self.config = config
        self.agent = agent
        self.max_episodes = max_episodes

        self.env = self._make_eval_env()
        self.agent.prepare_to_play()

    def _make_eval_env(self):
        """Create the Atari environment with video recording enabled."""
        env = make_atari(
            self.config.env.env_name, 
            render_mode="rgb_array",
            episodic_life=False, 
            clip_reward=False, 
            seed=self.config.seed,
        )

        video_dir = self._get_video_path(self.config)
        
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep_id: True,
            name_prefix="eval"
        )
        return env


    def evaluate(self):
        total_reward = 0
        print("--------Play mode--------")

        for ep in range(self.max_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.agent.choose_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

            print(f"[Episode {ep + 1}] Reward: {episode_reward:.2f}")
            total_reward += episode_reward

        avg_reward = total_reward / self.max_episodes
        print(f"Average Total Reward: {avg_reward:.2f}")
        self.env.close()
        self.agent.restore_after_play()

    def _get_video_path(self, config):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        agent_name = config.agent.name.upper()
        env_name = config.env.env_name.replace("NoFrameskip-v4", "") \
                       .replace("-", "") \
                       .capitalize()
        run_name = f"run-{agent_name}-{env_name}-{timestamp}"
        video_path = Path("video") / run_name
        video_path.mkdir(parents=True, exist_ok=True)
        return str(video_path)


