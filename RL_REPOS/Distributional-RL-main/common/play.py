import gymnasium as gym
import numpy as np
from .atari_wrappers import make_atari
import time
from gymnasium.wrappers import RecordVideo
from pathlib import Path
import datetime


class Evaluator:
    def __init__(self, agent, config, max_episode=3):
        self.config = config
        self.env = make_atari(
            self.config.env.env_name, 
            render_mode="rgb_array",
            episodic_life=False, 
            clip_reward=False, 
            seed=config.seed,
        )


        video_dir = self.get_video_path(config)
        
        self.env = RecordVideo(
            self.env,
            video_folder=video_dir,
            episode_trigger=lambda ep_id: True,
            name_prefix="eval"
        )

        self.max_episode = max_episode
        self.agent = agent
        self.agent.prepare_to_play()

    def evaluate(self):
        total_reward = 0
        print("--------Play mode--------")

        for ep in range(self.max_episode):
            obs, info = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.agent.choose_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

            print(f"Episode {ep + 1} reward: {episode_reward}")
            total_reward += episode_reward

        print("Average Total Reward:", total_reward / self.max_episode)
        self.env.close()
        self.agent.restore_after_play()

    def get_video_path(self, config):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")#datetime.datetime.now().strftime("%Y%m%d_%H%M")
        agent_name = config.agent.name.upper()
        env_name = config.env.env_name.replace("NoFrameskip-v4", "") \
                       .replace("-", "") \
                       .capitalize()
        run_name = f"run-{agent_name}-{env_name}-{timestamp}"
        video_path = Path("video") / run_name
        video_path.mkdir(parents=True, exist_ok=True)
        return str(video_path)


