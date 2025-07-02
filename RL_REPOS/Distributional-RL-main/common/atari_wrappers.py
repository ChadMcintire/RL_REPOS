from abc import ABC
import cv2
import gymnasium as gym
import ale_py
import numpy as np
from gymnasium.wrappers import TimeLimit


def make_atari(env_name: str,
               episodic_life: bool = True,
               clip_reward: bool = True,
               seed: int = 123,
               render_mode=None
               ):
    gym.register_envs(ale_py)
    env = gym.make(env_name, render_mode=render_mode)
    if "NoFrameskip" not in env.spec.id:  # noqa
        raise ValueError(f"env should be from `NoFrameskip` type got: {env_name}")  # noqa
    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ResizedAndGrayscaleEnv(env)
    if clip_reward:
        env = ClipRewardEnv(env)
    env = StackFrameEnv(env)
    env = TimeLimit(env, max_episode_steps=100000)  # explicitly set step limit

    env.reset(seed=seed)

    np.random.seed(seed)
    return env


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        noops = np.random.randint(1, self.noop_max + 1)  # noqa
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(seed=seed, options=options)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)

        self.obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self.skip = skip

    def step(self, action):
        total_reward = 0
        terminated = truncated = False
        info = {}
        for i in range(self.skip):
            obs, reward, term, trunc, info = self.env.step(action)

            if i == self.skip - 2:
                self.obs_buffer[0] = obs
            if i == self.skip - 1:
                self.obs_buffer[1] = obs
            total_reward += reward
            if term or trunc:
                terminated, truncated = term, trunc
                break
        max_frame = self.obs_buffer.max(axis=0)  # noqa
        return max_frame, total_reward, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        if self.was_real_done:
            obs, info = self.env.reset(seed=seed, options=options)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(seed=seed, options=options)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(seed=seed, options=options)
        return obs, info


class ResizedAndGrayscaleEnv(gym.ObservationWrapper, ABC):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width), dtype=np.uint8
        )

    def observation(self, observation):
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame


# np.sign(x) =
#   -1 if x < 0
#    0 if x == 0
#   +1 if x > 0
class ClipRewardEnv(gym.RewardWrapper, ABC):
    def reward(self, reward):
        return np.sign(reward)


class StackFrameEnv(gym.Wrapper):
    def __init__(self, env, stack_size=4):
        super().__init__(env)
        self.stack_size = stack_size
        w, h = env.observation_space.shape
        self.frames = np.zeros((stack_size, w, h), dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = obs.astype(np.uint8)
        self.frames = np.stack([obs] * self.stack_size, axis=0)# PyTorch's channel axis is 0!
        return self.frames, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = obs.astype(np.uint8)
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = obs
        return self.frames, reward, terminated, truncated, info
