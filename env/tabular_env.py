"""TabularPredictionEnv (ASHRAE 专用)
- 无 gym 依赖；提供简易 Box 空间描述
- 支持 sequential 与 single_step 两种采样模式
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any, Tuple

@dataclass
class Box:
    low: np.ndarray
    high: np.ndarray
    shape: Tuple[int, ...]
    dtype: Any
    def seed(self, seed=None):
        pass

class TabularPredictionEnv:
    def __init__(self, X: np.ndarray, y_norm: np.ndarray, max_episode_length: int | None = None,
                 shuffle: bool = False, seed: int | None = None, sample_mode: str = 'sequential'):
        assert len(X) == len(y_norm)
        self.X = X.astype(np.float32)
        self.y = y_norm.astype(np.float32).reshape(-1)
        self.n = len(self.X)
        obs_dim = self.X.shape[1]
        self.observation_space = Box(low=np.full((obs_dim,), -np.inf, dtype=np.float32),
                                     high=np.full((obs_dim,), np.inf, dtype=np.float32),
                                     shape=(obs_dim,), dtype=np.float32)
        self.action_space = Box(low=np.array([-1.0], dtype=np.float32),
                                high=np.array([1.0], dtype=np.float32),
                                shape=(1,), dtype=np.float32)
        self._idx = 0
        self.max_episode_length = max_episode_length or self.n
        self.shuffle = shuffle
        self.sample_mode = sample_mode  # 'sequential' or 'single_step'
        self.rng = np.random.default_rng(seed)
        self._order = np.arange(self.n)
        if shuffle and self.sample_mode == 'sequential':
            self.rng.shuffle(self._order)

    def seed(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            if self.shuffle and self.sample_mode == 'sequential':
                self.rng.shuffle(self._order)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.seed(seed)
        if self.sample_mode == 'single_step':
            # pick a random data index each episode
            rand_pos = int(self.rng.integers(0, self.n))
            self._idx = rand_pos
            obs = self.X[rand_pos]
            return obs, {}
        # sequential mode
        self._idx = 0
        if self.shuffle:
            self.rng.shuffle(self._order)
        obs = self.X[self._order[self._idx]]
        return obs, {}

    def step(self, action):
        if self.sample_mode == 'single_step':
            a = float(np.clip(action, -1.0, 1.0))
            target = float(self.y[self._idx])
            reward = - (a - target) ** 2
            info = {"row_index": int(self._idx), "target": target, "pred": a}
            # single step episode ended
            return self.X[self._idx], reward, True, False, info
        # sequential mode
        a = float(np.clip(action, -1.0, 1.0))
        target = float(self.y[self._order[self._idx]])
        reward = - (a - target) ** 2
        self._idx += 1
        terminated = False
        truncated = False
        if self._idx >= self.n or self._idx >= self.max_episode_length:
            terminated = True
            obs = self.X[self._order[min(self._idx-1, self.n-1)]]  # last obs
        else:
            obs = self.X[self._order[self._idx]]
        info = {"row_index": int(self._order[self._idx-1]), "target": target, "pred": a}
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass  # no-op for tabular data
