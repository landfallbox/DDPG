import gymnasium

from gymnasium import spaces

class NormalizedEnv(gymnasium.ActionWrapper):
    """ Wrap action """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.env.action_space, spaces.Box), "只支持 Box 连续动作空间"
        self._orig_low = self.env.action_space.low
        self._orig_high = self.env.action_space.high
        # 对外暴露规范化后的动作空间 [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self.env.action_space.shape,
            dtype=self.env.action_space.dtype,
        )

    def action(self, action):
        """ 将动作从 [-1, 1] 映射到环境的实际动作空间 """
        act_k = (self._orig_high - self._orig_low) / 2.0
        act_b = (self._orig_high + self._orig_low) / 2.0
        return act_k * action + act_b

    def reverse_action(self, action):
        """ 将动作从环境的实际动作空间映射到 [-1, 1] """
        act_k_inv = 2.0 / (self._orig_high - self._orig_low)
        act_b = (self._orig_high + self._orig_low) / 2.0
        return act_k_inv * (action - act_b)

    def seed(self, seed: int | None = None):
        """兼容旧的 env.seed(seed) 接口。"""
        try:
            self.env.reset(seed=seed)
        except TypeError:
            # 老版本环境
            if hasattr(self.env, 'seed'):
                self.env.seed(seed)
        # 同步空间的随机性（如果支持）
        if hasattr(self.env.action_space, 'seed'):
            self.env.action_space.seed(seed)
        if hasattr(self.env.observation_space, 'seed'):
            self.env.observation_space.seed(seed)
