from collections import deque
from common.utils import zeroed_observation

class Memory(object):
    """Abstract base class for memory.
    
    Memory is used to store previous observations of the agent.
    This is useful for two reasons:
    1. It can be used to construct a state from the recent observations.
       This is especially important for many Atari games where the most recent observation does not contain the velocity information.
    2. It can be used to implement experience replay. This is especially important for off-policy algorithms like DQN or DDPG.
    
    Attributes:
        window_length: int
            The number of recent observations to return as a state.
        ignore_episode_boundaries: bool
            If True, `get_recent_state()` will ignore whether observations are from different episodes.
            This is useful in some cases, for example, when the environment is non-episodic.
            But in general, it is safer to set this to False so that states do not span across episode boundaries.
        recent_observations: deque
            A deque storing the most recent observations.
        recent_terminals: deque
            A deque storing the most recent terminal flags (indicating whether an observation is terminal).
    """

    def __init__(self, window_length, ignore_episode_boundaries=False):
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

    def sample(self):
        raise NotImplementedError()

    def append(self, observation, terminal):
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)

    def get_recent_state(self, current_observation):
        """Return the most recent state.

        将当前的 observation 和之前的 observation 结合起来，形成一个 state。State 的长度由 window_length 决定。
        如果之前的 observation 中有终止状态（terminal），并且 ignore_episode_boundaries 为 False，则不会跨越终止状态来构建 state， 避免状态序列跨越不同的 episode。
        如果之前的 observation 不足以填满 state，则使用零值填充，保证 state 的长度始终为 window_length。
        
        Parameters:
            current_observation: object
                The most recent observation to be appended to the state.

        Returns:
            list
                The most recent state consisting of the current observation and previous observations.
        """
        # This code is slightly complicated by the fact that subsequent observations might be
        # from different episodes. We ensure that an experience never spans multiple episodes.
        # This is probably not that important in practice but it seems cleaner.
        state = [current_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            current_terminal = self.recent_terminals[current_idx - 1] if current_idx - 1 >= 0 else False
            if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                # The previously handled observation was terminal, don't add the current one.
                # Otherwise we would leak into a different episode.
                break
            state.insert(0, self.recent_observations[current_idx])
        while len(state) < self.window_length:
            state.insert(0, zeroed_observation(state[0]))
        return state

    def get_config(self):
        """Return configuration of the memory."""
        config = {
            'window_length': self.window_length,
            'ignore_episode_boundaries': self.ignore_episode_boundaries,
        }
        return config