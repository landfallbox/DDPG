import numpy as np

from model.replay_buffer.memory import Memory
from model.replay_buffer.ring_buffer import RingBuffer
from common.utils import zeroed_observation, sample_batch_indexes
from common.constants import Experience

class SequentialMemory(Memory):
    """Fixed-size sequential experience replay buffer.
    
    This is a memory efficient implementation of the experience replay buffer.
    It stores the observations, actions, rewards, and terminal flags in separate ring buffers.
    This allows for efficient storage and retrieval of experiences, especially when the memory size is large.
    
    Attributes:
        limit: int
            The maximum number of experiences to store in the buffer.
        actions: RingBuffer
            A ring buffer to store actions.
        rewards: RingBuffer
            A ring buffer to store rewards.
        terminals: RingBuffer
            A ring buffer to store terminal flags.
        observations: RingBuffer
            A ring buffer to store observations.
    """
    def __init__(self, limit, **kwargs):
        super().__init__(**kwargs)
        
        self.limit = limit

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        """Sample a batch of experiences.
        
        Parameters:
            batch_size: int
                The number of experiences to sample.
            batch_idxs: list or None
                If provided, the specific indexes to sample. If None, random indexes will be sampled.
                
        Returns:
            list of Experience
                A list of sampled experiences.
        """
        if batch_idxs is None:
            # Draw random indexes such that we have at least a single entry before each index.
            batch_idxs = sample_batch_indexes(0, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        
        # ensure sampled indexes are valid
        assert np.min(batch_idxs) >= 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx = sample_batch_indexes(1, self.nb_entries, size=1)[0]
                terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            assert 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                current_terminal = self.terminals[current_idx - 1] if current_idx - 1 > 0 else False
                if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size
        return experiences

    def sample_and_split(self, batch_size, batch_idxs=None):
        """Sample a batch of experiences and split them into their components.
        
        Parameters:
            batch_size: int
                The number of experiences to sample.
            batch_idxs: list or None
                If provided, the specific indexes to sample. If None, random indexes will be sampled.
                
        Returns:
            tuple of np.ndarray
                A tuple containing batches of (state0, action, reward, state1, terminal1).
        """
        experiences = self.sample(batch_size, batch_idxs)

        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state0_batch.append(e.state0)
            state1_batch.append(e.state1)
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            terminal1_batch.append(0. if e.terminal1 else 1.)

        # Prepare and validate parameters.
        state0_batch = np.array(state0_batch).reshape(batch_size,-1)
        state1_batch = np.array(state1_batch).reshape(batch_size,-1)
        terminal1_batch = np.array(terminal1_batch).reshape(batch_size,-1)
        reward_batch = np.array(reward_batch).reshape(batch_size,-1)
        action_batch = np.array(action_batch).reshape(batch_size,-1)

        return state0_batch, action_batch, reward_batch, state1_batch, terminal1_batch


    def append(self, observation, action, reward, terminal, training=True):
        """Append a new experience to the memory.
        
        Parameters:
            observation: object
                The observation at the current time step.
            action: object
                The action taken at the current time step.
            reward: float
                The reward received at the current time step.
            terminal: bool
                Whether the next state is terminal or not.
            training: bool
                Whether the memory is being updated during training or not.
        """
        # 更新基类的最近缓存（仅需要 observation 与 terminal）
        super(SequentialMemory, self).append(observation, terminal)

        # 训练期间写入环形缓冲
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    @property
    def nb_entries(self):
        """Return the number of experiences currently stored in the buffer.
        
        Returns:
            int
                The number of experiences currently stored in the buffer.
        """
        return len(self.observations)

    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config