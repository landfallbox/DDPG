"""
@Project     : DDPG 
@File        : ddpg.py
@IDE         : PyCharm 
@Author      : landfallbox
@Date        : 2025/9/9 星期二 17:16 
@Description : 标准 DDPG 算法实现 (PyTorch)。
"""
import numpy as np

import torch
from torch.optim import Adam
from model.actor import Actor
from model.critic import Critic
from common.utils import hard_update, to_tensor, soft_update, to_numpy
from model.replay_buffer.sequential_memory import SequentialMemory
from model.random_process import OrnsteinUhlenbeckProcess
from common.constants import USE_CUDA, criterion

class DDPG(object):
    """Deep Deterministic Policy Gradient (DDPG) Agent.
    
    Attributes:
        nb_states: int
            The number of dimensions in the state space.
        nb_actions: int
            The number of dimensions in the action space.
        actor: Actor
            The actor network that maps states to actions.
        actor_target: Actor
            The target actor network used for stable training.
        actor_optim: Adam
            The optimizer for the actor network.
        critic: Critic
            The critic network that estimates the Q-value of state-action pairs.
        critic_target: Critic
            The target critic network used for stable training.
        critic_optim: Adam
            The optimizer for the critic network.
        memory: SequentialMemory
            The replay buffer to store experiences.
        random_process: OrnsteinUhlenbeckProcess
            The noise process for action exploration.
        batch_size: int
            The number of experiences to sample from the replay buffer for each training step.
        tau: float
            The soft update parameter for updating target networks.
        discount: float
            The discount factor for future rewards.
        depsilon: float
            The rate at which exploration noise decays.
        epsilon: float
            The current exploration noise scale.
        s_t: np.array or None
            The most recent state observed by the agent.
        a_t: np.array or None
            The most recent action taken by the agent.
        is_training: bool
            Flag indicating whether the agent is in training mode.
    """

    def __init__(self, nb_states, nb_actions, args):
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon
 
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True
 
        if USE_CUDA: self.cuda()

    def update_policy(self):
        """Update the policy by sampling from the replay buffer.

        Uses the Adam optimizer to update the actor and critic networks based on a batch of experiences
        sampled from the replay buffer. The target networks are updated using soft updates.
        """

        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        with torch.no_grad():
            next_states = to_tensor(next_state_batch)
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target([next_states, next_actions])

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float32))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1.,1.,self.nb_actions)
        self.a_t = action
        return action
    
    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)
        action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action
    
    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self, output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
