"""
@Project     : DDPG 
@File        : util.py
@IDE         : PyCharm 
@Author      : landfallbox
@Date        : 2025/9/10 星期三 14:43 
@Description : 提供工具函数
"""
from copy import deepcopy
import os
import numpy as np
import torch
import random

from common.constants import results_dir, FLOAT, USE_CUDA
from common.logger import logger


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all the results
    on the same plots with different names.

    Parameters:
        parent_dir: str
            Path of the directory containing all experiment runs.
        env_name: str
            Name of the environment.

    Returns:
        output_dir: str
            Path of directory to save this run's.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            logger.error("Failed to get experiment id: {}".format(folder_name))
            raise
    experiment_id += 1

    if not os.path.exists(results_dir):
        logger.debug("Creating results dir: {} for directory does not exist.".format(results_dir))
        os.makedirs(results_dir, exist_ok=True)

    output_dir = os.path.join(results_dir, env_name)
    output_dir = output_dir + '-run{}'.format(experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def fanin_init(size, fanin=None):
    """Fanin initialization of network weights.

    Parameters:
        size: tuple
            Size of the weight tensor.
        fanin: int
            Number of input units in the weight tensor.

    Returns:
        torch.Tensor
    """
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

def hard_update(target, source):
    """Hard update model parameters.

    Copy network parameters from source to target.

    Parameters:
        target: torch.nn.Module
            Target network (weights will be copied to).
        source: torch.nn.Module
            Source network (weights will be copied from).
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    """Soft update model parameters.
    
    Performs a soft update of the target network's parameters towards the source network's parameters.
    
    Parameters:
        target: torch.nn.Module
            Target network (weights will be updated).
        source: torch.nn.Module
            Source network (weights will be used for the update).
        tau: float
            Interpolation parameter, with 0 < tau <= 1. A value of 1 means a hard update.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def zeroed_observation(observation):
    """Create a zeroed observation.
    
    Parameters:
        observation: object
            Observation to be zeroed.
    Returns:
        object
            Zeroed observation which has the same shape as the input observation.
    """
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    elif hasattr(observation, '__iter__'):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.

def sample_batch_indexes(low, high, size):
    """Sample batch indexes.
    
    Parameters:
        low: int
            Lower bound (inclusive).
        high: int
            Upper bound (exclusive).
        size: int
            Number of indexes to sample.
            
    Returns:
        list of int
            List of sampled indexes.
    """
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        logger.warning('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.randint(low, high, size=size)
        
    assert len(batch_idxs) == size, "len(batch_idxs)={}, size={}".format(len(batch_idxs), size)
    return batch_idxs

def to_tensor(ndarray, requires_grad=False, dtype=FLOAT):
    # 忽略传入的 dtype 常量，统一使用 float32 并放置到合适设备
    device = 'cuda' if USE_CUDA else 'cpu'
    t = torch.as_tensor(ndarray, dtype=torch.float32, device=device)
    if requires_grad:
        t.requires_grad_(True)
    return t

def to_numpy(var):
    # 安全地从 Tensor 转为 numpy，自动搬到 CPU 并去梯度
    if isinstance(var, torch.Tensor):
        return var.detach().cpu().numpy()
    return np.array(var)

def train(num_iterations, agent, env, evaluate, validate_steps, output, warmup, max_episode_length=None, debug=False):
    """Train the agent in the environment.
    
    Parameters:
        num_iterations: int
            Number of training iterations.
        agent: Agent
            The agent to be trained.
        env: gym.Env
            The environment to train the agent in.
        evaluate: Evaluator or None
            Evaluator for evaluating the agent during training. If None, no evaluation is performed.
        validate_steps: int
            Interval (in timesteps) between evaluations.
        output: str
            Path to save models and results.
        warmup: int
            Number of initial timesteps to fill the replay memory before training starts.
        max_episode_length: int or None
            Maximum length of an episode. If None, episodes can run indefinitely.
        debug: bool
            If True, debug information will be printed. Default is False.
    """
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None

    while step < num_iterations:
        # reset if it is the start of episode
        if observation is None:
            # Gymnasium API: reset returns (obs, info)
            obs_reset = env.reset()
            observation = deepcopy(obs_reset[0] if isinstance(obs_reset, tuple) else obs_reset)
            agent.reset(observation)

        # agent pick action ...
        if step <= warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)

        # env response with next_observation, reward, terminate_info (Gymnasium API)
        step_out = env.step(action)
        if len(step_out) == 5:
            observation2, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            # fallback to old Gym API
            observation2, reward, done, info = step_out
        observation2 = deepcopy(observation2)

        if max_episode_length and episode_steps >= (max_episode_length - 1):
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > warmup:
            agent.update_policy()

        # [optional] evaluate
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward = evaluate(env, policy, debug=False, visualize=False)
            
            if debug: 
                logger.debug('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

        # [optional] save intermideate model
        if step % int(num_iterations / 3) == 0:
            agent.save_model(output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # end of episode
            if debug: logger.debug('#{}: episode_reward:{} steps:{}'.format(episode, episode_reward, step))

            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1

def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):
    """Test the agent in the environment.
    
    Parameters:
        num_episodes: int
            Number of episodes to test.
        agent: Agent
            The agent to be tested.
        env: gym.Env
            The environment to test the agent in.
        evaluate: Evaluator or None
            Evaluator for evaluating the agent during testing. If None, no evaluation is performed.
        model_path: str
            Path to the saved model weights.
        visualize: bool
            If True, the environment will be rendered during testing. Default is True.
        debug: bool
            If True, debug information will be printed. Default is False.
    """
    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: 
            logger.debug('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))

def main():
    logger.info(get_output_folder(results_dir, "Pendulum-v0"))

if __name__ == '__main__':
    main()
