import numpy as np
import matplotlib
# Use a non-interactive backend to avoid GUI figures accumulation in headless runs
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import savemat

from common.logger import logger

class Evaluator(object):
    """Evaluator for evaluating a policy in a given environment.
    
    Attributes:
        num_episodes: int
            Number of episodes to run for evaluation.
        max_episode_length: int or None
            Maximum length of an episode. If None, episodes can run indefinitely.
        interval: int
            Interval (in timesteps) between evaluations.
        save_path: str
            Path to save evaluation results.
        results: np.ndarray
            Array to store evaluation results.
        is_training: bool
            Flag indicating whether the policy is in training mode.
    """

    def __init__(self, num_episodes, interval, save_path='', max_episode_length=None):
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.save_path = save_path
        self.results = np.array([]).reshape(num_episodes,0)

    def __call__(self, env, policy, debug=False, visualize=False, save=True):
        self.is_training = False
        observation = None
        result = []

        for episode in range(self.num_episodes):
            # reset at the start of episode (Gymnasium: reset -> (obs, info))
            obs_reset = env.reset()
            observation = obs_reset[0] if isinstance(obs_reset, tuple) else obs_reset
            episode_steps = 0
            episode_reward = 0.
                
            assert observation is not None

            # start episode
            done = False
            while not done:
                # basic operation, action ,reward, blablabla ...
                action = policy(observation)

                step_out = env.step(action)
                if isinstance(step_out, tuple) and len(step_out) == 5:
                    observation, reward, terminated, truncated, info = step_out
                    done = bool(terminated or truncated)
                else:
                    observation, reward, done, info = step_out

                if self.max_episode_length and episode_steps >= self.max_episode_length -1:
                    done = True
                
                if visualize:
                    try:
                        env.render()
                    except TypeError:
                        env.render(mode='human')

                # update
                episode_reward += reward
                episode_steps += 1

            if debug: 
                logger.debug('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            
            result.append(episode_reward)

        # store and save results, each row is an episode
        result = np.array(result).reshape(-1,1)
        self.results = np.hstack([self.results, result])

        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))
        return np.mean(result)

    def save_results(self, fn):
        """Save evaluation results to a file.
        
        Parameters:
            fn: str
                Filename (without extension) to save the results.
        """
        y = np.mean(self.results, axis=0)
        # error=np.std(self.results, axis=0)
                    
        x = range(0,self.results.shape[1]*self.interval,self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Average Reward')
        # ax.errorbar(x, y, yerr=error, fmt='-o')
        fig.tight_layout()
        fig.savefig(fn+'.png')
        plt.close(fig)  # close the figure to avoid accumulating too many open figures
        savemat(fn+'.mat', {'reward':self.results})