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
        # 新增: 记录每次评估的 MSE / MAE (逐步平均) —— 针对监督式 tabular 奖励结构
        self.mse_hist = []  # per evaluation
        self.mae_hist = []
        self.last_eval_stats = {}

    def __call__(self, env, policy, debug=False, visualize=False, save=True):
        self.is_training = False
        result = []
        total_squared_error = 0.0
        total_abs_error = 0.0
        total_steps_all = 0

        for episode in range(self.num_episodes):
            obs_reset = env.reset()
            observation = obs_reset[0] if isinstance(obs_reset, tuple) else obs_reset
            episode_steps = 0
            episode_reward = 0.
            assert observation is not None
            done = False
            while not done:
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
                episode_reward += reward
                episode_steps += 1
                # 统计 error: 针对 tabular env info 中有 target/pred
                if isinstance(info, dict) and 'target' in info and 'pred' in info:
                    err = float(info['pred']) - float(info['target'])
                    se = err * err
                    total_squared_error += se
                    total_abs_error += abs(err)
                else:
                    # 退化：无法获取 pred/target 时用 reward 估计 (若 reward = -err^2)
                    if reward <= 0:  # 避免正奖励场景误判
                        total_squared_error += (-reward)
                total_steps_all += 1
            if debug:
                logger.debug('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            result.append(episode_reward)

        result = np.array(result).reshape(-1,1)
        self.results = np.hstack([self.results, result])

        # 计算整体 MSE/MAE (跨 num_episodes 聚合)
        if total_steps_all > 0:
            mse = total_squared_error / total_steps_all
            mae = total_abs_error / total_steps_all if total_abs_error > 0 else float('nan') if total_squared_error == 0 else np.sqrt(total_squared_error/total_steps_all)
        else:
            mse = float('nan')
            mae = float('nan')
        self.mse_hist.append(mse)
        self.mae_hist.append(mae)
        self.last_eval_stats = {'mse': mse, 'mae': mae, 'steps': total_steps_all}

        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))
        return np.mean(result)

    def save_results(self, fn):
        """Save evaluation results to a file.
        
        Parameters:
            fn: str
                Filename (without extension) to save the results.
        """
        if self.results.size == 0:
            logger.warning("Evaluator.save_results 调用时没有可保存的数据 (results 为空)。")
            return
        y = np.mean(self.results, axis=0)
        x = list(range(0, self.results.shape[1] * self.interval, self.interval))
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Average Reward (episode cumulative)')
        if len(y) != len(x):
            min_len = min(len(y), len(x))
            y = y[:min_len]
            x = x[:min_len]
        if len(y) > 0:
            ax.plot(x, y, '-o', linewidth=1.5, markersize=4, label='Avg Episode Reward')
            # 若有 MSE 记录, 同图右轴显示
            if self.mse_hist:
                ax2 = ax.twinx()
                ax2.set_ylabel('MSE')
                mses = self.mse_hist[:len(y)]
                ax2.plot(x[:len(mses)], mses, '--', color='tab:orange', label='MSE')
                # 合并图例
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc='best')
            else:
                ax.legend(loc='best')
        else:
            logger.warning("Evaluator.save_results: y 为空，未绘制曲线。")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fn + '.png')
        plt.close(fig)
        # 保存所有指标
        try:
            savemat(fn + '.mat', {
                'reward': self.results,
                'mse': np.array(self.mse_hist).reshape(1,-1),
                'mae': np.array(self.mae_hist).reshape(1,-1)
            })
        except Exception as e:
            logger.warning(f"保存评估 mat 失败: {e}")