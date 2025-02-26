from pathlib import Path
import sys
project_root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, project_root)

import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import csv
import glob
import yaml
import re

from hockey.hockey_env import HockeyEnv_BasicOpponent, Mode
from models.td3.TD3 import TD3Agent
from models.td3.TD3Trainer import OpponentPool

class TD3Evaluator:
    """
    Evaluates TD3 checkpoints against different opponents.
    """
    
    def __init__(self, experiment_path, model_config=None):
        """
        Initialize the evaluator with the experiment path and model configuration.
        
        Args:
            experiment_path (str or Path): Path to the experiment directory
            model_config (dict, optional): Model configuration. If None, will try to load from experiment path
        """
        self.experiment_path = Path(experiment_path)
        self.checkpoints_dir = self.experiment_path / "results/training/saved_models"
        self.eval_results_dir = self.experiment_path / "results/evaluation"
        self.eval_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load model config from experiment path if not provided
        if model_config is None:
            config_path = self.experiment_path / "configs" / "evaluation_config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                model_config = config.get('model_config', config)
            else:
                config_path = self.experiment_path / "configs" / "training_config.yaml"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    model_config = config.get('model_config', config)
                else:
                    raise ValueError("No model config provided and couldn't find evaluation_config.yaml or training_config.yaml in experiment path/configs")
        
        self.model_config = model_config
        self.logger = self._initialize_logger()
        
        # Initialize environment and agent
        self.env = HockeyEnv_BasicOpponent(mode=Mode.NORMAL)
        self.agent = TD3Agent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            **self.model_config
        )
        
        # Initialize opponent pool
        self.opponent_pool = OpponentPool(
            env_creator=lambda: HockeyEnv_BasicOpponent(mode=Mode.NORMAL),
            experiment_path=self.experiment_path,
            file_prefix="eval"
        )

    def _initialize_logger(self):
        """Set up logging to file and console."""
        logger = logging.getLogger("TD3_Evaluator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler
            log_file = self.eval_results_dir / "evaluation.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger

    def evaluate_episode(self, opponent_type='random', deterministic=True):
        """
        Run a single evaluation episode against a specified opponent.
        
        Args:
            opponent_type (str): Type of opponent ('random', 'weak', 'strong', 'ddpg', 'dreamer')
            deterministic (bool): Whether to use deterministic actions
            
        Returns:
            tuple: (episode_reward, won, episode_length)
        """
        obs, _info = self.env.reset()
        obs_agent2 = self.env.obs_agent_two()
        episode_reward = 0
        
        # Set up opponent based on type
        if opponent_type == 'random':
            opponent = None  # Random actions will be sampled in _get_opponent_action
        elif opponent_type == 'weak':
            opponent = HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=True)
        elif opponent_type == 'strong':
            opponent = HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=False)
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")
        
        self.opponent_pool.current_opponent = opponent
        self.opponent_pool.current_opponent_type = opponent_type
        
        for t in range(250):  # max episode length
            # Get actions from both agents
            action = self.agent.act(obs, deterministic=deterministic)
            opponent_action = self._get_opponent_action(obs_agent2)
            
            # Combine actions and step environment
            joint_action = np.hstack([action, opponent_action])
            next_obs, reward, done, trunc, info = self.env.step(joint_action)
            
            episode_reward += reward
            obs = next_obs
            obs_agent2 = self.env.obs_agent_two()
            
            if done or trunc:
                break
        
        # Determine if agent won
        won = 1 if info.get("winner", 0) == 1 else 0
        return episode_reward, won, t + 1

    def _get_opponent_action(self, obs_agent2):
        """Get action from current opponent."""
        if self.opponent_pool.current_opponent is None:
            return self.env.action_space.sample()
        else:
            if hasattr(self.opponent_pool.current_opponent, 'opponent'):
                return self.opponent_pool.current_opponent.opponent.act(obs_agent2)
            return self.opponent_pool.current_opponent.act(obs_agent2, deterministic=True)

    def evaluate_checkpoint(self, checkpoint_path, num_episodes=1000):
        """
        Evaluate a checkpoint against all opponent types.
        
        Args:
            checkpoint_path (Path): Path to the checkpoint file
            num_episodes (int): Number of episodes to evaluate for each opponent type
            
        Returns:
            dict: Evaluation metrics for each opponent type
        """
        # Load checkpoint
        self.logger.info(f"Evaluating checkpoint: {checkpoint_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.agent.restore_state(checkpoint)
        
        results = {}
        opponent_types = ['random', 'weak', 'strong']
        
        for opponent_type in opponent_types:
            rewards = []
            wins = []
            lengths = []
            
            # Run evaluation episodes
            for _ in range(num_episodes):
                reward, won, length = self.evaluate_episode(opponent_type=opponent_type)
                rewards.append(reward)
                wins.append(won)
                lengths.append(length)
            
            # Compute metrics, including standard deviation for win rate
            results[opponent_type] = {
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'win_rate': float(np.mean(wins)),
                'std_win_rate': float(np.std(wins)),
                'mean_length': float(np.mean(lengths)),
                'std_length': float(np.std(lengths))
            }
            
            self.logger.info(
                f"Results vs {opponent_type}: "
                f"Win Rate = {results[opponent_type]['win_rate']:.3f} ± {results[opponent_type]['std_win_rate']:.3f}, "
                f"Mean Reward = {results[opponent_type]['mean_reward']:.3f} ± {results[opponent_type]['std_reward']:.3f}"
            )
        
        return results

    def evaluate_all_checkpoints(self, num_episodes=1000):
        """
        Evaluate all checkpoints in the experiment directory.
        
        Args:
            num_episodes (int): Number of episodes to evaluate for each opponent type
        """
        # Find all checkpoint files
        checkpoint_pattern = str(self.checkpoints_dir / "*.pth")
        checkpoint_files = sorted(
            glob.glob(checkpoint_pattern),
            key=lambda file: int(re.search(r'ep(\d+)', Path(file).stem, re.IGNORECASE).group(1))
        )
        
        # Filter to include only checkpoints for every 1000 episodes
        checkpoint_files = [file for file in checkpoint_files if int(re.search(r'ep(\d+)', Path(file).stem, re.IGNORECASE).group(1)) % 1000 == 0]
        
        if not checkpoint_files:
            self.logger.error(f"No checkpoint files found in {self.checkpoints_dir}")
            return
        
        # Prepare results CSV file with added std_win_rate field
        results_file = self.eval_results_dir / "evaluation_results.csv"
        fieldnames = ['checkpoint', 'opponent_type', 'mean_reward', 'std_reward', 
                     'win_rate', 'std_win_rate', 'mean_length', 'std_length']
        
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Evaluate each checkpoint and write results
            for checkpoint_path in checkpoint_files:
                results = self.evaluate_checkpoint(checkpoint_path, num_episodes)
                for opponent_type, metrics in results.items():
                    row = {
                        'checkpoint': Path(checkpoint_path).name,
                        'opponent_type': opponent_type,
                        **metrics
                    }
                    writer.writerow(row)
        
        self.logger.info(f"Evaluation results saved to {results_file}")
        self._plot_results(results_file)

    def _plot_results(self, results_file):
        """
        Create plots from evaluation results with correct data alignment.
        Uses EMA smoothing and consistent styling similar to log_stats.py.
        """
        def running_mean(x, N):
            """Compute exponential moving average with smoothing factor 2/(N+1)"""
            x = np.array(x, dtype=float)
            if len(x) == 0:
                return x
            alpha = 2.0 / (N + 1)
            ema = np.empty_like(x)
            ema[0] = x[0]
            for t in range(1, len(x)):
                ema[t] = alpha * x[t] + (1 - alpha) * ema[t-1]
            return ema

        # Read all CSV rows into a list
        with open(results_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            results = list(reader)

        # Build a dictionary for quick lookup: (checkpoint, opponent_type) -> row
        data_map = {}
        for row in results:
            checkpoint = row['checkpoint']
            opp_type = row['opponent_type']
            data_map[(checkpoint, opp_type)] = row

        # Get a sorted list of unique checkpoints based on episode number
        checkpoints = sorted(set(r['checkpoint'] for r in results),
                           key=lambda x: int(re.search(r'ep(\d+)', x).group(1)))

        # Convert checkpoint filenames to integer episodes
        episodes_map = {}
        for cp in checkpoints:
            ep_num = int(re.search(r'ep(\d+)', cp).group(1))
            episodes_map[cp] = ep_num

        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 12))
        opp_types = ['random', 'weak', 'strong']
        colors = ['C0', 'C1', 'C2']  # Consistent color scheme
        window_size = 50  # EMA window size

        # Plot win rates
        for opp_type, color in zip(opp_types, colors):
            sorted_win_rates = []
            sorted_win_rate_stds = []
            sorted_episodes = []
            for cp in checkpoints:
                row = data_map[(cp, opp_type)]
                sorted_win_rates.append(float(row['win_rate']))
                sorted_win_rate_stds.append(float(row['std_win_rate']))
                sorted_episodes.append(episodes_map[cp])
            
            sorted_episodes, sorted_win_rates, sorted_win_rate_stds = zip(*sorted(zip(sorted_episodes, sorted_win_rates, sorted_win_rate_stds)))
            smoothed_win_rates = running_mean(sorted_win_rates, window_size)
            smoothed_win_rate_stds = running_mean(sorted_win_rate_stds, window_size)
            
            axs[0].plot(sorted_episodes, sorted_win_rates, 
                       label=f'Raw vs {opp_type}', alpha=0.3, color=color)
            axs[0].plot(sorted_episodes, smoothed_win_rates, 
                       label=f'EMA vs {opp_type} (window={window_size})', 
                       linewidth=2, color=color)
            axs[0].fill_between(sorted_episodes, 
                               np.array(smoothed_win_rates) - np.array(smoothed_win_rate_stds),
                               np.array(smoothed_win_rates) + np.array(smoothed_win_rate_stds),
                               color=color, alpha=0.1)

        axs[0].set_xlabel('Training Episodes')
        axs[0].set_ylabel('Win Rate')
        axs[0].set_title('Win Rate vs Different Opponents')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        # Plot mean rewards with error bars
        for opp_type, color in zip(opp_types, colors):
            sorted_rewards = []
            sorted_reward_stds = []
            sorted_episodes = []
            for cp in checkpoints:
                row = data_map[(cp, opp_type)]
                sorted_rewards.append(float(row['mean_reward']))
                sorted_reward_stds.append(float(row['std_reward']))
                sorted_episodes.append(episodes_map[cp])
            
            # Sort by episode number
            sorted_episodes, sorted_rewards, sorted_reward_stds = zip(*sorted(zip(sorted_episodes, sorted_rewards, sorted_reward_stds)))
            smoothed_rewards = running_mean(sorted_rewards, window_size)
            smoothed_stds = running_mean(sorted_reward_stds, window_size)
            
            axs[1].plot(sorted_episodes, sorted_rewards, 
                       label=f'Raw vs {opp_type}', alpha=0.3, color=color)
            axs[1].plot(sorted_episodes, smoothed_rewards, 
                       label=f'EMA vs {opp_type} (window={window_size})', 
                       linewidth=2, color=color)
            axs[1].fill_between(sorted_episodes, 
                               smoothed_rewards - smoothed_stds,
                               smoothed_rewards + smoothed_stds,
                               color=color, alpha=0.1)

        axs[1].set_xlabel('Training Episodes')
        axs[1].set_ylabel('Mean Reward')
        axs[1].set_title('Mean Reward vs Different Opponents')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        # Plot mean episode length with error bars
        for opp_type, color in zip(opp_types, colors):
            sorted_lengths = []
            sorted_length_stds = []
            sorted_episodes = []
            for cp in checkpoints:
                row = data_map[(cp, opp_type)]
                sorted_lengths.append(float(row['mean_length']))
                sorted_length_stds.append(float(row['std_length']))
                sorted_episodes.append(episodes_map[cp])
            
            # Sort by episode number
            sorted_episodes, sorted_lengths, sorted_length_stds = zip(*sorted(zip(sorted_episodes, sorted_lengths, sorted_length_stds)))
            smoothed_lengths = running_mean(sorted_lengths, window_size)
            smoothed_length_stds = running_mean(sorted_length_stds, window_size)
            
            axs[2].plot(sorted_episodes, sorted_lengths, 
                       label=f'Raw vs {opp_type}', alpha=0.3, color=color)
            axs[2].plot(sorted_episodes, smoothed_lengths, 
                       label=f'EMA vs {opp_type} (window={window_size})', 
                       linewidth=2, color=color)
            axs[2].fill_between(sorted_episodes, 
                               smoothed_lengths - smoothed_length_stds,
                               smoothed_lengths + smoothed_length_stds,
                               color=color, alpha=0.1)

        axs[2].set_xlabel('Training Episodes')
        axs[2].set_ylabel('Mean Episode Length')
        axs[2].set_title('Episode Length vs Different Opponents')
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.eval_results_dir / "evaluation_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Evaluation plots saved to {plot_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate TD3 checkpoints against different opponents')
    parser.add_argument('experiment_path', type=str, help='Path to the experiment directory')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of evaluation episodes per checkpoint')
    args = parser.parse_args()
    
    evaluator = TD3Evaluator(args.experiment_path)
    evaluator.evaluate_all_checkpoints(num_episodes=args.episodes)

if __name__ == '__main__':
    main() 