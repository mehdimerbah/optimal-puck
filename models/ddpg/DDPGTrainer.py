"""
DDPG Trainer - Environment interaction, logging, checkpointing, and orchestrating the training loop for DDPG.

This module handles both standard training and self-play training, managing:
  - Environment initialization (supports Hockey and gym environments)
  - Logging and saving of training statistics and checkpoints
  - Interaction with the DDPG agent and (optionally) opponent agents
  - Visualization of training progress

"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import gymnasium as gym
import pickle
import logging
import sys
from pathlib import Path

from hockey.hockey_env import HockeyEnv_BasicOpponent, Mode, HockeyEnv, BasicOpponent
from models.ddpg.DDPG import DDPGAgent


class DDPGTrainer:
    """
    Trainer for the DDPG agent, managing environment interaction, logging, checkpointing, and statistics.

    Attributes:
        env_name (str): Name of the environment.
        env_mode (str): Mode of the environment (e.g., 'train_shooting', 'train_defense', 'self_play', 'evaluation').
        env (gym.Env): The training environment.
        training_config (dict): Configuration parameters for training.
        model_config (dict): Model-specific configuration parameters.
        experiment_path (Path): Base path for saving results.
        agent (DDPGAgent): The DDPG agent instance.
        wandb_run: Optional wandb run object for logging.
        opponent_pool (list): List of saved opponent checkpoints.
        rewards (list): Accumulated episode rewards.
        lengths (list): Episode lengths.
        losses (list): Training losses.
        timestep (int): Global timestep counter.
        results_path (Path): Directory for storing results.
        training_stats_path (Path): Directory for saving statistics.
        training_logs_path (Path): Directory for saving logs.
        training_plots_path (Path): Directory for saving training plots.
        logger (logging.Logger): Logger for training messages.
    """

    def __init__(self, env_name, training_config, model_config, experiment_path, env_mode="normal", wandb_run=None):
        """
        Initialize the DDPG trainer.

        Args:
            env_name (str): Name of the environment.
            training_config (dict): Training configuration parameters.
            model_config (dict): Model configuration parameters.
            experiment_path (str or Path): Base path for the experiment.
            env_mode (str): Mode for the environment (default "normal").
            wandb_run: Optional wandb run object for logging.
        """
        self.env_name = env_name
        self.env_mode = env_mode
        self.training_config = training_config
        self.model_config = model_config
        self.experiment_path = Path(experiment_path)
        self.wandb_run = wandb_run

        # Set up the environment based on the provided mode.
        self._setup_environment()

        # Create directories for results, logs, stats, and plots.
        self._setup_directories()

        # Instantiate the DDPG agent.
        self.agent = self._initialize_agent()

        # Initialize training statistics and counters.
        self.opponent_pool = []
        self.rewards = []
        self.lengths = []
        self.losses = []
        self.timestep = 0

        # Initialize the logger and random seeds.
        self.logger = self._initialize_logger()
        self._initialize_seed()

    def _setup_environment(self):
        """Set up the environment based on env_name and env_mode."""
        if self.env_name == "HockeyEnv":
            if self.env_mode == 'train_shooting':
                self.env = HockeyEnv_BasicOpponent(mode=Mode.TRAIN_SHOOTING, weak_opponent=True)
            elif self.env_mode == 'train_defense':
                self.env = HockeyEnv_BasicOpponent(mode=Mode.TRAIN_DEFENSE, weak_opponent=True)
            elif self.env_mode == 'self_play':
                self.env = HockeyEnv(keep_mode=True, mode="NORMAL")
                self.opponent = self._sample_opponent_checkpoint
            elif self.env_mode == 'evaluation':
                self.env = HockeyEnv_BasicOpponent(mode="NORMAL", weak_opponent=True)
            else:
                self.env = HockeyEnv(keep_mode=True, mode="NORMAL")
                self.opponent = BasicOpponent(weak=True, keep_mode=True)
        elif self.env_name == "BipedalWalker-v3":
            self.env = gym.make(self.env_name, hardcore=False, render_mode="rgb_array")
        else:
            self.env = gym.make(self.env_name)

    def _setup_directories(self):
        """Create necessary directories for results, logs, stats, and plots."""
        self.results_path = self.experiment_path / "results"
        self.training_stats_path = self.results_path / "training" / "stats"
        self.training_logs_path = self.results_path / "training" / "logs"
        self.training_plots_path = self.results_path / "training" / "plots"
        self.training_stats_path.mkdir(parents=True, exist_ok=True)
        self.training_logs_path.mkdir(parents=True, exist_ok=True)
        self.training_plots_path.mkdir(parents=True, exist_ok=True)

    def _initialize_agent(self):
        """
        Instantiate the DDPGAgent using the model configuration.

        Returns:
            DDPGAgent: An initialized agent.
        """
        # Ensure the action space matches expected dimensions.
        correct_action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        agent = DDPGAgent(
            observation_space=self.env.observation_space,
            action_space=correct_action_space,
            **self.model_config
        )
        return agent

    def _update_opponent_pool(self):
        """
        Update the opponent pool by scanning the saved_models directory for checkpoints.
        """
        saved_models_dir = self.results_path / "training" / "saved_models"
        checkpoint_files = list(saved_models_dir.glob("DDPG_*_checkpoint_ep*.pth"))
        # Sort checkpoints by episode number extracted from filename.
        checkpoint_files.sort(key=lambda path: int(path.stem.split("ep")[-1]))
        self.opponent_pool = checkpoint_files

    def _sample_opponent_checkpoint(self):
        """
        Sample an opponent checkpoint from the pool using weighted probabilities.

        Returns:
            Path or None: A selected checkpoint path or None if the pool is empty.
        """
        if not self.opponent_pool:
            return None

        episodes = []
        for cp in self.opponent_pool:
            try:
                ep_str = cp.stem.split("ep")[-1]
                ep = int(ep_str)
            except Exception:
                ep = 0
            episodes.append(ep)
        episodes = np.array(episodes, dtype=np.float64)
        bias_factor = 0.01
        weights = np.exp(bias_factor * episodes)
        probabilities = weights / np.sum(weights)
        return np.random.choice(self.opponent_pool, p=probabilities)

    def _load_opponent(self, checkpoint_path):
        """
        Load an opponent agent from a saved checkpoint.

        Args:
            checkpoint_path (Path): Path to the checkpoint file.

        Returns:
            DDPGAgent: An agent with loaded state.
        """
        correct_action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        opponent = DDPGAgent(
            observation_space=self.env.observation_space,
            action_space=correct_action_space,
            **self.model_config
        )
        state = torch.load(checkpoint_path)
        opponent.restore_state(state)
        return opponent

    def _initialize_seed(self):
        """Set random seeds for reproducibility."""
        seed = self.training_config.get("seed", 42)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.env.reset(seed=seed)
            self.logger.info(f"Initialized random seeds to {seed}.")

    def _initialize_logger(self):
        """
        Initialize a logger to log training details to both stdout and a log file.

        Returns:
            logging.Logger: Configured logger instance.
        """
        log_file = self.training_logs_path / (
            f"DDPG_{self.env_name}_noise{self.model_config['noise_scale']}_"
            f"alr{float(self.model_config['learning_rate_actor'])}_"
            f"clr{float(self.model_config['learning_rate_critic'])}_"
            f"gamma{self.model_config['discount']}.log"
        )
        logger = logging.getLogger("DDPG_Trainer")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        logger.info(f"Logger initialized. Writing logs to {log_file}")
        return logger

    def _save_statistics(self):
        """
        Save training statistics (rewards, lengths, losses) to a pickle file.
        """
        stats_file = self.training_stats_path / (
            f"DDPG_{self.env_name}_noise{self.model_config['noise_scale']}_"
            f"alr{float(self.model_config['learning_rate_actor'])}_"
            f"clr{float(self.model_config['learning_rate_critic'])}_"
            f"gamma{self.model_config['discount']}_stats.pkl"
        )
        data = {
            "rewards": self.rewards,
            "lengths": self.lengths,
            "losses": self.losses,
        }
        with open(stats_file, "wb") as f:
            pickle.dump(data, f)
        self.logger.info(f"Saved training statistics to -> {stats_file}")

    def _save_checkpoint(self, episode):
        """
        Save a checkpoint of the agent's parameters.

        Args:
            episode (int): Current episode number.
        """
        saved_models_dir = self.results_path / "training" / "saved_models"
        saved_models_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = saved_models_dir / (
            f"DDPG_{self.env_name}_noise{self.model_config['noise_scale']}_"
            f"alr{float(self.model_config['learning_rate_actor'])}_"
            f"clr{float(self.model_config['learning_rate_critic'])}_"
            f"gamma{self.model_config['discount']}_checkpoint_ep{episode}.pth"
        )
        torch.save(self.agent.state(), checkpoint_path)
        self.logger.info(f"Saved checkpoint at episode {episode} -> {checkpoint_path}")

    def _final_metrics(self):
        """
        Compute final training metrics.

        Returns:
            dict: Dictionary containing average reward, average length, and final losses.
        """
        final_loss = self.losses[-1] if self.losses else (None, None)
        avg_reward = np.mean(self.rewards) if self.rewards else 0
        avg_length = np.mean(self.lengths) if self.lengths else 0
        metrics = {
            "average_reward": float(avg_reward),
            "average_length": float(avg_length),
            "final_loss_critic": float(final_loss[0]) if final_loss[0] is not None else None,
            "final_loss_actor": float(final_loss[1]) if final_loss[1] is not None else None
        }
        self.logger.info(f"Final training metrics: {metrics}")
        return metrics

    def _plot_statistics(self, window_size=50):
        """
        Plot training statistics (rewards, Q-loss, and actor loss).

        Args:
            window_size (int): Window size for smoothing curves.
        """
        def running_mean(x, N):
            cumsum = np.cumsum(np.insert(x, 0, 0))
            return (cumsum[N:] - cumsum[:-N]) / float(N)

        episodes = np.arange(1, len(self.rewards) + 1)
        smoothed_rewards = running_mean(self.rewards, window_size) if len(self.rewards) >= window_size else self.rewards
        smoothed_episodes = np.arange(window_size, len(self.rewards) + 1) if len(self.rewards) >= window_size else episodes

        # Aggregate losses per episode (averaged over training steps per episode).
        losses_array = np.array(self.losses)
        if self.losses and losses_array.size:
            losses_per_episode = np.mean(losses_array.reshape(len(self.rewards), -1, 2), axis=1)
            q_losses = losses_per_episode[:, 0]
            actor_losses = losses_per_episode[:, 1]
            smoothed_q_losses = running_mean(q_losses, window_size) if len(q_losses) >= window_size else q_losses
            smoothed_actor_losses = running_mean(actor_losses, window_size) if len(actor_losses) >= window_size else actor_losses
        else:
            q_losses = actor_losses = smoothed_q_losses = smoothed_actor_losses = []

        plt.figure(figsize=(16, 12))

        # Plot Rewards.
        plt.subplot(3, 1, 1)
        plt.plot(episodes, self.rewards, label="Raw Rewards", alpha=0.4)
        if len(self.rewards) >= window_size:
            plt.plot(smoothed_episodes, smoothed_rewards, label=f"Smoothed Rewards (window={window_size})", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Reward vs Episodes (Noise Scale={self.model_config['noise_scale']})")
        plt.legend()
        plt.grid()

        # Plot Q-Loss.
        plt.subplot(3, 1, 2)
        plt.plot(episodes, q_losses, label="Raw Q-Loss", alpha=0.4)
        if len(q_losses) >= window_size:
            plt.plot(smoothed_episodes, smoothed_q_losses, label=f"Smoothed Q-Loss (window={window_size})", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Q-Loss")
        plt.title("Q-Loss vs Episodes")
        plt.legend()
        plt.grid()

        # Plot Actor Loss.
        plt.subplot(3, 1, 3)
        plt.plot(episodes, actor_losses, label="Raw Actor Loss", alpha=0.4)
        if len(actor_losses) >= window_size:
            plt.plot(smoothed_episodes, smoothed_actor_losses, label=f"Smoothed Actor Loss (window={window_size})", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Actor Loss")
        plt.title("Actor Loss vs Episodes")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plot_file = self.training_plots_path / (
            f"DDPG_{self.env_name}_noise{self.model_config['noise_scale']}_"
            f"alr{float(self.model_config['learning_rate_actor'])}_"
            f"clr{float(self.model_config['learning_rate_critic'])}_"
            f"gamma{self.model_config['discount']}_training_plot.png"
        )
        plt.savefig(plot_file)
        self.logger.info(f"Saved training plot to {plot_file}")

    def train(self):
        """
        Execute the main training loop for the DDPG agent.

        Returns:
            dict: Final training metrics.
        """
        max_episodes = self.training_config["max_episodes"]
        max_timesteps = self.training_config["max_timesteps"]
        log_interval = self.training_config.get("log_interval", 20)
        save_interval = self.training_config.get("save_interval", 500)
        render = self.training_config.get("render", False)
        train_iter = self.training_config.get("train_iter", 32)

        self.logger.info("Starting DDPG Training...")
        self.logger.info(f"Environment: {self.env_name}, max_episodes={max_episodes}, "
                         f"max_timesteps={max_timesteps}, train_iter={train_iter}")

        for i_episode in range(1, max_episodes + 1):
            obs, _info = self.env.reset()
            self.agent.reset()
            episode_reward = 0
            win = 0
            touched = 0
            first_time_touch = 1

            for t in range(max_timesteps):
                self.timestep += 1
                action = self.agent.act(obs)
                next_obs, reward, done, trunc, _info = self.env.step(action)

                touched = max(touched, _info.get('reward_touch_puck', 0))
                current_reward = (reward +
                                  2 * _info.get('reward_closeness_to_puck', 0) +
                                  2 * _info.get('reward_puck_direction', 0) -
                                  (1 - touched) * 0.1 +
                                  touched * first_time_touch * 0.01 * t)
                first_time_touch = 1 - touched

                self.agent.store_transition((obs, action, current_reward, next_obs, done))
                obs = next_obs
                episode_reward += current_reward

                if render:
                    self.env.render()

                if done or trunc:
                    break

            win = 1 if _info.get('winner', 0) == 1 else 0

            # Update the agent using collected transitions.
            batch_losses = self.agent.train(num_updates=train_iter)
            self.losses.extend(batch_losses)
            self.rewards.append(episode_reward)
            self.lengths.append(t)

            # Log statistics to wandb if available.
            if self.wandb_run is not None:
                self.wandb_run.log({
                    "Reward": episode_reward,
                    "EpisodeLength": t,
                    "TouchRate": touched,
                    "WinRate": win
                })

            # Save checkpoint and statistics periodically.
            if i_episode % save_interval == 0:
                self._save_checkpoint(i_episode)
                self._save_statistics()

            if i_episode % log_interval == 0:
                avg_reward = np.mean(self.rewards[-log_interval:])
                avg_length = np.mean(self.lengths[-log_interval:])
                self.logger.info(
                    f"Episode {i_episode}\tAvg Length: {avg_length:.2f}\tAvg Reward: {avg_reward:.3f}"
                )

        self._save_statistics()
        self._plot_statistics()
        final_metrics = self._final_metrics()

        if self.wandb_run is not None:
            self.wandb_run.log({'average_reward': final_metrics['average_reward']})
            self.wandb_run.summary.update(final_metrics)

        return final_metrics

    def self_play_training(self):
        """
        Execute the training loop for self-play, where an opponent agent is sampled from a pool.

        Returns:
            dict: Final training metrics.
        """
        max_episodes = self.training_config["max_episodes"]
        max_timesteps = self.training_config["max_timesteps"]
        log_interval = self.training_config.get("log_interval", 20)
        save_interval = self.training_config.get("save_interval", 500)
        render = self.training_config.get("render", False)
        train_iter = self.training_config.get("train_iter", 32)

        self.logger.info("Starting DDPG Self Play Training...")
        self.logger.info(f"Environment: {self.env_name}, max_episodes={max_episodes}, "
                         f"max_timesteps={max_timesteps}, train_iter={train_iter}")

        pool_update_interval = 10
        min_basic_opponent_training = 100

        for i_episode in range(1, max_episodes + 1):
            obs, _info = self.env.reset()
            self.agent.reset()
            episode_reward = 0
            win = 0
            touched = 0
            first_time_touch = 1

            if i_episode % pool_update_interval == 0 and i_episode > min_basic_opponent_training:
                self._update_opponent_pool()
                self.logger.info(f"Updated opponent pool. Pool now has {len(self.opponent_pool)} checkpoints.")

            checkpoint = self._sample_opponent_checkpoint()
            if checkpoint is not None:
                self.opponent = self._load_opponent(checkpoint)
            else:
                self.opponent = BasicOpponent(weak=True, keep_mode=True)

            for t in range(max_timesteps):
                self.timestep += 1
                obs_agent1 = obs
                obs_agent2 = self.env.obs_agent_two()

                act1 = self.agent.act(obs_agent1)
                act2 = self.opponent.act(obs_agent2)
                combined_act = np.hstack([act1, act2])
                next_obs, reward, done, trunc, _info = self.env.step(combined_act)

                touched = max(touched, _info.get('reward_touch_puck', 0))
                current_reward = (reward +
                                  2 * _info.get('reward_closeness_to_puck', 0) -
                                  (1 - touched) * 0.1 +
                                  touched * first_time_touch * 0.01 * t +
                                  2 * _info.get('reward_puck_direction', 0))
                first_time_touch = 1 - touched

                self.agent.store_transition((obs_agent1, act1, current_reward, next_obs, done))
                obs = next_obs
                episode_reward += current_reward

                if render:
                    self.env.render()

                if done or trunc:
                    break

            win = 1 if _info.get('winner', 0) == 1 else 0

            batch_losses = self.agent.train(num_updates=train_iter)
            self.losses.extend(batch_losses)
            self.rewards.append(episode_reward)
            self.lengths.append(t)

            if self.wandb_run is not None:
                self.wandb_run.log({
                    "Reward": episode_reward,
                    "EpisodeLength": t,
                    "TouchRate": touched,
                    "WinRate": win
                })

            if i_episode % save_interval == 0:
                self._save_checkpoint(i_episode)
                self._save_statistics()

            if i_episode % log_interval == 0:
                avg_reward = np.mean(self.rewards[-log_interval:])
                avg_length = np.mean(self.lengths[-log_interval:])
                self.logger.info(
                    f"Episode {i_episode}\tAvg Length: {avg_length:.2f}\tAvg Reward: {avg_reward:.3f}"
                )

        self._save_statistics()
        self._plot_statistics()
        final_metrics = self._final_metrics()

        if self.wandb_run is not None:
            self.wandb_run.log({'average_reward': final_metrics['average_reward']})
            self.wandb_run.summary.update(final_metrics)

        return final_metrics
