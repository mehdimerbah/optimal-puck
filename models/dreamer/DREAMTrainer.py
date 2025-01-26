"""
DREAMTrainer.py

DreamerV3Trainer for HockeyEnv with multiple modes (NORMAL, TRAIN_SHOOTING, TRAIN_DEFENSE)
and opponents (basic_weak, basic_strong, none, self_play).
"""

import numpy as np
import torch
import gymnasium as gym
import logging
import sys
import pickle
import pylab as plt
from pathlib import Path

# If installed: import hockey
import hockey.hockey_env as h_env
from hockey.hockey_env import BasicOpponent

from hockey.hockey_env import HockeyEnv_BasicOpponent, Mode



# Import your Dreamer agent
from models.dreamer.DREAM import DreamerV3Agent


class DreamerV3Trainer:
    """
    A class for environment interaction, logging, checkpointing,
    saving statistics, and orchestrating the training loop for Dreamer.
    """

    def __init__(self, env_name, training_config, model_config, experiment_path, wandb_run=None):
        """
        env_name        : Name of the environment (e.g. "HockeyEnv" or "BipedalWalker-v3").
        training_config : Dictionary with training parameters (max_episodes, max_timesteps, etc.).
        model_config    : Dictionary for DreamerV3Agent hyperparameters.
        experiment_path : Path where logs & checkpoints will be saved.
        wandb_run       : optional wandb run for logging (if None, no wandb logging).
        """

        self.env_name = env_name

        # 1) Create environment
        if self.env_name == "HockeyEnv":
            # based config
            self.env = h_env.HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=False)
        else:
            self.env = gym.make(self.env_name)

        # 2) Save configs
        self.training_config = training_config
        self.model_config = model_config
        self.experiment_path = Path(experiment_path)
        self.wandb_run = wandb_run

        # 3) Initialize Dreamer agent
        self.agent = DreamerV3Agent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            **self.model_config
        )

        # 4) Stats trackers
        self.rewards = []
        self.lengths = []
        self.losses = []  # (wm_loss, actor_loss, critic_loss) tuples
        self.timestep = 0

        # 5) Prepare directories
        self.results_path = self.experiment_path / "results"
        self.training_stats_path = self.results_path / "training" / "stats"
        self.training_logs_path = self.results_path / "training" / "logs"
        self.training_plots_path = self.results_path / "training" / "plots"
        self.training_stats_path.mkdir(parents=True, exist_ok=True)
        self.training_logs_path.mkdir(parents=True, exist_ok=True)
        self.training_plots_path.mkdir(parents=True, exist_ok=True)

        # 6) Setup logger
        self.logger = self._initialize_logger()

        # 7) Set random seeds if specified
        self._initialize_seed()

    def _initialize_logger(self):
        """
        Configure a logger that logs to both stdout and a file in training/logs/dreamer.log
        """
        log_file = self.training_logs_path / "dreamer_training.log"
        logger = logging.getLogger("Dreamer_Trainer")
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

    def _initialize_seed(self):
        """
        Optionally set random seeds for reproducibility.
        """
        seed = self.training_config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        try:
            self.env.reset(seed=seed)
        except TypeError:
            # Some older gym versions or custom envs might not support `seed` in reset().
            pass
        self.logger.info(f"Initialized random seeds to {seed}.")

    def _save_checkpoint(self, episode):
        """
        Saves a checkpoint of the Dreamer agent's parameters.
        """
        saved_models_dir = self.results_path / "training" / "saved_models"
        saved_models_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = saved_models_dir / f"dreamer_checkpoint_ep{episode}.pth"

        torch.save(self.agent.state(), checkpoint_path)
        self.logger.info(f"Saved checkpoint at episode {episode} -> {checkpoint_path}")

    def _save_statistics(self):
        """
        Save training statistics (rewards, lengths, losses) as a pickle file.
        """
        stats_file = self.training_stats_path / "dreamer_training_stats.pkl"
        data = {
            "rewards": self.rewards,
            "lengths": self.lengths,
            "losses": self.losses,
        }
        with open(stats_file, "wb") as f:
            pickle.dump(data, f)
        self.logger.info(f"Saved training statistics to -> {stats_file}")

    def _plot_statistics(self, window_size=50):
        """
        Plot and save the training curves (rewards and losses).
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        def running_mean(x, N):
            cumsum = np.cumsum(np.insert(x, 0, 0))
            return (cumsum[N:] - cumsum[:-N]) / float(N)

        episodes = len(self.rewards)
        if episodes < 1:
            return

        # 1) Rewards
        smoothed_rewards = running_mean(self.rewards, min(window_size, episodes))
        smoothed_eps = np.arange(len(smoothed_rewards)) + window_size

        # 2) Losses
        # losses is a list of tuples: (wm_loss, actor_loss, critic_loss)
        wm_losses = [l[0] for l in self.losses]
        actor_losses = [l[1] for l in self.losses]
        critic_losses = [l[2] for l in self.losses]

        plt.figure(figsize=(14, 10))

        # -- Rewards Plot --
        plt.subplot(4, 1, 1)
        plt.plot(range(1, episodes + 1), self.rewards, label="Rewards", alpha=0.4)
        if len(smoothed_eps) == len(smoothed_rewards):
            plt.plot(smoothed_eps, smoothed_rewards, label="Smoothed", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Rewards")
        plt.legend()
        plt.grid(True)

        # -- World Model Loss Plot --
        plt.subplot(4, 1, 2)
        plt.plot(wm_losses, label="WorldModel Loss")
        plt.ylabel("WM Loss")
        plt.grid(True)
        plt.legend()

        # -- Actor Loss Plot --
        plt.subplot(4, 1, 3)
        plt.plot(actor_losses, label="Actor Loss", color='orange')
        plt.ylabel("Actor Loss")
        plt.grid(True)
        plt.legend()

        # -- Critic Loss Plot --
        plt.subplot(4, 1, 4)
        plt.plot(critic_losses, label="Critic Loss", color='green')
        plt.ylabel("Critic Loss")
        plt.xlabel("Training Step")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        plot_file = self.training_plots_path / "dreamer_training_plots.png"
        plt.savefig(plot_file)
        self.logger.info(f"Saved training plot to {plot_file}")

    def _final_metrics(self):
        """
        Return final metrics after training finishes.
        """
        if len(self.losses) > 0:
            final_wm_loss = self.losses[-1][0]
            final_actor_loss = self.losses[-1][1]
            final_critic_loss = self.losses[-1][2]
        else:
            final_wm_loss = None
            final_actor_loss = None
            final_critic_loss = None

        avg_reward = float(np.mean(self.rewards)) if len(self.rewards) > 0 else 0.0
        avg_length = float(np.mean(self.lengths)) if len(self.lengths) > 0 else 0.0

        metrics = {
            "average_reward": avg_reward,
            "average_length": avg_length,
            "final_world_model_loss": final_wm_loss,
            "final_actor_loss": final_actor_loss,
            "final_critic_loss": final_critic_loss
        }
        self.logger.info(f"Final training metrics: {metrics}")
        return metrics

    def train(self):
        """
        Main training loop:
          1) Interacts with the environment
          2) Collects transitions in replay buffer
          3) Calls agent.train() after each episode
          4) Periodically logs, saves stats, and checkpoints
        """
        max_episodes = self.training_config["max_episodes"]
        max_timesteps = self.training_config["max_timesteps"]
        log_interval = self.training_config.get("log_interval", 20)
        save_interval = self.training_config.get("save_interval", 500)
        render = self.training_config.get("render", False)
        train_iter = self.training_config.get("train_iter", 32)

        self.logger.info(f"Starting DreamerV3 Training on {self.env_name}")
        self.logger.info(f"max_episodes={max_episodes}, max_timesteps={max_timesteps}, train_iter={train_iter}")

        for i_episode in range(1, max_episodes + 1):
            obs, info = self.env.reset()
            self.agent.reset()
            episode_reward = 0.0

            # If you have special "touch" logic or environment-specific reward shaping, do it here.
            # Example:
            touched = 0
            first_time_touch = 1

            for t in range(max_timesteps):
                self.timestep += 1
                action = self.agent.act(obs)

                next_obs, reward, done, trunc, info = self.env.step(action)

                # Example: Custom reward shaping (like in the DDPG code).
                # This is optional and depends on your environment design:
                touched = max(touched, info.get('reward_touch_puck', 0))
                current_reward = reward + 5 * info.get('reward_closeness_to_puck', 0) - \
                                 (1 - touched) * 0.1 + touched * first_time_touch * 0.1 * t
                first_time_touch = 1 - touched

                # Store transition
                self.agent.store_transition((obs, action, current_reward, next_obs, done))

                obs = next_obs
                episode_reward += current_reward

                if render:
                    self.env.render()

                if done or trunc:
                    break

            # Training updates after the episode
            batch_losses = self.agent.train(num_updates=train_iter)
            self.losses.extend(batch_losses)

            self.rewards.append(episode_reward)
            self.lengths.append(t)

            # Optionally log to Weights & Biases
            if self.wandb_run is not None:
                self.wandb_run.log({
                    "EpisodeReward": episode_reward,
                    "EpisodeLength": t,
                    "TouchRate": touched,
                })

            # Save & log
            if i_episode % save_interval == 0:
                self._save_checkpoint(i_episode)
                self._save_statistics()

            if i_episode % log_interval == 0:
                avg_reward = np.mean(self.rewards[-log_interval:])
                avg_length = np.mean(self.lengths[-log_interval:])
                self.logger.info(f"Episode {i_episode}:\tAvg Reward={avg_reward:.2f}\tAvg Length={avg_length:.1f}")

        # End of training
        self._save_statistics()
        self._plot_statistics()

        final_metrics = self._final_metrics()
        if self.wandb_run is not None:
            self.wandb_run.log({"average_reward": final_metrics["average_reward"]})
            self.wandb_run.summary.update(final_metrics)

        return final_metrics
