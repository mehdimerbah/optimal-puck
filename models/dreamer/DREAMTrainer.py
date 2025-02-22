"""
Improved DreamerV3Trainer for HockeyEnv with multiple modes (NORMAL, TRAIN_SHOOTING, TRAIN_DEFENSE)
and opponents (basic_weak, basic_strong, none, self_play).

This version improves reward shaping, logging, checkpointing, and overall stability.
"""

import numpy as np
import torch
import gymnasium as gym
import logging
import sys
import pickle
from pathlib import Path

from types import SimpleNamespace

# If installed: import hockey
import hockey.hockey_env as h_env
from hockey.hockey_env import BasicOpponent, Mode, HockeyEnv_BasicOpponent

# Import your Dreamer agent
from models.dreamer.DREAM import DreamerV3Agent


class DreamerV3Trainer:
    """
    Trainer class for DreamerV3. It manages the training loop, logging, checkpointing, and statistics.
    """
    def __init__(self, env_name, training_config, model_config, experiment_path, wandb_run=None):
        self.env_name = env_name

        # Initialize environment
        if self.env_name == "HockeyEnv":
            self.env = HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=True)
        else:
            self.env = gym.make(self.env_name)

        # Save configurations
        self.training_config = training_config
        #self.model_config = model_config
        self.model_config = SimpleNamespace(**model_config)
        self.experiment_path = Path(experiment_path)
        self.wandb_run = wandb_run

        # Initialize Agent
        self.agent = DreamerV3Agent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            **vars(self.model_config)
        )

        # Statistics
        self.rewards = []
        self.lengths = []
        self.losses = []
        self.timestep = 0

        # Create directories for saving results
        self.results_path = self.experiment_path / "results"
        self.training_stats_path = self.results_path / "training" / "stats"
        self.training_logs_path  = self.results_path / "training" / "logs"
        self.training_plots_path = self.results_path / "training" / "plots"
        self.training_stats_path.mkdir(parents=True, exist_ok=True)
        self.training_logs_path.mkdir(parents=True, exist_ok=True)
        self.training_plots_path.mkdir(parents=True, exist_ok=True)

        # Initialize logger and random seed
        self.logger = self._initialize_logger()
        self._initialize_seed()

    def _initialize_logger(self):
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

        logger.info(f"Logger initialized. Logs will be written to {log_file}")
        return logger

    def _initialize_seed(self):
        seed = self.training_config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        try:
            self.env.reset(seed=seed)
        except TypeError:
            pass
        self.logger.info(f"Random seed initialized to {seed}")

    def _save_checkpoint(self, episode):
        saved_models_dir = self.results_path / "training" / "saved_models"
        saved_models_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = saved_models_dir / f"dreamer_checkpoint_ep{episode}.pth"
        torch.save(self.agent.state(), checkpoint_path)
        self.logger.info(f"Checkpoint saved at episode {episode}: {checkpoint_path}")

    def _save_statistics(self):
        stats_file = self.training_stats_path / "dreamer_training_stats.pkl"
        data = {
            "rewards": self.rewards,
            "lengths": self.lengths,
            "losses": self.losses,
        }
        with open(stats_file, "wb") as f:
            pickle.dump(data, f)
        self.logger.info(f"Training statistics saved to {stats_file}")

    def _plot_statistics(self, window_size=50):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        def running_mean(x, N):
            cumsum = np.cumsum(np.insert(x, 0, 0))
            return (cumsum[N:] - cumsum[:-N]) / float(N)

        episodes = len(self.rewards)
        if episodes < 1:
            return

        smoothed_rewards = running_mean(self.rewards, min(window_size, episodes))
        smoothed_eps = np.arange(len(smoothed_rewards)) + window_size

        # Extract loss components if available
        if len(self.losses) > 0:
            wm_losses    = [l.get("wm_loss", 0.0) for l in self.losses]
            ens_losses   = [l.get("ens_loss", 0.0) for l in self.losses]
            actor_losses = [l.get("actor_loss", 0.0) for l in self.losses]
            critic_losses= [l.get("critic_loss", 0.0) for l in self.losses]
        else:
            wm_losses = ens_losses = actor_losses = critic_losses = []

        plt.figure(figsize=(14, 10))

        # Plot Episode Rewards
        plt.subplot(5, 1, 1)
        plt.plot(range(1, episodes+1), self.rewards, label="Reward", alpha=0.4)
        if len(smoothed_rewards) == len(smoothed_eps):
            plt.plot(smoothed_eps, smoothed_rewards, label="Smoothed", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Rewards")
        plt.legend()
        plt.grid(True)

        # World Model Loss
        plt.subplot(5, 1, 2)
        plt.plot(wm_losses, label="WM Loss")
        plt.ylabel("WM Loss")
        plt.grid(True)
        plt.legend()

        # Ensemble Loss
        plt.subplot(5, 1, 3)
        plt.plot(ens_losses, label="Ensemble Loss", color='magenta')
        plt.ylabel("Ensemble Loss")
        plt.grid(True)
        plt.legend()

        # Actor Loss
        plt.subplot(5, 1, 4)
        plt.plot(actor_losses, label="Actor Loss", color='orange')
        plt.ylabel("Actor Loss")
        plt.grid(True)
        plt.legend()

        # Critic Loss
        plt.subplot(5, 1, 5)
        plt.plot(critic_losses, label="Critic Loss", color='green')
        plt.xlabel("Training Step")
        plt.ylabel("Critic Loss")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plot_file = self.training_plots_path / "dreamer_training_plots.png"
        plt.savefig(plot_file)
        self.logger.info(f"Training plots saved to {plot_file}")

    def _final_metrics(self):
        if len(self.losses) > 0:
            final_wm_loss = self.losses[-1].get("wm_loss", None)
            final_actor_loss = self.losses[-1].get("actor_loss", None)
            final_critic_loss = self.losses[-1].get("critic_loss", None)
        else:
            final_wm_loss = final_actor_loss = final_critic_loss = None

        avg_reward = float(np.mean(self.rewards)) if self.rewards else 0.0
        avg_length = float(np.mean(self.lengths)) if self.lengths else 0.0
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
        max_episodes = self.training_config["max_episodes"]
        max_timesteps = self.training_config["max_timesteps"]
        log_interval = self.training_config.get("log_interval", 20)
        save_interval = self.training_config.get("save_interval", 500)
        render = self.training_config.get("render", False)
        train_iter = self.training_config.get("train_iter", 32)

        self.logger.info(f"Starting DreamerV3 training on {self.env_name}")
        self.logger.info(f"max_episodes={max_episodes}, max_timesteps={max_timesteps}, train_iter={train_iter}")

        # Main training loop over episodes
        for i_episode in range(1, max_episodes + 1):
            obs, info = self.env.reset()
            obs = np.array(obs)
            self.agent.reset()
            ep_reward = 0.0

            # For hockey-specific reward shaping
            touched = False
            first_touch = True  # flag to mark the first time the puck is touched

            for t in range(max_timesteps):
                self.timestep += 1

                # Get action from agent
                action = self.agent.act(obs, sample=True)
                next_obs, reward, done, trunc, info = self.env.step(action)
                next_obs = np.array(next_obs)

                # Update touch flag using environment info
                current_touch = info.get('reward_touch_puck', 0)
                if current_touch:
                    touched = True

                # Improved reward shaping:
                # - Add bonus only at the first puck touch
                # - Use closeness reward with a scaling factor
                bonus = 0.1 if (current_touch and first_touch) else 0.0
                if current_touch and first_touch:
                    first_touch = False



                # Here we remove the small negative penalty which might destabilize learning.
                shaped_reward = reward + 5.0 * info.get('reward_closeness_to_puck', 0) + bonus


                # Create transition dictionary with ensured correct types
                transition = {
                    "obs": np.array(obs),
                    "action": np.array(action),
                    "reward": np.array(shaped_reward, dtype=np.float32),
                    "next_obs": np.array(next_obs),
                    "done": np.array(float(done or trunc))
                }
                # Store the transition in agent's memory (priority now comes from Plan2Explore)
                self.agent.store_transition(transition)

                obs = next_obs
                ep_reward += shaped_reward

                if render:
                    self.env.render()
                if done or trunc:
                    break

            # Perform training iterations after each episode
            batch_losses = self.agent.train(num_updates=train_iter)
            self.losses.extend(batch_losses)
            self.rewards.append(ep_reward)
            self.lengths.append(t)


            # Log metrics to wandb if provided
            if self.wandb_run is not None:
                self.wandb_run.log({
                    "EpisodeReward": ep_reward,
                    "EpisodeLength": t,
                    "Touched": int(touched)
                })

            # Periodically save checkpoints and statistics
            if i_episode % save_interval == 0:
                self._save_checkpoint(i_episode)
                self._save_statistics()

            if i_episode % log_interval == 0:
                avg_r = np.mean(self.rewards[-log_interval:])
                avg_len = np.mean(self.lengths[-log_interval:])
                self.logger.info(f"Episode {i_episode}: Avg Reward = {avg_r:.2f}, Avg Length = {avg_len:.1f}")

        # Save final statistics and plots
        self._save_statistics()
        self._plot_statistics()
        final_metrics = self._final_metrics()

        if self.wandb_run:
            self.wandb_run.log({"average_reward": final_metrics["average_reward"]})
            self.wandb_run.summary.update(final_metrics)

        return final_metrics




