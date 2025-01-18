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
from hockey.hockey_env import Mode


# Import your Dreamer agent
from models.dreamer.DREAM import DreamerV3Agent

class DreamerV3Trainer:
    def __init__(self, env_config, training_config, model_config, experiment_path, wandb_run=None):
        """
        env_config: dict with keys { 'name': "HockeyEnv", 'mode': "NORMAL", 'opponent_type': ... }
        training_config: dict with standard training hyperparams
        model_config: dict for the DreamerV3Agent
        experiment_path: path string where logs & checkpoints will be saved
        wandb_run: optional wandb run
        """
        self.env_config = env_config
        self.training_config = training_config
        self.model_config = model_config
        self.experiment_path = Path(experiment_path)
        self.wandb_run = wandb_run

        # ============ Environment setup ============
        self.logger = self._init_logger()
        self.mode_str = self.env_config.get("mode", "NORMAL")  # e.g. "NORMAL"
        self.opponent_type = self.env_config.get("opponent_type", "none")  # e.g. "basic_weak"
        self.env = self._create_env()

        # Initialize agent for the left player
        self.agent = DreamerV3Agent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            **self.model_config
        )

        # If you want self-play, create a second agent or reuse the same agent
        self.self_play = (self.opponent_type == "self_play")
        if self.self_play:
            # Option 1: same agent controlling both sides
            self.agent2 = self.agent
            # Option 2: different agent for the right side
            # self.agent2 = DreamerV3Agent(
            #    observation_space=self.env.observation_space,
            #    action_space=self.env.action_space,
            #    **self.model_config
            # )
        else:
            self.agent2 = None

        # If we use a basic opponent
        self.basic_opponent = None
        if "basic" in self.opponent_type:
            weak = True if "weak" in self.opponent_type else False
            self.basic_opponent = BasicOpponent(weak=weak)

        # Stats
        self.rewards = []
        self.lengths = []
        self.losses = []  # (wm_loss, actor_loss, critic_loss) tuples
        self.timestep = 0

        # Prepare directories
        self.results_path = self.experiment_path / "results"
        self.training_stats_path = self.results_path / "metrics"
        self.training_logs_path = self.experiment_path / "logs"
        self.training_plots_path = self.results_path / "plots"
        self.training_stats_path.mkdir(parents=True, exist_ok=True)
        self.training_logs_path.mkdir(parents=True, exist_ok=True)
        self.training_plots_path.mkdir(parents=True, exist_ok=True)

        # Seed
        seed = training_config.get("seed", 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env.reset(seed=seed)

    def _init_logger(self):
        logger = logging.getLogger("DreamerV3TrainerLogger")
        logger.setLevel(logging.INFO)

        # prevent multiple handlers
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
        return logger

    def _create_env(self):
        """
        Creates a HockeyEnv with the specified mode.
        e.g., mode = "NORMAL", "TRAIN_SHOOTING", or "TRAIN_DEFENSE".
        The environment already has the correct obs_space / action_space.
        """
        mode_map = {
            "NORMAL": Mode.NORMAL,
            "TRAIN_SHOOTING": Mode.TRAIN_SHOOTING,
            "TRAIN_DEFENSE": Mode.TRAIN_DEFENSE
        }

        # Wähle den Modus basierend auf mode_str aus, standardmäßig Mode.NORMAL
        selected_mode = mode_map.get(self.mode_str, Mode.NORMAL)

        # Erstelle das Environment mit dem ausgewählten Modus
        env = h_env.HockeyEnv(mode=selected_mode)
        return env

    def train(self):
        """
        Main training loop:
          - If self_play or basic_opponent, we get action2 from them
          - We store transitions for the left agent
          - We perform multiple training iterations after each episode
        """
        max_episodes = self.training_config["max_episodes"]
        max_timesteps = self.training_config["max_timesteps"]
        log_interval = self.training_config.get("log_interval", 20)
        save_interval = self.training_config.get("save_interval", 200)
        render = self.training_config.get("render", False)
        train_iter = self.training_config.get("train_iter", 32)

        self.logger.info(f"Starting DreamerV3 on HockeyEnv (mode={self.mode_str}, opponent={self.opponent_type})")
        self.logger.info(f"max_episodes={max_episodes}, max_timesteps={max_timesteps}")

        for ep in range(1, max_episodes + 1):
            obs, info = self.env.reset()
            self.agent.reset()
            if self.self_play and self.agent2 != self.agent:
                self.agent2.reset()

            episode_reward = 0.0

            for t in range(max_timesteps):
                self.timestep += 1

                # Left player's action (our agent)
                action1 = self.agent.act(obs)

                # Right player's action
                if self.self_play:
                    # Option 1: same agent controlling both sides
                    obs_agent2 = self.env.obs_agent_two()  # second agent's observation
                    action2 = self.agent2.act(obs_agent2)
                elif self.basic_opponent:
                    obs_agent2 = self.env.obs_agent_two()
                    action2 = self.basic_opponent.act(obs_agent2)
                else:
                    # e.g. random
                    action2 = np.random.uniform(-1, 1, size=self.env.action_space.shape[0])

                # Step environment
                joint_action = np.hstack([action1, action2])
                next_obs, reward, done, trunc, info = self.env.step(joint_action)
                episode_reward += reward

                # Store transition from the left player's perspective
                self.agent.store_transition((obs, action1, reward, next_obs, done))

                obs = next_obs

                if render:
                    self.env.render()

                if done or trunc:
                    break

            # After each episode, do multiple training updates
            batch_losses = self.agent.train(num_updates=train_iter)
            self.losses.extend(batch_losses)

            self.rewards.append(episode_reward)
            self.lengths.append(t)

            # wandb logging
            if self.wandb_run:
                self.wandb_run.log({"episode_reward": episode_reward, "episode_length": t})

            # save & log
            if ep % save_interval == 0:
                self._save_checkpoint(ep)
                self._save_statistics()

            if ep % log_interval == 0:
                avg_rew = np.mean(self.rewards[-log_interval:])
                avg_len = np.mean(self.lengths[-log_interval:])
                self.logger.info(f"Episode {ep}: avg_reward={avg_rew:.2f}, avg_length={avg_len:.1f}")

        # End of training
        self._save_statistics()
        self._plot_statistics()
        final_metrics = self._final_metrics()
        if self.wandb_run:
            self.wandb_run.log({"average_reward": final_metrics["average_reward"]})
            self.wandb_run.summary.update(final_metrics)
        return final_metrics

    def _save_statistics(self):
        stats_file = self.training_stats_path / "dreamer_stats.pkl"
        data = {
            "rewards": self.rewards,
            "lengths": self.lengths,
            "losses": self.losses
        }
        with open(stats_file, "wb") as f:
            pickle.dump(data, f)
        self.logger.info(f"Saved training stats to {stats_file}")

    def _save_checkpoint(self, ep):
        save_dir = self.experiment_path / "models" / "dreamer"
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / f"dreamer_checkpoint_ep{ep}.pth"
        torch.save(self.agent.state(), checkpoint_path)
        self.logger.info(f"Saved checkpoint at episode {ep} -> {checkpoint_path}")

    def _final_metrics(self):
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

        return {
            "average_reward": avg_reward,
            "average_length": avg_length,
            "final_world_model_loss": final_wm_loss,
            "final_actor_loss": final_actor_loss,
            "final_critic_loss": final_critic_loss
        }

    def _plot_statistics(self, window_size=50):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        def running_mean(x, n):
            cumsum = np.cumsum(np.insert(x, 0, 0))
            return (cumsum[n:] - cumsum[:-n]) / float(n)

        if len(self.rewards) < 1:
            return

        # Rewards
        episodes = np.arange(1, len(self.rewards) + 1)
        smoothed_rewards = running_mean(self.rewards, min(window_size, len(self.rewards)))
        smoothed_eps = np.arange(len(smoothed_rewards)) + window_size

        # Losses
        wm_losses = [l[0] for l in self.losses]
        actor_losses = [l[1] for l in self.losses]
        critic_losses = [l[2] for l in self.losses]

        plt.figure(figsize=(12, 10))

        # 1) Rewards
        plt.subplot(4, 1, 1)
        plt.plot(episodes, self.rewards, label="Rewards", alpha=0.4)
        if len(smoothed_eps) == len(smoothed_rewards):
            plt.plot(smoothed_eps, smoothed_rewards, label="Smoothed", linewidth=2)
        plt.ylabel("Reward")
        plt.xlabel("Episode")
        plt.title("Rewards")
        plt.grid(True)
        plt.legend()

        # 2) World Model Loss
        plt.subplot(4, 1, 2)
        plt.plot(wm_losses, label="WorldModel Loss")
        plt.ylabel("WM Loss")
        plt.grid(True)
        plt.legend()

        # 3) Actor Loss
        plt.subplot(4, 1, 3)
        plt.plot(actor_losses, label="Actor Loss", color='orange')
        plt.ylabel("Actor Loss")
        plt.grid(True)
        plt.legend()

        # 4) Critic Loss
        plt.subplot(4, 1, 4)
        plt.plot(critic_losses, label="Critic Loss", color='green')
        plt.ylabel("Critic Loss")
        plt.xlabel("Training Step")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        out_file = self.training_plots_path / "dreamer_training_plot.png"
        plt.savefig(out_file)
        self.logger.info(f"Saved training plot to {out_file}")
