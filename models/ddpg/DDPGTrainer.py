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
from hockey.hockey_env import HockeyEnv_BasicOpponent, Mode

from models.ddpg.DDPG import DDPGAgent

class DDPGTrainer:
    """
    A class for environment interaction, logging, checkpointing, 
    saving statistics, and orchestrating the training loop for DDPG.
    """
    def __init__(self, env_name, training_config, model_config, experiment_path, wandb_run=None):
        # Initialize the environement
        self.env_name = env_name
        if self.env_name == "HockeyEnv":
            self.env = HockeyEnv_BasicOpponent( mode=Mode.NORMAL,   # or Mode.TRAIN_SHOOTING, Mode.TRAIN_DEFENSE,
                                               weak_opponent=True)
        elif self.env_name == "BipedalWalker-v3":
            self.env = gym.make(env_name, hardcore=False, render_mode="rgb_array")
        else:
            self.env = gym.make(env_name)

        self.training_config = training_config
        # self.current_epsilon = training_config["epsilon_start"]
        # self.epsilon_min = training_config["epsilon_min"]
        # self.epsilon_decay = training_config["epsilon_decay"]
        self.model_config = model_config
        self.experiment_path = Path(experiment_path)
        self.agent = self._initialize_agent()
        self.wandb_run = wandb_run

        self.rewards = []
        self.lengths = []
        self.losses = []
        self.timestep = 0

        # Prepare necessary directories: results/training/stats, results/training/logs
        self.results_path = self.experiment_path / "results"
        self.training_stats_path = self.results_path / "training" / "stats"
        self.training_logs_path = self.results_path / "training" / "logs"
        self.training_plots_path = self.results_path / "training" / "plots"
        self.training_stats_path.mkdir(parents=True, exist_ok=True)
        self.training_logs_path.mkdir(parents=True, exist_ok=True)
        self.training_plots_path.mkdir(parents=True, exist_ok=True)
        

        # Initialize logger
        self.logger = self._initialize_logger()
        

        # Set random seeds if specified
        self._initialize_seed()

    def _initialize_agent(self):
        """
        Instantiate the DDPGAgent with the config from the 'model_config'.
        """
        agent = DDPGAgent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            **self.model_config
        )
        return agent

    def _initialize_seed(self):
        seed = self.training_config.get("seed", 42)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.env.reset(seed=seed)
            self.logger.info(f"Initialized random seeds to {seed}.")

    def _initialize_logger(self):
        """
        Configure a logger that logs to both stdout and a file 
        in experiments/experiment_id/training/logs/training.log
        """
        log_file = self.training_logs_path / f"DDPG_{self.env_name}_noise{self.model_config['noise_scale']}_alr{float(self.model_config['learning_rate_actor'])}_clr{float(self.model_config['learning_rate_critic'])}_gamma{self.model_config['discount']}.log"

        # Create a logger
        logger = logging.getLogger("DDPG_Trainer")
        logger.setLevel(logging.INFO)

        # Avoid re-adding handlers if this is called multiple times
        if not logger.handlers:
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            # Stream handler (stdout)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)

            # Format
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
        Saves a pickle file containing training statistics 
        in experiments/experiment_id/training/stats/
        """
        stats_file = self.training_stats_path / f"DDPG_{self.env_name}_noise{self.model_config['noise_scale']}_alr{float(self.model_config['learning_rate_actor'])}_clr{float(self.model_config['learning_rate_critic'])}_gamma{self.model_config['discount']}_stats.pkl"
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
        Saves a checkpoint of the agent’s parameters in
        experiments/experiment_id/training/saved_models/

        """
        saved_models_dir = self.results_path / "training" / "saved_models"
        saved_models_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = saved_models_dir / f"DDPG_{self.env_name}_noise{self.model_config['noise_scale']}_alr{float(self.model_config['learning_rate_actor'])}_clr{float(self.model_config['learning_rate_critic'])}_gamma{self.model_config['discount']}_checkpoint_ep{episode}.pth"

        torch.save(self.agent.state(), checkpoint_path)
        self.logger.info(f"Saved checkpoint at episode {episode} -> {checkpoint_path}")

    def _final_metrics(self):
        """
        Return final metrics after training finishes.
        """
        final_loss = self.losses[-1] if self.losses else (None, None)
        avg_reward = np.mean(self.rewards)
        avg_length = np.mean(self.lengths)
        metrics = {
            "average_reward": float(avg_reward),
            "average_length": float(avg_length),
            "final_loss_critic": float(final_loss[0]) if final_loss else None,
            "final_loss_actor": float(final_loss[1]) if final_loss else None
        }
        self.logger.info(f"Final training metrics: {metrics}")
        return metrics
    
    def _plot_statistics(self, window_size=50):
        """
        Plots rewards, episode lengths, and losses from the given statistics.
        
        Parameters:
        - stats: Dictionary containing keys 'rewards', 'lengths', 'losses', etc.
        - window_size: Window size for smoothing curves.
        """
        def running_mean(x, N):
            cumsum = np.cumsum(np.insert(x, 0, 0)) 
            return (cumsum[N:] - cumsum[:-N]) / float(N)   
        
        rewards = self.rewards
        lengths = self.lengths
        losses = self.losses 

        # Smoothed Rewards
        smoothed_rewards = running_mean(rewards, window_size)

        # Adjust x-axis for rewards and lengths
        episodes = np.arange(1, len(rewards) + 1)
        smoothed_episodes = np.arange(window_size, len(rewards) + 1)

        # Aggregate losses by episode (average over training steps per episode)
        losses_per_episode = np.mean(np.array(losses).reshape(len(rewards), -1, 2), axis=1)
        q_losses = losses_per_episode[:, 0]
        actor_losses = losses_per_episode[:, 1]
        smoothed_q_losses = running_mean(q_losses, window_size)
        smoothed_actor_losses = running_mean(actor_losses, window_size)

        # Plot Rewards
        plt.figure(figsize=(16, 12))
        plt.subplot(3, 1, 1)
        plt.plot(episodes, rewards, label="Raw Rewards", alpha=0.4)
        plt.plot(smoothed_episodes, smoothed_rewards, label=f"Smoothed Rewards (window={window_size})", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Reward vs Episodes (Noise Scale={self.model_config['noise_scale']})")
        plt.legend()
        plt.grid()

        # Plot Q-Loss
        plt.subplot(3, 1, 2)
        plt.plot(episodes, q_losses, label="Raw Q-Loss", alpha=0.4)
        plt.plot(smoothed_episodes, smoothed_q_losses, label=f"Smoothed Q-Loss (window={window_size})", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Q-Loss")
        plt.title("Q-Loss vs Episodes")
        plt.legend()
        plt.grid()

        # Plot Actor Loss
        plt.subplot(3, 1, 3)
        plt.plot(episodes, actor_losses, label="Raw Actor Loss", alpha=0.4)
        plt.plot(smoothed_episodes, smoothed_actor_losses, label=f"Smoothed Actor Loss (window={window_size})", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Actor Loss")
        plt.title("Actor Loss vs Episodes")
        plt.legend()
        plt.grid()

        # Adjust layout and show the plots
        plt.tight_layout()
        plt.savefig(self.training_plots_path / f"DDPG_{self.env_name}_noise{self.model_config['noise_scale']}_alr{float(self.model_config['learning_rate_actor'])}_clr{float(self.model_config['learning_rate_critic'])}_gamma{self.model_config['discount']}_training_plot.png")


    def train(self):
            """
            Main training loop that interacts with the environment, collects transitions,
            and calls the agent's train method.
            """
            max_episodes   = self.training_config["max_episodes"]
            max_timesteps  = self.training_config["max_timesteps"]
            log_interval   = self.training_config.get("log_interval", 20)
            save_interval  = self.training_config.get("save_interval", 500)
            render         = self.training_config.get("render", False)
            train_iter     = self.training_config.get("train_iter", 32)

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
                    touched = max(touched, _info['reward_touch_puck'])
                    current_reward = reward + 2 * _info['reward_closeness_to_puck'] - (
                        1 - touched) * 0.1 + touched * first_time_touch * 0.01 * t


                    first_time_touch = 1 - touched

                    self.agent.store_transition((obs, action, current_reward, next_obs, done))
                    obs = next_obs
                   
                    episode_reward += current_reward

                    if render:
                        self.env.render()

                    if done or trunc:
                        break
                        
                if _info['winner'] == 1:
                    win = 1

                # Perform training updates
                batch_losses = self.agent.train(num_updates=train_iter)
                self.losses.extend(batch_losses)

                self.rewards.append(episode_reward)
                self.lengths.append(t)

                # Log to wandb
                if self.wandb_run is not None:
                    self.wandb_run.log({
                        "Reward": episode_reward,
                        "EpisodeLength": t,
                        "TouchRate": touched,
                        "WinRate": win
                    })

                # Save checkpoint & stats periodically
                if i_episode % save_interval == 0:
                    self._save_checkpoint(i_episode)
                    self._save_statistics()

                # Print training progress
                if i_episode % log_interval == 0:
                    avg_reward = np.mean(self.rewards[-log_interval:])
                    avg_length = np.mean(self.lengths[-log_interval:])
                    self.logger.info(
                        f"Episode {i_episode}\tAvg Length: {avg_length:.2f}\tAvg Reward: {avg_reward:.3f}"
                    )

                # self.current_epsilon = max(self.epsilon_min, self.current_epsilon * self.epsilon_decay)

            # Final stats saved and plotted
            self._save_statistics()
            self._plot_statistics()

            final_metrics = self._final_metrics()

            if self.wandb_run is not None:
                self.wandb_run.log({'average_reward': final_metrics['average_reward']})
                self.wandb_run.summary.update(final_metrics)
                

            return final_metrics