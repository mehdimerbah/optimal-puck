import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym
import pickle
import logging
import sys
from pathlib import Path
from datetime import datetime
from .TD3 import TD3Agent

class TD3Trainer:
    def __init__(self, env_name, training_config, model_config, experiment_path, wandb_run=None):
        self.env_name = env_name
        self.training_config = training_config
        self.model_config = model_config
        self.experiment_path = Path(experiment_path)
        self.wandb_run = wandb_run
        
        # Create environment
        self.env = gym.make(env_name, continuous=True)
        self.eval_env = gym.make(env_name, continuous=True)
        
        # Set random seeds
        random_seed = model_config['training']['random_seed']
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.env.reset(seed=random_seed)
        self.eval_env.reset(seed=random_seed)
        
        # Initialize agent
        self.agent = TD3Agent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            **model_config
        )
        
        # Initialize metrics tracking
        self.rewards = []
        self.lengths = []
        self.losses = []
        self.timestep = 0
        
        # Create directories for saving
        self.results_path = self.experiment_path / "results"
        self.training_stats_path = self.results_path / "training" / "stats"
        self.training_logs_path = self.results_path / "training" / "logs"
        self.training_plots_path = self.results_path / "training" / "plots"
        
        for path in [self.training_stats_path, self.training_logs_path, self.training_plots_path]:
            path.mkdir(parents=True, exist_ok=True)
            
        # Initialize logger
        self.logger = self._initialize_logger()
        
        # Store max steps from model config
        self.max_steps = model_config.get('max_steps', 2000)  # Default to 2000 if not specified

    def evaluate_policy(self, eval_episodes=10):
        avg_reward = 0.
        for _ in range(eval_episodes):
            state, _ = self.eval_env.reset()
            done = False
            truncated = False
            while not (done or truncated):
                action = self.agent.act(state, eps=0)  # No exploration during evaluation
                state, reward, done, truncated, _ = self.eval_env.step(action)
                avg_reward += reward
        avg_reward /= eval_episodes
        return avg_reward

    def save_checkpoint(self, episode, metrics):
        checkpoint = {
            'model_state': self.agent.state(),
            'metrics': metrics,
            'episode': episode,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, self.model_dir / f'checkpoint_episode_{episode}.pt')

    def train(self):
        """
        Main training loop that interacts with the environment, collects transitions,
        and trains the agent.
        """
        max_episodes = self.training_config["max_episodes"]
        max_timesteps = self.training_config.get("max_timesteps", self.max_steps)
        log_interval = self.training_config.get("log_interval", 20)
        save_interval = self.training_config.get("save_interval", 500)
        train_iter = self.training_config.get("train_iter", 32)

        self.logger.info("Starting TD3 Training...")
        self.logger.info(f"Environment: {self.env_name}, max_episodes={max_episodes}, "
                        f"max_timesteps={max_timesteps}, train_iter={train_iter}")

        best_eval_reward = float('-inf')

        for i_episode in range(1, max_episodes + 1):
            state, _ = self.env.reset()
            self.agent.noise.reset()
            episode_reward = 0
            episode_losses = []

            for t in range(max_timesteps):
                self.timestep += 1
                
                # Select action with exploration noise during training
                action = self.agent.act(state, eps=self.model_config.get('eps', 0.1))
                
                # Execute action
                next_state, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward
                
                # Store transition
                self.agent.store_transition((state, action, reward, next_state, done))
                
                # Train agent if enough samples
                if self.agent.buffer.size >= self.model_config['batch_size']:
                    losses = self.agent.train(iter_fit=train_iter)
                    episode_losses.extend(losses)
                
                if done or truncated:
                    self.logger.info(f"Episode {i_episode} finished after {t+1} steps with reward {episode_reward:.2f}")
                    break
                    
                state = next_state

            # Log training progress
            self.rewards.append(episode_reward)
            self.lengths.append(t)
            self.losses.extend(episode_losses)

            # Log to wandb if available
            if self.wandb_run is not None:
                self.wandb_run.log({
                    "reward": episode_reward,
                    "length": t,
                    "q_loss": np.mean([l[0] for l in episode_losses]) if episode_losses else None,
                    "actor_loss": np.mean([l[1] for l in episode_losses]) if episode_losses else None
                })
            
            # Print training progress every log_interval episodes
            if i_episode % log_interval == 0:
                avg_reward = np.mean(self.rewards[-log_interval:])
                avg_length = np.mean(self.lengths[-log_interval:])
                avg_q_loss = np.mean([l[0] for l in episode_losses]) if episode_losses else 0
                avg_actor_loss = np.mean([l[1] for l in episode_losses]) if episode_losses else 0
                
                self.logger.info(
                    f"Episode {i_episode}\tAvg Length: {avg_length:.2f}\tAvg Reward: {avg_reward:.3f}\t"
                    f"Q-Loss: {avg_q_loss:.4f}\tActor-Loss: {avg_actor_loss:.4f}"
                )
                
                # Evaluate policy without exploration noise
                eval_reward = self.evaluate_policy()
                self.logger.info(f"Evaluation reward: {eval_reward:.2f}")
                
                # Save if best
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.logger.info(f"New best evaluation reward: {best_eval_reward:.2f}")
                    self._save_checkpoint(i_episode)
            
            # Save checkpoint and statistics periodically
            if i_episode % save_interval == 0:
                self._save_checkpoint(i_episode)
                self._save_statistics()
                self._plot_statistics()

        # Final saves and cleanup
        self._save_checkpoint(max_episodes)
        self._save_statistics()
        self._plot_statistics()
        
        # Close environments
        self.env.close()
        self.eval_env.close()
        
        final_metrics = self._final_metrics()
        
        if self.wandb_run is not None:
            self.wandb_run.log({'average_reward': final_metrics['average_reward']})
            self.wandb_run.summary.update(final_metrics)
        
        return final_metrics

    def _initialize_logger(self):
        """
        Configure a logger that logs to both stdout and a file.
        """
        log_file = self.training_logs_path / f"TD3_{self.env_name}_eps{self.model_config['eps']}_alr{float(self.model_config['learning_rate_actor'])}_clr{float(self.model_config['learning_rate_critic'])}_gamma{self.model_config['discount']}.log"

        # Create a logger
        logger = logging.getLogger("TD3_Trainer")
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
        Saves a pickle file containing training statistics.
        """
        stats_file = self.training_stats_path / f"TD3_{self.env_name}_eps{self.model_config['eps']}_alr{float(self.model_config['learning_rate_actor'])}_clr{float(self.model_config['learning_rate_critic'])}_gamma{self.model_config['discount']}_stats.pkl"
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
        Saves a checkpoint of the agent's parameters.
        """
        saved_models_dir = self.results_path / "training" / "saved_models"
        saved_models_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = saved_models_dir / f"TD3_{self.env_name}_eps{self.model_config['eps']}_alr{float(self.model_config['learning_rate_actor'])}_clr{float(self.model_config['learning_rate_critic'])}_gamma{self.model_config['discount']}_checkpoint_ep{episode}.pth"

        torch.save(self.agent.state(), checkpoint_path)
        self.logger.info(f"Saved checkpoint at episode {episode} -> {checkpoint_path}")

    def _final_metrics(self):
        """
        Return final metrics after training finishes.
        """
        if len(self.losses) > 0:
            final_q_loss = self.losses[-1][0]
            final_actor_loss = self.losses[-1][1]
        else:
            final_q_loss = None
            final_actor_loss = None

        avg_reward = float(np.mean(self.rewards)) if len(self.rewards) > 0 else 0.0
        avg_length = float(np.mean(self.lengths)) if len(self.lengths) > 0 else 0.0

        metrics = {
            "average_reward": avg_reward,
            "average_length": avg_length,
            "final_q_loss": final_q_loss,
            "final_actor_loss": final_actor_loss
        }
        self.logger.info(f"Final training metrics: {metrics}")
        return metrics

    def _plot_statistics(self, window_size=50):
        """
        Plots rewards, episode lengths, and losses from the training statistics.
        
        Parameters:
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
        plt.title(f"Reward vs Episodes (Epsilon={self.model_config['eps']})")
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

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.training_plots_path / f"TD3_{self.env_name}_eps{self.model_config['eps']}_alr{float(self.model_config['learning_rate_actor'])}_clr{float(self.model_config['learning_rate_critic'])}_gamma{self.model_config['discount']}_training_plot.png")
        plt.close()
