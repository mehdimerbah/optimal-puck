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
import imageio
import csv

from hockey.hockey_env import (HockeyEnv_BasicOpponent, Mode, CENTER_X, CENTER_Y, VIEWPORT_W, VIEWPORT_H, SCALE)
from .TD3 import TD3Agent

class OpponentPool:
    """
    Handles a range of opponent strategies and adjusts their selection probabilities as training advances,
    based on performance metrics and predefined training phases.
    """

    def __init__(self, env_creator, experiment_path, file_prefix):
        """
        Set up the opponent pool with default opponent instances, tracking metrics, and initial sampling weights.
        
        Args:
            env_creator: Function to create environment instances
            experiment_path (Path): Directory where phase state files will be saved
            file_prefix (str): Prefix to identify trial-specific state files
        """
        self.env_creator = env_creator

        # Initialize opponent instances
        self.opponent_instances = {
            'random': None,
            'weak': HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=True),
            'strong': HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=False)
        }

        # Set default sampling weights for each opponent type
        self.weights = {
            'random': 1.0,
            'weak': 0.0,
            'strong': 0.0,
            'self': 0.0
        }
        self.current_phase = 'phase1'

        # Initialize metrics for tracking win rates and game counts per opponent type
        self.opponent_metrics = {
            'random': {'wins': 0, 'games': 0, 'history': []},
            'weak': {'wins': 0, 'games': 0, 'history': []},
            'strong': {'wins': 0, 'games': 0, 'history': []},
            'self': {'wins': 0, 'games': 0, 'history': []}
        }

        # Determine the file to persist the current phase state
        self.phase_state_file = experiment_path / f"phase_state_{file_prefix}.pkl"

        # Load existing phase state if available; otherwise, initialize defaults
        if self.phase_state_file.exists():
            with open(self.phase_state_file, "rb") as f:
                state = pickle.load(f)
                self.current_phase = state.get("current_phase", "phase1")
                self.weights = state.get("weights", {
                    'random': 1.0,
                    'weak': 0.0,
                    'strong': 0.0,
                    'self': 0.0
                })
                self.cumulative_episode_count = state.get("cumulative_episode_count", 0)
        else:
            self.current_phase = 'phase1'
            self.weights = {
                'random': 1.0,
                'weak': 0.0,
                'strong': 0.0,
                'self': 0.0
            }
            self.cumulative_episode_count = 0

    def sample_opponent(self):
        """
        Choose an opponent based on the current sampling probabilities.

        Returns:
            tuple: (opponent, opponent_type) where opponent is an instance or 'self' for self-play
        """
        opponent_types = list(self.weights.keys())
        weight_values = np.array([self.weights[t] for t in opponent_types])
        normalized_weights = weight_values / weight_values.sum()
        opponent_type = np.random.choice(opponent_types, p=normalized_weights)

        # Handle self-play opponent type separately
        if opponent_type == 'self':
            return 'self', 'self'
        return self.opponent_instances[opponent_type], opponent_type

    def update_metrics(self, opponent_type, won):
        """
        Record the outcome (win or loss) for a specified opponent type.

        Args:
            opponent_type (str): Type of opponent (e.g., 'random', 'weak', 'strong', 'self')
            won (bool): True if the agent won, False otherwise
        """
        if opponent_type in self.opponent_metrics:
            self.opponent_metrics[opponent_type]['games'] += 1
            if won:
                self.opponent_metrics[opponent_type]['wins'] += 1

    def get_opponent_win_rate(self, opponent_type, window_size=100):
        """
        Compute the win rate for a given opponent type over the most recent episodes.

        Args:
            opponent_type (str): The opponent type to evaluate
            window_size (int, optional): Number of recent episodes to consider

        Returns:
            float: Win rate over the window, or 0.0 if no history is available
        """
        if opponent_type in self.opponent_metrics:
            history = self.opponent_metrics[opponent_type].get('history', [])
            if history:
                recent_results = history[-window_size:]
                return sum(recent_results) / len(recent_results)
        return 0.0

    def update_weights(self, metrics):
        """
        Update the sampling probabilities for opponents and switch training phases based on performance.

        Args:
            metrics (dict): Contains 'win_rate', 'reward_std', and 'episode_count'
        """
        win_rate = metrics.get('win_rate', 0)
        reward_std = metrics.get('reward_std', 0)
        episode_count = metrics.get('episode_count', 0)

        # Define minimum episode count required for each phase
        min_phase_duration = {
            'phase1': 200,
            'phase2': 300,
            'phase3': 400,
            'phase4': 500
        }

        # Phase 1: Only random opponents are used.
        if self.current_phase == 'phase1':
            if episode_count >= min_phase_duration['phase1'] and win_rate > 0.7 and reward_std < 15:
                self.current_phase = 'phase2'
                self.weights = {
                    'random': 0.2,
                    'weak': 0.55,
                    'strong': 0.0,
                    'self': 0.15
                }

        # Phase 2: Focus on weak opponents with dynamic adjustments.
        elif self.current_phase == 'phase2':
            weak_win_rate = self.get_opponent_win_rate('weak')
            if episode_count >= min_phase_duration['phase2'] and weak_win_rate > 0.75 and reward_std < 14:
                self.current_phase = 'phase3'
                self.weights = {
                    'random': 0.1,
                    'weak': 0.4,
                    'strong': 0.3,
                    'self': 0.2
                }
            else:
                # Adjust weights based on trends in weak opponent win rate
                if weak_win_rate > 0.6:
                    self.weights['weak'] = min(0.8, self.weights['weak'] + 0.05)
                    self.weights['random'] = max(0.1, self.weights['random'] - 0.05)
                elif weak_win_rate < 0.45:
                    self.weights['weak'] = max(0.4, self.weights['weak'] - 0.05)
                    self.weights['random'] = min(0.5, self.weights['random'] + 0.05)

        # Phase 3: Mix of strong opponents and self-play.
        elif self.current_phase == 'phase3':
            strong_win_rate = self.get_opponent_win_rate('strong')
            if episode_count >= min_phase_duration['phase3'] and strong_win_rate > 0.65 and reward_std < 13:
                self.current_phase = 'phase4'
                self.weights = {
                    'random': 0.0,
                    'weak': 0.1,
                    'strong': 0.3,
                    'self': 0.6
                }
            else:
                # Adjust weights based on trends in strong opponent win rate
                if strong_win_rate > 0.7:
                    self.weights['strong'] = min(0.6, self.weights['strong'] + 0.05)
                    self.weights['weak'] = max(0.05, self.weights['weak'] - 0.05)
                elif strong_win_rate < 0.5:
                    self.weights['strong'] = max(0.2, self.weights['strong'] - 0.05)
                    self.weights['weak'] = min(0.3, self.weights['weak'] + 0.05)

        # Phase 4: Predominantly self-play with some strong opponents.
        elif self.current_phase == 'phase4':
            self_win_rate = self.get_opponent_win_rate('self')
            if self_win_rate > 0.70:
                self.weights['self'] = min(0.7, self.weights['self'] + 0.05)
                self.weights['strong'] = max(0.2, self.weights['strong'] - 0.05)
            elif self_win_rate < 0.50:
                self.weights['weak'] = min(0.3, self.weights['weak'] + 0.05)
                self.weights['strong'] = max(0.3, self.weights['strong'] + 0.05)

    def save_phase_state(self, cumulative_episode_count):
        """
        Save the current training phase, weights, and total episode count to disk for resuming training later.

        Args:
            cumulative_episode_count (int): Total completed episodes
        """
        state = {
            "current_phase": self.current_phase,
            "weights": self.weights,
            "cumulative_episode_count": cumulative_episode_count
        }
        with open(self.phase_state_file, "wb") as f:
            pickle.dump(state, f)

class TD3Trainer:
    """
    Trainer for the TD3 agent.

    This class manages the training process including logging, checkpointing,
    evaluation, and visualization. It supports both standard environments and a specialized HockeyEnv.
    """

    def __init__(self, env_name, training_config, model_config, experiment_path, wandb_run=None, agent_role="player1"):
        """
        Set up the TD3 trainer with the given environment and configuration.

        Args:
            env_name (str): Name of the environment
            training_config (dict): Training configuration parameters
            model_config (dict): TD3 model settings
            experiment_path (str or Path): Directory for experiment outputs
            wandb_run: (Optional) Weights & Biases run instance
            agent_role (str, optional): Agent identifier
        """
        self.env_name = env_name
        self.training_config = training_config
        self.model_config = model_config
        self.experiment_path = Path(experiment_path)
        self.wandb_run = wandb_run
        self.agent_role = agent_role

        # Create directories for saving results, logs, and plots
        self.results_path = self.experiment_path / "results"
        self.training_stats_path = self.experiment_path / "results/training/stats"
        self.training_logs_path = self.experiment_path / "results/training/logs"
        self.training_plots_path = self.experiment_path / "results/training/plots"
        self.evaluation_gifs_path = self.experiment_path / "results/evaluation/gifs"

        for path in [self.training_stats_path, self.training_logs_path, self.training_plots_path, self.evaluation_gifs_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.logger = self._initialize_logger()

        # Create environment (handle HockeyEnv separately)
        if self.env_name == "HockeyEnv":
            self.env_creator = lambda: HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=False)
            self.env = self.env_creator()
            self.eval_env = self.env_creator()
        else:
            self.env = gym.make(env_name)
            self.eval_env = gym.make(env_name)

        # Initialize random seeds for reproducibility
        self._initialize_seed()

        # Initialize the TD3 agent using the model configuration
        self.agent = self._initialize_agent()

        # Metrics tracking for training performance
        self.rewards = []
        self.lengths = []
        self.losses = []
        self.timestep = 0

        # Win history for performance tracking against opponents
        self.win_history = []

        # Initialize opponent pool using a file prefix based on hyperparameters
        file_prefix = self._get_file_prefix()
        self.opponent_pool = OpponentPool(self.env_creator, self.experiment_path, file_prefix)
        self.cumulative_episode_count = self.opponent_pool.cumulative_episode_count

        # Additional performance tracking configuration
        self.performance_window = 100
        self.current_opponent = None
        self.current_opponent_type = None

        # Self-play checkpoint buffer
        self.self_play_checkpoints = []
        self.max_checkpoints = 5

    def _initialize_seed(self):
        """
        Seed random number generators for PyTorch, NumPy, and the environment to ensure reproducibility.

        Uses the seed provided in the training configuration.
        """
        seed = self.training_config.get("seed", 42)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.env.reset(seed=seed)
            self.logger.info(f"Initialized random seeds to {seed}.")

    def _initialize_agent(self):
        """
        Create and return a TD3Agent based on the current model settings.

        Returns:
            TD3Agent: The created TD3 agent
        """
        return TD3Agent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            **self.model_config
        )

    def _get_file_prefix(self):
        """
        Build a file prefix that encodes key hyperparameters.

        Returns:
            str: The constructed prefix
        """
        prefix = (
            f"TD3_{self.env_name}"
            f"_alr{self.model_config['learning_rate_actor']:.4g}"
            f"_clr{self.model_config['learning_rate_critic']:.4g}"
            f"_gamma{self.model_config['discount']}"
            f"_policyNoise{self.model_config['policy_noise']}"
            f"_noiseClip{self.model_config['noise_clip']}"
            f"_policyDelay{self.model_config['policy_delay']}"
            f"_polyak{self.model_config['polyak']}"
            f"_warmupSteps{self.model_config['warmup_steps']}"
        )
        # Append reward shaping parameters if available
        rs = self.model_config.get("reward_shaping", {})
        aim_char = str(rs.get("aim_multiplier", 0))[0]
        defend_char = str(rs.get("defend_multiplier", 0))[0]
        block_char = str(rs.get("block_multiplier", 0))[0]
        touch_char = str(rs.get("touch_multiplier", 0))[0]
        closeness_char = str(rs.get("closeness_multiplier", 0))[0]
        wall_char = str(rs.get("wall_multiplier", 0))[0]
        rs_str = (
            f"_a{aim_char}"
            f"_d{defend_char}"
            f"_b{block_char}"
            f"_t{touch_char}"
            f"_c{closeness_char}"
            f"_w{wall_char}"
        )
        return prefix + rs_str

    def _initialize_logger(self):
        """
        Set up a logger that writes messages to both a file and the console.

        Returns:
            logging.Logger: The configured logger
        """
        log_file = self.training_logs_path / f"{self._get_file_prefix()}.log"

        logger = logging.getLogger("TD3_Trainer")
        logger.setLevel(logging.INFO)

        # Prevent adding duplicate handlers
        if not logger.handlers:
            # File handler for persistent logging
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            # Stream handler for console logging
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)

            # Log formatting
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
        Store training statistics (rewards, episode lengths, losses) in a pickle file.
        """
        stats_file = self.training_stats_path / f"{self._get_file_prefix()}_stats.pkl"
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
        Save the agent's parameters as a checkpoint for the given episode.

        Also updates the self-play checkpoint buffer.
        
        Args:
            episode (int): The current episode number
        """
        saved_models_dir = self.results_path / "training" / "saved_models"
        saved_models_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = saved_models_dir / f"{self._get_file_prefix()}_checkpoint_ep{episode}.pth"

        torch.save(self.agent.state(), checkpoint_path)
        self.logger.info(f"Saved checkpoint at episode {episode} -> {checkpoint_path}")

        # Maintain a buffer of recent checkpoints for self-play
        if len(self.self_play_checkpoints) >= self.max_checkpoints:
            self.self_play_checkpoints.pop(0)
        self.self_play_checkpoints.append(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        """
        Load an agent checkpoint from disk for self-play.
        
        Args:
            checkpoint_path (Path): Path to the checkpoint file
        
        Returns:
            dict: The agent state from the checkpoint
        """
        return torch.load(checkpoint_path)

    def _select_random_from_top_checkpoints(self, top_fraction=0.3):
        """
        Pick a random checkpoint from the top-performing ones (based on win rate).

        Args:
            top_fraction (float, optional): Fraction of top checkpoints to consider

        Returns:
            Path or None: The selected checkpoint path or None if none are available
        """
        if not self.self_play_checkpoints:
            return None

        # Get win rates for each checkpoint
        checkpoint_win_rates = [(ckpt, self._get_checkpoint_win_rate(ckpt)) for ckpt in self.self_play_checkpoints]
        checkpoint_win_rates.sort(key=lambda x: x[1], reverse=True)
        top_n = max(1, int(len(checkpoint_win_rates) * top_fraction))
        top_checkpoints = [ckpt for ckpt, _ in checkpoint_win_rates[:top_n]]
        return np.random.choice(top_checkpoints)

    def _get_checkpoint_win_rate(self, ckpt):
        """
        Returns a default win rate (stub function) for a checkpoint.

        Args:
            ckpt (Path): The checkpoint file

        Returns:
            float: Win rate, currently set to 0.5
        """
        return 0.5

    def _get_opponent_action(self, obs_agent2):
        """
        Decide the opponent's action based on its type.

        If no opponent is set, or for self-play, it samples or retrieves an action appropriately.
        
        Args:
            obs_agent2: Observation for the opponent agent
        
        Returns:
            np.ndarray: The selected action
        """
        # If no specific opponent is set, sample a random action
        if self.current_opponent is None:
            return self.env.action_space.sample()
        elif self.current_opponent == 'self':
            # For self-play, select a strong checkpoint opponent
            best_checkpoint = self._select_random_from_top_checkpoints(top_fraction=0.3)
            if best_checkpoint is None:
                return self.env.action_space.sample()
            opponent_state = self._load_checkpoint(best_checkpoint)
            temp_agent = TD3Agent(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                **self.model_config
            )
            temp_agent.restore_state(opponent_state)
            return temp_agent.act(obs_agent2, deterministic=True)
        else:
            # For basic opponents (weak or strong), call their action
            if hasattr(self.current_opponent, 'opponent'):
                return self.current_opponent.opponent.act(obs_agent2)
            return self.current_opponent.act(obs_agent2, deterministic=True)

    def _save_gif(self, episode="final"):
        """
        Execute an evaluation episode, capture frames, and save the sequence as a GIF.

        Args:
            episode (str or int, optional): Identifier for the GIF filename
        """
        frames = []
        obs, info = self.eval_env.reset()
        done = False

        while not done:
            # Capture the rendered frame
            frame_rgb = self.eval_env.render(mode='rgb_array')
            frames.append(frame_rgb)

            # Agent selects an action deterministically for evaluation
            action = self.agent.act(obs, deterministic=True)
            obs, reward, done, trunc, info = self.eval_env.step(action)
            if done or trunc:
                break

        gif_filename = f"{self._get_file_prefix()}_checkpoint_ep{episode}.gif"
        gif_path = self.evaluation_gifs_path / gif_filename
        imageio.mimsave(str(gif_path), frames, fps=15)
        self.logger.info(f"Saved evaluation GIF to {gif_path}")

    def _final_metrics(self):
        """
        Calculate and return final training metrics, including average reward, episode length, and losses.

        Returns:
            dict: Final metrics such as average reward, episode length, Q loss, actor loss, and win rate
        """
        # Compute final Q and actor losses
        if self.losses:
            last_losses = self.losses[-100:] if len(self.losses) >= 100 else self.losses
            q_losses = [l[0] for l in last_losses]
            actor_losses = [l[1] for l in last_losses]
            final_q_loss = float(np.mean(q_losses))
            final_actor_loss = float(np.mean(actor_losses))
        else:
            final_q_loss = 0.0
            final_actor_loss = 0.0

        # Compute average reward over a fixed window
        final_window = 100
        if len(self.rewards) >= final_window:
            avg_reward = float(np.mean(self.rewards[-final_window:]))
        else:
            avg_reward = float(np.mean(self.rewards)) if self.rewards else 0.0

        # Compute win rate over the same window
        if self.win_history and len(self.win_history) >= final_window:
            win_rate = float(np.mean(self.win_history[-final_window:]))
        else:
            win_rate = float(np.mean(self.win_history)) if self.win_history else 0.0

        avg_length = float(np.mean(self.lengths)) if self.lengths else 0.0

        metrics = {
            "average_reward": avg_reward,
            "average_length": avg_length,
            "final_q_loss": final_q_loss,
            "final_actor_loss": final_actor_loss,
            "win_rate": win_rate
        }
        self.logger.info(f"Final training metrics: {metrics}")
        return metrics

    def _plot_statistics(self, window_size=50):
        """
        Visualize training statistics (rewards, losses, win rate) and save the plot as a PNG.

        Args:
            window_size (int, optional): Smoothing window size
        """
        # Use the performance window for smoothing
        window_size = self.performance_window

        def running_mean(x, N):
            """Compute running mean using a window of size N."""
            cumsum = np.cumsum(np.insert(x, 0, 0))
            return (cumsum[N:] - cumsum[:-N]) / float(N)

        rewards = self.rewards
        lengths = self.lengths
        losses = self.losses

        # Smooth the rewards curve
        smoothed_rewards = running_mean(rewards, window_size)
        episodes = np.arange(1, len(rewards) + 1)
        smoothed_episodes = np.arange(window_size, len(rewards) + 1)

        # Aggregate losses per episode (average over training steps per episode)
        losses_per_episode = np.mean(np.array(losses).reshape(len(rewards), -1, 2), axis=1)
        q_losses = losses_per_episode[:, 0]
        actor_losses = losses_per_episode[:, 1]
        smoothed_q_losses = running_mean(q_losses, window_size)
        smoothed_actor_losses = running_mean(actor_losses, window_size)

        plt.figure(figsize=(16, 16))

        # Plot Rewards
        plt.subplot(4, 1, 1)
        plt.plot(episodes, rewards, label="Raw Rewards", alpha=0.4)
        plt.plot(smoothed_episodes, smoothed_rewards, label=f"Smoothed Rewards (window={window_size})", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward vs Episodes")
        plt.legend()
        plt.grid()

        # Plot Q Loss
        plt.subplot(4, 1, 2)
        plt.plot(episodes, q_losses, label="Raw Q-Loss", alpha=0.4)
        plt.plot(smoothed_episodes, smoothed_q_losses, label=f"Smoothed Q-Loss (window={window_size})", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Q-Loss")
        plt.title("Q-Loss vs Episodes")
        plt.legend()
        plt.grid()

        # Plot Actor Loss
        plt.subplot(4, 1, 3)
        plt.plot(episodes, actor_losses, label="Raw Actor Loss", alpha=0.4)
        plt.plot(smoothed_episodes, smoothed_actor_losses, label=f"Smoothed Actor Loss (window={window_size})", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Actor Loss")
        plt.title("Actor Loss vs Episodes")
        plt.legend()
        plt.grid()

        # Plot Win Rate if available
        if self.win_history:
            smoothed_win_rate = running_mean(self.win_history, window_size)
            plt.subplot(4, 1, 4)
            plt.plot(episodes, self.win_history, label="Raw Win Rate", alpha=0.4)
            plt.plot(smoothed_episodes, smoothed_win_rate, label=f"Smoothed Win Rate (window={window_size})", linewidth=2)
            plt.xlabel("Episode")
            plt.ylabel("Win Rate")
            plt.title("Win Rate vs Episodes")
            plt.legend()
            plt.grid()

        plt.tight_layout()
        plot_path = self.training_plots_path / f"{self._get_file_prefix()}_training_plot.png"
        plt.savefig(plot_path)
        plt.close()

    def _save_metrics_csv(self, episode, metrics, filename="metrics.csv"):
        """
        Log the training metrics to a CSV file.

        Args:
            episode (int): The episode number
            metrics (dict): Dictionary of metric names and values
            filename (str, optional): Name of the CSV file
        """
        file_path = self.training_stats_path / filename
        write_header = not file_path.exists()

        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)

    def train(self):
        """
        Main training loop for TD3 with curriculum learning using different opponent strategies.

        The loop starts with a warmup phase to populate the replay buffer with random transitions.
        Then, the agent trains over multiple episodes while interacting with a chosen opponent,
        logging metrics, saving checkpoints, and adjusting opponent difficulty based on performance.

        Returns:
            dict: Final training metrics including average reward, episode length, losses, and win rate
        """
        # Retrieve training hyperparameters from configuration
        max_episodes = self.training_config.get("max_episodes")
        max_timesteps = self.training_config.get("max_timesteps")
        warmup_steps = self.model_config.get("warmup_steps")
        log_interval = self.training_config.get("log_interval", 20)
        save_interval = self.training_config.get("save_interval", 500)
        train_iter = self.training_config.get("train_iter", 32)
        render = self.training_config.get("render", False)

        self.logger.info("Starting TD3 Training with Curriculum Learning...")
        self.logger.info(f"Environment: {self.env_name}, max_episodes={max_episodes}, "
                         f"max_timesteps={max_timesteps}, train_iter={train_iter}, "
                         f"warmup_steps={warmup_steps}")

        # Define reward shaping parameters (used in HockeyEnv)
        reward_params = {
            "aim_threshold": 0.8,
            "aim_multiplier": 0.1,
            "defend_multiplier": 0.1,
            "block_multiplier": 0.1,
            "touch_multiplier": 0.1,
            "max_defense_distance": 100.0,
            "max_offset": 20.0,
            "closeness_multiplier": 0.1,
            "wall_threshold": 0.8,
            "wall_multiplier": 0.1
        }

        # --- Warmup ---
        # Populate the replay buffer with random actions
        total_steps = 0
        obs, _info = self.env.reset()
        self.logger.info("Warmup phase started...")
        while total_steps < warmup_steps:
            action = self.env.action_space.sample()
            next_obs, reward, done, trunc, _info = self.env.step(action)
            self.agent.store_transition((obs, action, reward, next_obs, done))
            # Reset environment if episode terminates, else continue with next observation
            obs = next_obs if not (done or trunc) else self.env.reset()[0]
            total_steps += 1
        self.logger.info(f"Warmup phase completed: {warmup_steps} random steps collected")

        # --- Main Training Loop ---
        for i_episode in range(1, max_episodes + 1):
            # Update cumulative episode count
            self.cumulative_episode_count += 1

            # Select an opponent for this episode (random, weak, strong, or self-play)
            self.current_opponent, self.current_opponent_type = self.opponent_pool.sample_opponent()

            # Determine episode length for the current phase
            current_episode_steps = self.training_config.get("environment", {}).get("max_episode_steps", 250)

            # Reset the environment and get observations for both agents
            obs, _info = self.env.reset()
            obs_agent2 = self.env.obs_agent_two()

            # Set up initial metrics for the episode
            episode_reward = 0
            touched = 0
            first_time_touch = 1

            # Accumulators for reward component diagnostics
            base_reward_sum = 0.0
            reward_aim_sum = 0.0
            reward_defend_sum = 0.0
            reward_block_sum = 0.0
            touch_bonus_sum = 0.0
            closeness_reward_sum = 0.0
            reward_wall_sum = 0.0

            # --- Episode Loop ---
            for t in range(current_episode_steps):
                self.timestep += 1
                total_steps += 1

                # Get actions from the agent and the opponent
                action = self.agent.act(obs)
                opponent_action = self._get_opponent_action(obs_agent2)

                # Combine the actions from both agents into one joint action
                joint_action = np.hstack([action, opponent_action])

                # Take a step in the environment using the joint action
                next_obs, reward, done, trunc, _info = self.env.step(joint_action)

                # Record if a puck touch event occurred during this step
                touched = max(touched, _info.get('reward_touch_puck', 0))
                current_reward = reward

                # --- Custom Reward Shaping for HockeyEnv ---
                if self.env_name == "HockeyEnv":
                    # Compute field dimensions
                    W = VIEWPORT_W / SCALE
                    H = VIEWPORT_H / SCALE

                    # Extract environment information based on the agent's role
                    if self.agent_role == "player1":
                        # Player position is stored in obs[0:2] relative to the center
                        player_abs = np.array(obs[0:2]) + np.array([CENTER_X, CENTER_Y])
                        player_angle = obs[2]
                        # Puck position is stored in obs[12:14] relative to the center
                        puck_abs = np.array(obs[12:14]) + np.array([CENTER_X, CENTER_Y])
                        # Puck velocity is stored in obs[14:16]
                        puck_vel = np.array(obs[14:16])
                        # For player1, the opponent goal is on the right
                        opponent_goal = np.array([W / 2 + 245.0 / SCALE + 10.0 / SCALE, H / 2])
                        own_goal = np.array([W / 2 - 245.0 / SCALE - 10.0 / SCALE, H / 2])
                    else:
                        # Mirror the positions for player2
                        player_abs = -np.array(obs[0:2]) + np.array([CENTER_X, CENTER_Y])
                        player_angle = obs[2]
                        puck_abs = -np.array(obs[12:14]) + np.array([CENTER_X, CENTER_Y])
                        puck_vel = -np.array(obs[14:16])
                        # For player2, swap the goals
                        opponent_goal = np.array([W / 2 - 245.0 / SCALE - 10.0 / SCALE, H / 2])
                        own_goal = np.array([W / 2 + 245.0 / SCALE + 10.0 / SCALE, H / 2])

                    # --- Aim Reward ---
                    # Compute the agent's facing direction from its angle
                    player_facing = np.array([np.cos(player_angle), np.sin(player_angle)])
                    # Determine the direction vector from the agent to the opponent's goal
                    to_goal = opponent_goal - player_abs
                    # Measure alignment between the agent's facing and the goal direction using cosine similarity
                    aim_alignment = np.dot(player_facing, to_goal / (np.linalg.norm(to_goal) + 1e-6))
                    # Reward only if the alignment exceeds a given threshold
                    reward_aim = max(0, aim_alignment - reward_params["aim_threshold"]) * reward_params["aim_multiplier"]

                    # --- Wall Reward ---
                    reward_wall = 0.0
                    # If the puck is near the top or bottom walls, calculate a reward based on its alignment with the opponent's goal
                    if puck_abs[1] < 20.0 or puck_abs[1] > (VIEWPORT_H - 20.0):
                        if np.linalg.norm(puck_vel) > 1e-6:
                            # Compute the vector from the puck to the opponent goal
                            puck_to_goal = opponent_goal - puck_abs
                            # Measure how well the puck's movement aligns with the direction toward the goal
                            alignment = np.dot(puck_vel / (np.linalg.norm(puck_vel) + 1e-6),
                                               puck_to_goal / (np.linalg.norm(puck_to_goal) + 1e-6))
                            reward_wall = max(0, alignment - reward_params["wall_threshold"]) * reward_params["wall_multiplier"]

                    # --- Defensive Rewards ---
                    # If the opponent has the puck, calculate rewards:
                    # Defend Reward: Encourage moving towards the agent's own goal
                    # Block Reward: Reward positioning between the puck and the agent's goal
                    opponent_has_puck = obs[-1] > 0
                    if opponent_has_puck:
                        # DEFEND REWARD: Encourage the agent to move towards its own goal
                        distance_to_own = np.linalg.norm(player_abs - own_goal)
                        reward_defend = max(0, 1 - (distance_to_own / reward_params["max_defense_distance"])) * reward_params["defend_multiplier"]

                        # BLOCK REWARD: Reward the agent for positioning between its own goal and the puck
                        puck_vector = puck_abs - own_goal
                        if np.linalg.norm(puck_vector) > 1e-6:
                            # Project the agent's position onto the line connecting its own goal and the puck
                            projection = np.dot(player_abs - own_goal, puck_vector) / (np.linalg.norm(puck_vector)**2) * puck_vector
                            # Compute the perpendicular distance from the agent to this projection
                            perp = player_abs - own_goal - projection
                            d = np.linalg.norm(perp)
                        else:
                            d = 0
                        reward_block = max(0, 1 - d / reward_params["max_offset"]) * reward_params["block_multiplier"]
                    else:
                        reward_defend = 0.0
                        reward_block = 0.0

                    # --- Closeness Reward ---
                    # Reward based on the agent's proximity to the puck
                    closeness_reward = reward_params["closeness_multiplier"] * _info.get('reward_closeness_to_puck', 0)

                    # --- Touch Bonus ---
                    # Add a bonus for the first puck touch, scaled by the timestep
                    touch_term = - (1 - touched) * reward_params["touch_multiplier"] \
                                 + touched * first_time_touch * reward_params["touch_multiplier"] * t

                    # Combine all reward components with the base environment reward
                    base_reward = reward
                    current_reward = (base_reward + reward_aim + reward_wall +
                                      reward_defend + reward_block +
                                      closeness_reward + touch_term)
                    # Ensure that the touch bonus is applied only once
                    first_time_touch = 1 - touched

                    # Update diagnostic accumulators
                    base_reward_sum += base_reward
                    reward_aim_sum += reward_aim
                    reward_defend_sum += reward_defend
                    reward_block_sum += reward_block
                    touch_bonus_sum += touch_term
                    closeness_reward_sum += closeness_reward
                    reward_wall_sum += reward_wall

                    # Record win information based on environment feedback
                    if done or trunc:
                        if self.agent_role == "player1":
                            self.win_history.append(1 if _info.get("winner", 0) == 1 else 0)
                        else:
                            self.win_history.append(1 if _info.get("winner", 0) == -1 else 0)

                # Save the transition and update observations for the next step
                self.agent.store_transition((obs, action, current_reward, next_obs, done))
                obs = next_obs
                obs_agent2 = self.env.obs_agent_two()
                episode_reward += current_reward

                if render:
                    self.env.render()

                if done or trunc:
                    break

            # --- Training Update ---
            # Perform training updates on the agent using mini-batches sampled from the replay buffer
            batch_losses = self.agent.train(num_updates=train_iter)
            self.losses.extend(batch_losses)

            # Log metrics for the episode
            self.rewards.append(episode_reward)
            self.lengths.append(t)

            # Update metrics for the selected opponent
            if done or trunc:
                if self.agent_role == "player1":
                    won = _info.get("winner", 0) == 1
                else:
                    won = _info.get("winner", 0) == -1

                if self.current_opponent_type in self.opponent_pool.opponent_metrics:
                    self.opponent_pool.opponent_metrics[self.current_opponent_type]['games'] += 1
                    if won:
                        self.opponent_pool.opponent_metrics[self.current_opponent_type]['wins'] += 1
                    self.opponent_pool.opponent_metrics[self.current_opponent_type]['history'].append(1 if won else 0)

            # --- Curriculum Update ---
            # Adjust opponent difficulty based on recent training performance
            if len(self.rewards) >= self.performance_window:
                metrics = {
                    'win_rate': np.mean(self.win_history[-self.performance_window:]),
                    'average_reward': np.mean(self.rewards[-self.performance_window:]),
                    'reward_std': np.std(self.rewards[-self.performance_window:]),
                    'episode_count': self.cumulative_episode_count
                }
                self.opponent_pool.update_weights(metrics)
                self.opponent_pool.save_phase_state(self.cumulative_episode_count)

            # Logging to Weights & Biases
            if self.wandb_run is not None:
                if batch_losses:
                    q_losses = [l[0] for l in batch_losses]
                    actor_losses = [l[1] for l in batch_losses]
                    avg_q_loss = np.mean(q_losses)
                    avg_actor_loss = np.mean(actor_losses)
                else:
                    avg_q_loss = 0.0
                    avg_actor_loss = 0.0

                window = min(self.performance_window, len(self.rewards))
                avg_reward = np.mean(self.rewards[-window:])
                win_rate = np.mean(self.win_history[-window:]) if self.win_history else 0.0

                random_win_rate = self.opponent_pool.get_opponent_win_rate('random')
                weak_win_rate = self.opponent_pool.get_opponent_win_rate('weak')
                strong_win_rate = self.opponent_pool.get_opponent_win_rate('strong')
                self_win_rate = self.opponent_pool.get_opponent_win_rate('self')
                reward_std = np.std(self.rewards[-window:]) if self.rewards else 0.0

                self.wandb_run.log({
                    "Reward": episode_reward,
                    "EpisodeLength": t,
                    "TouchRate": touched,
                    "Base_Reward": base_reward_sum,
                    "Closeness_Reward": closeness_reward_sum,
                    "Aim_Reward": reward_aim_sum,
                    "Wall_Reward": reward_wall_sum,
                    "Defend_Reward": reward_defend_sum,
                    "Block_Reward": reward_block_sum,
                    "Touch_Bonus": touch_bonus_sum,
                    "Q_Loss": avg_q_loss,
                    "Actor_Loss": avg_actor_loss,
                    "average_reward": avg_reward,
                    "win_rate": win_rate,
                    "current_episode_steps": current_episode_steps,
                    "random_win_rate": random_win_rate,
                    "weak_win_rate": weak_win_rate,
                    "strong_win_rate": strong_win_rate,
                    "self_win_rate": self_win_rate,
                    "reward_std": reward_std
                }, step=i_episode)

            # Save checkpoints periodically
            if i_episode % save_interval == 0:
                self._save_checkpoint(i_episode)
                self._save_statistics()
                self._save_gif(i_episode)

            # Log training progress
            if i_episode % log_interval == 0:
                avg_reward = np.mean(self.rewards[-log_interval:])
                avg_length = np.mean(self.lengths[-log_interval:])
                avg_win_rate = np.mean(self.win_history[-log_interval:]) if self.win_history else 0.0

                random_win_rate = self.opponent_pool.get_opponent_win_rate('random')
                weak_win_rate = self.opponent_pool.get_opponent_win_rate('weak')
                strong_win_rate = self.opponent_pool.get_opponent_win_rate('strong')
                self_win_rate = self.opponent_pool.get_opponent_win_rate('self')
                reward_std = np.std(self.rewards[-log_interval:]) if self.rewards else 0.0

                self.logger.info(
                    f"Episode {i_episode}\tAvg Length: {avg_length:.2f}\tAvg Reward: {avg_reward:.3f}\tWin Rate: {avg_win_rate:.3f}\t"
                    f"Phase: {self.opponent_pool.current_phase}\tRandom WR: {random_win_rate:.3f}\t"
                    f"Weak WR: {weak_win_rate:.3f}\tStrong WR: {strong_win_rate:.3f}\tSelf WR: {self_win_rate:.3f}\tReward Std: {reward_std:.3f}"
                )
                self.logger.info(
                    f"Episode {i_episode}: Total Reward: {episode_reward:.3f}, Aim: {reward_aim_sum:.3f}, Wall: {reward_wall_sum:.3f}, "
                    f"Defend: {reward_defend_sum:.3f}, Block: {reward_block_sum:.3f}, "
                    f"Closeness: {closeness_reward_sum:.3f}, Touch: {touch_bonus_sum:.3f}"
                )

                metrics_csv = {
                    "episode": i_episode,
                    "avg_reward": avg_reward,
                    "avg_length": avg_length,
                    "win_rate": avg_win_rate,
                    "phase": self.opponent_pool.current_phase
                }
                self._save_metrics_csv(i_episode, metrics_csv)

        # Final saving and metrics computation
        self._save_statistics()
        self._plot_statistics()
        self._save_gif()

        final_metrics = self._final_metrics()

        if self.wandb_run is not None:
            self.wandb_run.log({'average_reward': final_metrics['average_reward'], 'win_rate': final_metrics['win_rate']})
            self.wandb_run.summary.update(final_metrics)

        return final_metrics
