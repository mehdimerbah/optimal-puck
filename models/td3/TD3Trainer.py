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
import imageio
from hockey.hockey_env import HockeyEnv_BasicOpponent, Mode, CENTER_X, CENTER_Y, VIEWPORT_W, VIEWPORT_H, SCALE
from .TD3 import TD3Agent

class TD3Trainer:
    def __init__(self, env_name, training_config, model_config, experiment_path, wandb_run=None, agent_role="player1"):
        self.env_name = env_name
        self.training_config = training_config
        self.model_config = model_config
        self.experiment_path = Path(experiment_path)
        self.wandb_run = wandb_run
        self.agent_role = agent_role
        
        # Create directories for saving
        self.results_path = self.experiment_path / "results"
        self.training_stats_path = self.experiment_path / "results/training/stats"
        self.training_logs_path = self.experiment_path / "results/training/logs"
        self.training_plots_path = self.experiment_path / "results/training/plots"
        # Create evaluation GIF directory
        self.evaluation_gifs_path = self.experiment_path / "results/evaluation/gifs"
        
        for path in [self.training_stats_path, self.training_logs_path, self.training_plots_path, self.evaluation_gifs_path]:
            path.mkdir(parents=True, exist_ok=True)
            
        # Initialize logger first
        self.logger = self._initialize_logger()
        
        # Create environment - handle HockeyEnv first before any gym.make calls
        if self.env_name == "HockeyEnv":
            self.env = HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=False)
            self.eval_env = HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=False)
        else:
            self.env = gym.make(env_name)
            self.eval_env = gym.make(env_name)
        
        # Initialize seeds
        self._initialize_seed()
        
        # Initialize agent
        self.agent = self._initialize_agent()
        
        # Initialize metrics tracking
        self.rewards = []
        self.lengths = []
        self.losses = []
        self.timestep = 0
        
        # Initialize win tracking
        self.win_history = []
        
    def _initialize_seed(self):
        """
        Sets the random seed for PyTorch, NumPy, and the environment's reset() method.
        """
        seed = self.training_config.get("seed", 42)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.env.reset(seed=seed)
            self.logger.info(f"Initialized random seeds to {seed}.")

    def _initialize_agent(self):
        """
        Simple helper that constructs and returns a TD3Agent instance 
        with the config from the 'model_config'.
        """
        return TD3Agent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            **self.model_config
        )

    def _get_file_prefix(self):
        """
        Creates a standardized file prefix containing all hyperparameter information.
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
        rs = self.model_config.get("reward_shaping", {})
        aim_char = str(rs.get("aim_multiplier", 0))[0]
        defend_char = str(rs.get("defend_multiplier", 0))[0]
        block_char = str(rs.get("block_multiplier", 0))[0]
        touch_char = str(rs.get("touch_multiplier", 0))[0]
        closeness_char = str(rs.get("closeness_multiplier", 0))[0]
        wall_char = str(rs.get("wall_multiplier", 0))[0]
        rs_str = (f"_a{aim_char}"
                f"_d{defend_char}"
                f"_b{block_char}"
                f"_t{touch_char}"
                f"_c{closeness_char}"
                f"_w{wall_char}")
        return prefix + rs_str

    def _initialize_logger(self):
        """
        Configures a logger that:
          - Writes logs to a file named with hyperparameter details
          - Also logs to console (stdout)
          - Applies a consistent format (timestamp, log level, message)
        """
        log_file = self.training_logs_path / f"{self._get_file_prefix()}.log"

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
        Saves a checkpoint of the agent's parameters.
        """
        saved_models_dir = self.results_path / "training" / "saved_models"
        saved_models_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = saved_models_dir / f"{self._get_file_prefix()}_checkpoint_ep{episode}.pth"

        torch.save(self.agent.state(), checkpoint_path)
        self.logger.info(f"Saved checkpoint at episode {episode} -> {checkpoint_path}")

    def _save_gif(self, episode="final"):
        """
        Runs one evaluation episode using self.eval_env, collects frames, and saves a GIF.
        If no episode argument is given, defaults to 'final'.
        """
        frames = []
        obs, info = self.eval_env.reset()
        done = False
        while not done:
            frame_rgb = self.eval_env.render(mode='rgb_array')
            frames.append(frame_rgb)
            action = self.agent.act(obs, evaluate=True)
            obs, reward, done, trunc, info = self.eval_env.step(action)
            if done or trunc:
                break
        gif_filename = f"{self._get_file_prefix()}_checkpoint_ep{episode}.gif"
        gif_path = self.evaluation_gifs_path / gif_filename
        imageio.mimsave(str(gif_path), frames, fps=15)
        self.logger.info(f"Saved evaluation GIF to {gif_path}")

    def _final_metrics(self):
        """
        Return final metrics after training finishes.
        Uses a fixed evaluation window for a consistent view of performance.
        """
        if len(self.losses) > 0:
            final_q_loss = self.losses[-1][0]
            final_actor_loss = self.losses[-1][1]
        else:
            final_q_loss = None
            final_actor_loss = None

        # Use a fixed window (last 100 episodes) for final metrics
        final_window = 100
        if len(self.rewards) >= final_window:
            avg_reward = float(np.mean(self.rewards[-final_window:]))
        else:
            avg_reward = float(np.mean(self.rewards)) if len(self.rewards) > 0 else 0.0

        if self.win_history and len(self.win_history) >= final_window:
            win_rate = float(np.mean(self.win_history[-final_window:]))
        else:
            win_rate = float(np.mean(self.win_history)) if len(self.win_history) > 0 else 0.0

        avg_length = float(np.mean(self.lengths)) if len(self.lengths) > 0 else 0.0

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
        plt.title("Reward vs Episodes")
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
        plt.savefig(self.training_plots_path / f"{self._get_file_prefix()}_training_plot.png")
        plt.close()

    def train(self):
        """
        Main training loop that interacts with the environment, collects transitions,
        and trains the agent.
        """
        max_episodes = self.training_config.get("max_episodes")
        max_timesteps = self.training_config.get("max_timesteps")
        warmup_steps = self.model_config.get("warmup_steps")
        log_interval = self.training_config.get("log_interval", 20)
        save_interval = self.training_config.get("save_interval", 500)
        train_iter = self.training_config.get("train_iter", 32)
        render = self.training_config.get("render", False)

        self.logger.info("Starting TD3 Training...")
        self.logger.info(f"Environment: {self.env_name}, max_episodes={max_episodes}, "
                        f"max_timesteps={max_timesteps}, train_iter={train_iter}, "
                        f"warmup_steps={warmup_steps}")

        # Define parameters for reward shaping
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

        # Warmup phase with random actions
        total_steps = 0
        obs, _info = self.env.reset()
        self.logger.info(f"Warmup phase started...")
        while total_steps < warmup_steps:
            action = self.env.action_space.sample()
            next_obs, reward, done, trunc, _info = self.env.step(action)
            self.agent.store_transition((obs, action, reward, next_obs, done))
            obs = next_obs if not (done or trunc) else self.env.reset()[0]
            total_steps += 1
        self.logger.info(f"Warmup phase completed: {warmup_steps} random steps collected")

        for i_episode in range(1, max_episodes + 1):
            obs, _info = self.env.reset()
            episode_reward = 0
            touched = 0
            first_time_touch = 1

            # Initialize accumulators for logging
            base_reward_sum = 0.0
            reward_aim_sum = 0.0
            reward_defend_sum = 0.0
            reward_block_sum = 0.0
            touch_bonus_sum = 0.0
            closeness_reward_sum = 0.0
            reward_wall_sum = 0.0

            for t in range(max_timesteps):
                self.timestep += 1
                total_steps += 1

                action = self.agent.act(obs)
                next_obs, reward, done, trunc, _info = self.env.step(action)
                
                touched = max(touched, _info.get('reward_touch_puck', 0))
                current_reward = reward

                # Extra reward shaping is applied for HockeyEnv
                if self.env_name == "HockeyEnv":
                    # Compute the width and height of the playing field in scaled units
                    W = VIEWPORT_W / SCALE
                    H = VIEWPORT_H / SCALE

                    # === Extract Environment Information Based on Agent Role ===
                    # For player1:
                    #   - The observation stores player1 position relative to [CENTER_X, CENTER_Y] in obs[0:2]
                    #   - The puck position relative to [CENTER_X, CENTER_Y] is in obs[12:14]
                    #   - The puck velocity is in obs[14:16]
                    # For player2:
                    #   - The observation is mirrored
                    #   - Puck positions and velocities are also mirrored
                    if self.agent_role == "player1":
                        player_abs = np.array(obs[0:2]) + np.array([CENTER_X, CENTER_Y])
                        player_angle = obs[2]
                        puck_abs = np.array(obs[12:14]) + np.array([CENTER_X, CENTER_Y])
                        puck_vel = np.array(obs[14:16])
                        # For player1, the opponents goal is on the right side
                        opponent_goal = np.array([W/2 + 245.0/SCALE + 10.0/SCALE, H/2])
                        own_goal = np.array([W/2 - 245.0/SCALE - 10.0/SCALE, H/2])
                    else:  # self.agent_role == "player2"
                        player_abs = -np.array(obs[0:2]) + np.array([CENTER_X, CENTER_Y])
                        player_angle = obs[2]
                        puck_abs = -np.array(obs[12:14]) + np.array([CENTER_X, CENTER_Y])
                        puck_vel = -np.array(obs[14:16])
                        # For player2, swap the goals:
                        opponent_goal = np.array([W/2 - 245.0/SCALE - 10.0/SCALE, H/2])
                        own_goal = np.array([W/2 + 245.0/SCALE + 10.0/SCALE, H/2])

                    # === Aim Reward ===
                    # Calculate the facing direction from agent angle
                    player_facing = np.array([np.cos(player_angle), np.sin(player_angle)])
                    # Calculate the target direction to the opponent goal
                    to_goal = opponent_goal - player_abs
                    # Calculate the alignment of facing and target directions
                    aim_alignment = np.dot(player_facing, to_goal / (np.linalg.norm(to_goal) + 1e-6))
                    # Only the alignment beyond the threshold is rewarded, scaled by aim multiplier
                    reward_aim = max(0, aim_alignment - reward_params["aim_threshold"]) * reward_params["aim_multiplier"]

                    # === Wall Reward ===
                    # Activated when the puck is near the top or bottom walls
                    reward_wall = 0.0
                    if puck_abs[1] < 20.0 or puck_abs[1] > (VIEWPORT_H - 20.0):
                        # Only compute if the puck is moving
                        if np.linalg.norm(puck_vel) > 1e-6:
                            # Compute the vector from the puck to the opponent goal
                            puck_to_goal = opponent_goal - puck_abs
                            # Compute the alignment of puck velocity and the direction to the opponent goal
                            alignment = np.dot(puck_vel / (np.linalg.norm(puck_vel) + 1e-6),
                                            puck_to_goal / (np.linalg.norm(puck_to_goal) + 1e-6))
                            # Check if alignment above a treshold, reward scaled by wall multiplier
                            reward_wall = max(0, alignment - reward_params["wall_threshold"]) * reward_params["wall_multiplier"]

                    # Last observation element indicates puck possession
                    opponent_has_puck = obs[-1] > 0

                    # === Defensive Rewards ===
                    if opponent_has_puck:
                        # DEFEND REWARD: Encourage to move towards own goal
                        # Compute the distance between the agent and its own goal
                        distance_to_own = np.linalg.norm(player_abs - own_goal)
                        # Reward based on the distance to the own goal, scaled by defend multiplier
                        reward_defend = max(0, 1 - (distance_to_own / reward_params["max_defense_distance"])) * reward_params["defend_multiplier"]

                        # BLOCK REWARD: Encourage to position between own goal and puck
                        # Compute the vector from own goal to the puck
                        puck_vector = puck_abs - own_goal
                        if np.linalg.norm(puck_vector) > 1e-6:
                            # Project agent position onto the line from its own goal to the puck
                            projection = np.dot(player_abs - own_goal, puck_vector) / (np.linalg.norm(puck_vector)**2) * puck_vector
                            # Take difference between the agent's position and the projection
                            perp = player_abs - own_goal - projection
                            d = np.linalg.norm(perp)
                        else:
                            d = 0
                        # Reward based on the offset from the blocking position, scaled by block multiplier
                        reward_block = max(0, 1 - d / reward_params["max_offset"]) * reward_params["block_multiplier"]
                    else:
                        reward_defend = 0.0
                        reward_block = 0.0

                    # === Closeness Reward ===
                    # Rewarding the closeness to the puck
                    closeness_reward = reward_params["closeness_multiplier"] * _info.get('reward_closeness_to_puck', 0)

                    # === Touch Bonus ===
                    # No touch results in a small penalty, while touching gives a timestep-scaled bonus
                    touch_term = - (1 - touched) * reward_params["touch_multiplier"] \
                                + touched * first_time_touch * reward_params["touch_multiplier"] * t

                    # Combine the rewards
                    base_reward = reward
                    current_reward = (base_reward 
                                    + reward_aim 
                                    + reward_wall 
                                    + reward_defend 
                                    + reward_block 
                                    + closeness_reward 
                                    + touch_term)
                    # Touch bonus is only applied on the first touch
                    first_time_touch = 1 - touched

                    # Update accumulators for logging
                    base_reward_sum += base_reward
                    reward_aim_sum += reward_aim
                    reward_defend_sum += reward_defend
                    reward_block_sum += reward_block
                    touch_bonus_sum += touch_term
                    closeness_reward_sum += closeness_reward
                    reward_wall_sum += reward_wall

                    # Track wins at episode end
                    if done or trunc:
                        if self.agent_role == "player1":
                            self.win_history.append(1 if _info.get("winner", 0) == 1 else 0)
                        else:  # For player2, winner == -1
                            self.win_history.append(1 if _info.get("winner", 0) == -1 else 0)

                self.agent.store_transition((obs, action, current_reward, next_obs, done))
                obs = next_obs
                episode_reward += current_reward

                if render:
                    self.env.render()

                if done or trunc:
                    break

            # Perform training updates
            batch_losses = self.agent.train(num_updates=train_iter)
            self.losses.extend(batch_losses)

            self.rewards.append(episode_reward)
            self.lengths.append(t)

            # Log to wandb
            if self.wandb_run is not None:
                if batch_losses:  # Only calculate if there are losses
                    q_losses = [l[0] for l in batch_losses]
                    actor_losses = [l[1] for l in batch_losses]
                    avg_q_loss = np.mean(q_losses)
                    avg_actor_loss = np.mean(actor_losses)
                else:
                    avg_q_loss = 0.0
                    avg_actor_loss = 0.0

                # Calculate running averages for metrics (using last 50 episodes)
                window = min(50, len(self.rewards))
                avg_reward = np.mean(self.rewards[-window:])
                win_rate = np.mean(self.win_history[-window:]) if len(self.win_history) > 0 else 0.0

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
                    "win_rate": win_rate
                }, step=i_episode)

            # Save checkpoint & stats periodically, then save a GIF.
            if i_episode % save_interval == 0:
                self._save_checkpoint(i_episode)
                self._save_statistics()
                self._save_gif(i_episode)

            # Print training progress
            if i_episode % log_interval == 0:
                avg_reward = np.mean(self.rewards[-log_interval:])
                avg_length = np.mean(self.lengths[-log_interval:])
                avg_win_rate = np.mean(self.win_history[-log_interval:]) if len(self.win_history) > 0 else 0.0
                self.logger.info(
                    f"Episode {i_episode}\tAvg Length: {avg_length:.2f}\t"
                    f"Avg Reward: {avg_reward:.3f}\tWin Rate: {avg_win_rate:.3f}"
                )

        # Final stats saved and plotted
        self._save_statistics()
        self._plot_statistics()
        self._save_gif()
        
        final_metrics = self._final_metrics()

        if self.wandb_run is not None:
            self.wandb_run.log({'average_reward': final_metrics['average_reward'], 'win_rate': final_metrics['win_rate']})
            self.wandb_run.summary.update(final_metrics)
            
        return final_metrics
