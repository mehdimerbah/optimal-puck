"""
TD3 - An Implementation of the Twin Delayed DDPG Algorithm

This module provides an implementation of the TD3 algorithm, following the approach in:
"Addressing Function Approximation Error in Actor-Critic Methods" (Fujimoto et al., 2018).

Key components:
  - Twin critics to mitigate overestimation.
  - Delayed updates for the actor network.
  - Smoothing of target policy outputs.

Additional feature:
  - Exploration using colored noise, inspired by insights from:
  "Proceedings of the Eleventh International Conference on Learning Representations (ICLR)" (Eberhard et al., 2023).
"""

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces

from ..baseline.memory import Memory
from ..baseline.feedforward import Feedforward

# Global constants
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

class UnsupportedSpace(Exception):
    """Exception raised when an unsupported observation or action space is provided."""
    pass

class ColoredNoiseProcess:
    """
    A colored noise generator using FFT to produce temporally correlated noise.

    The noise color is controlled by the parameter beta:
      beta = 0 yields white noise (no correlation),
      beta = 1 gives pink noise (1/f),
      beta = 2 produces red/brown noise (1/f²).
    """
    
    def __init__(self, size, beta=1.0, dt=0.01, sigma=1.0, episode_length=2000):
        """
        Initialize the colored noise process.

        Args:
            size (int): Dimension of the noise vector (number of action dimensions)
            beta (float): Noise color parameter (default 1.0 for pink noise)
            dt (float): Time step for noise generation (default 0.01)
            sigma (float): Scale (standard deviation) of the noise (default 1.0)
            episode_length (int): Maximum number of steps per episode (default 2000)
        """
        self.size = size
        self.beta = beta
        self.dt = dt
        self.sigma = sigma
        self.episode_length = episode_length
        self.eps = 1e-8  # Numerical stability constant
        self.reset()
    
    def reset(self):
        """Reinitialize the noise process for a new episode by generating a fresh noise sequence and resetting the step counter."""
        self.noise_sequences = self._generate_colored_noise(n_steps=self.episode_length, n_dims=self.size)
        self.step_index = 0
        
    def _generate_colored_noise(self, n_steps, n_dims):
        """
        Generate colored noise using FFT.

        Args:
            n_steps (int): Total number of timesteps
            n_dims (int): Number of noise dimensions (typically matching the action dimensions)

        Returns:
            np.ndarray: A noise array with shape (n_steps, n_dims)
        """
        # Generate initial white noise sample
        white = np.random.normal(0.0, 1.0, (n_steps, n_dims))
        
        # Set up frequency components for FFT (avoid division by zero)
        freqs = np.fft.fftfreq(n_steps)
        freqs[0] = float('inf')  # Avoid division by zero for the DC component
        
        # Calculate scaling factors using the desired power spectrum
        Sf = np.abs(freqs)[:, np.newaxis] ** (-self.beta / 2)
        Sf[0] = 0  # Remove the DC component's influence
        
        # Apply FFT to the white noise and scale it according to the computed factors
        white_fft = np.fft.fft(white, axis=0)
        colored_fft = Sf * white_fft
        
        # Transform back to the time domain using inverse FFT
        colored = np.real(np.fft.ifft(colored_fft, axis=0))
        
        # Normalize each noise dimension to have zero mean and the specified standard deviation
        for d in range(n_dims):
            mean = np.mean(colored[:, d])
            std = np.std(colored[:, d])
            if std < self.eps:
                std = 1.0
                print(f"Warning: Using default scaling due to near-zero variance in noise dimension {d}.")
            colored[:, d] = self.sigma * (colored[:, d] - mean) / std
        
        return colored
    
    def __call__(self):
        """
        Return the noise vector for the current timestep and increment the counter.
        Resets the noise sequence if the end of the episode is reached.

        Returns:
            np.ndarray: The noise vector for the current step
        """
        if self.step_index >= self.episode_length:
            self.reset()
        noise = self.noise_sequences[self.step_index]
        self.step_index += 1
        return noise

    def sample(self):
        """
        Alias for __call__ to return the noise vector for the current timestep.

        Returns:
            np.ndarray: The current noise vector
        """
        return self.__call__()

class TwinQFunction(nn.Module):
    """
    Uses two Q-networks to help minimize overestimation bias in the TD3 algorithm.
    """
    def __init__(self, observation_dim, action_dim, hidden_sizes=[128, 128, 64],
                 learning_rate=0.0003, is_target=False):
        """
        Initialize two Q-networks (Q1 and Q2) for the TD3 algorithm.

        Args:
            observation_dim (int): Dimension of the observation space
            action_dim (int): Dimension of the action space
            hidden_sizes (list): Hidden layer sizes for the networks
            learning_rate (float): Learning rate for updates (use 0 for target networks)
            is_target (bool): Flag indicating if this instance is a target network
        """
        super(TwinQFunction, self).__init__()
        self.Q1 = Feedforward(input_size=observation_dim + action_dim, hidden_sizes=hidden_sizes, output_size=1)
        self.Q2 = Feedforward(input_size=observation_dim + action_dim, hidden_sizes=hidden_sizes, output_size=1)
        self.is_target = is_target

        if not is_target and learning_rate > 0:
            self.optimizer = torch.optim.Adam(
                list(self.Q1.parameters()) + list(self.Q2.parameters()),
                lr=learning_rate,
                eps=1e-6
            )
        self.loss_fn = nn.SmoothL1Loss()

    def fit(self, observations, actions, targets):
        """
        Update the twin Q-networks using a batch of transitions.

        Args:
            observations (torch.Tensor): Batch of observations
            actions (torch.Tensor): Batch of actions
            targets (torch.Tensor): Target Q-values for computing the loss

        Returns:
            float: The combined loss from both Q-networks
        """
        if self.is_target:
            raise ValueError("Cannot train target networks.")

        self.train()
        self.optimizer.zero_grad()

        # Forward pass through both Q-networks
        q1_pred = self.Q1(torch.cat([observations, actions], dim=1))
        q2_pred = self.Q2(torch.cat([observations, actions], dim=1))

        loss1 = self.loss_fn(q1_pred, targets)
        loss2 = self.loss_fn(q2_pred, targets)
        total_loss = loss1 + loss2

        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.Q1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.Q2.parameters(), max_norm=1.0)
        self.optimizer.step()

        return total_loss.item()

    def Q_value(self, observations, actions):
        """
        Compute Q-values using the first Q-network for the given observations and actions.

        Args:
            observations (torch.Tensor): Batch of observations
            actions (torch.Tensor): Batch of actions

        Returns:
            torch.Tensor: The Q-values from the first network
        """
        return self.Q1(torch.cat([observations, actions], dim=1))

    def Q2_value(self, observations, actions):
        """
        Compute Q-values using the second Q-network for the given observations and actions.

        Args:
            observations (torch.Tensor): Batch of observations
            actions (torch.Tensor): Batch of actions

        Returns:
            torch.Tensor: The Q-values from the second network
        """
        return self.Q2(torch.cat([observations, actions], dim=1))

class TD3Agent:
    """
    Implementation of the TD3 algorithm featuring:
      - Twin critics to reduce overestimation.
      - Delayed actor updates.
      - Target policy smoothing with noise.
      - Exploration using colored noise.
    """

    def __init__(self, observation_space, action_space, **config):
        """
        Set up the TD3 agent with the given observation and action spaces.

        Args:
            observation_space (gym.Space): The environment's observation space (should be Box)
            action_space (gym.Space): The environment's action space (should be Box)
            **config: Additional configuration parameters to override defaults

        Raises:
            UnsupportedSpace: If the provided spaces are not of type Box
        """
        if not isinstance(observation_space, spaces.Box):
            raise UnsupportedSpace("Observation space must be of type Box.")
        if not isinstance(action_space, spaces.Box):
            raise UnsupportedSpace("Action space must be of type Box.")

        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.action_space = action_space

        # Convert the action space limits to torch tensors on the correct device
        self.action_low = torch.from_numpy(action_space.low).to(DEVICE)
        self.action_high = torch.from_numpy(action_space.high).to(DEVICE)

        # Default configuration parameters
        self.config = {
            "eps": 0.1,                             # Exploration noise scale
            "noise_beta": 1.0,                      # Noise color (0=white, 1=pink, 2=red)
            "warmup_steps": 25000,                  # Steps for initial random exploration
            "policy_noise": 0.2,                    # Noise for target policy smoothing
            "noise_clip": 0.5,                      # Clipping range for policy noise
            "policy_delay": 2,                      # Frequency of policy updates
            "polyak": 0.995,                        # Polyak averaging coefficient for target updates
            "discount": 0.99,                       # Discount factor for rewards
            "buffer_size": int(1e6),                # Replay buffer size
            "batch_size": 256,                      # Batch size for training
            "learning_rate_actor": 0.0003,          # Actor network learning rate
            "learning_rate_critic": 0.0003,         # Critic network learning rate
            "hidden_sizes_actor": [128, 128],       # Hidden layer sizes for actor network layers
            "hidden_sizes_critic": [128, 128, 64],  # Hidden layer sizes for critic network layers
            "use_target_net": True,                 # Flag to indicate whether target networks are used
            "max_steps": 2000                       # Maximum steps per episode
        }
        self.config.update(config)

        # Initialize components
        self._setup_noise()
        self._setup_memory()
        self._setup_networks()
        self.train_iter = 0

    def _setup_noise(self):
        """Initialize the colored noise generator for exploration."""
        self.noise = ColoredNoiseProcess(
            size=self.act_dim,
            beta=self.config['noise_beta'],
            sigma=self.config['eps'],
            episode_length=self.config['max_steps']
        )

    def _setup_memory(self):
        """Set up the replay buffer for storing transitions."""
        self.buffer = Memory(max_size=self.config["buffer_size"])

    def _setup_networks(self):
        """Initialize the twin critic networks, actor network, and their corresponding target networks."""
        # Initialize twin critic networks
        self.Q = TwinQFunction(
            observation_dim=self.obs_dim,
            action_dim=self.act_dim,
            hidden_sizes=self.config["hidden_sizes_critic"],
            learning_rate=self.config["learning_rate_critic"],
            is_target=False
        ).to(DEVICE)

        # Create target twin critics (without an optimizer)
        self.Q_target = TwinQFunction(
            observation_dim=self.obs_dim,
            action_dim=self.act_dim,
            hidden_sizes=self.config["hidden_sizes_critic"],
            learning_rate=0,
            is_target=True
        ).to(DEVICE)

        # Build the actor network with output scaling to match the action bounds
        high = torch.from_numpy(self.action_space.high).to(DEVICE)
        low = torch.from_numpy(self.action_space.low).to(DEVICE)
        output_activation = lambda x: (torch.tanh(x) * (high - low) / 2) + (high + low) / 2

        self.policy = Feedforward(
            input_size=self.obs_dim,
            hidden_sizes=self.config["hidden_sizes_actor"],
            output_size=self.act_dim,
            activation_fun=nn.ReLU(),
            output_activation=output_activation
        ).to(DEVICE)

        # Initialize the target actor network
        self.policy_target = Feedforward(
            input_size=self.obs_dim,
            hidden_sizes=self.config["hidden_sizes_actor"],
            output_size=self.act_dim,
            activation_fun=nn.ReLU(),
            output_activation=output_activation
        ).to(DEVICE)

        # Copy weights from main networks to the target networks
        self._copy_nets()

        # Set up the optimizer for the actor network
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config["learning_rate_actor"],
            eps=1e-6
        )

    def act(self, obs, deterministic=False, evaluate=False):
        """
        Given an observation, produce an action. In non-deterministic mode, adds exploration noise.

        Args:
            obs (array-like): Current observation
            deterministic (bool): If True, do not add exploration noise
            evaluate (bool): Alias for deterministic mode

        Returns:
            np.ndarray: The chosen action
        """
        self.policy.eval()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        
        with torch.no_grad():
            action = self.policy(obs_tensor)
            # If not in deterministic mode, introduce exploration noise
            if not (deterministic or evaluate):
                noise = torch.tensor(self.noise.sample(), dtype=torch.float32, device=DEVICE)
                action.add_(noise)
                action.clamp_(self.action_low, self.action_high)
            return action.cpu().numpy()

    def _soft_update(self, target, source):
        """
        Smoothly update target network parameters using Polyak averaging.

        Updates: θ_target = polyak * θ_target + (1 - polyak) * θ_source
        """
        polyak = self.config["polyak"]
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.mul_(polyak).add_((1.0 - polyak) * param)

    def train(self, num_updates=32):
        """
        Execute training updates by sampling mini-batches from the replay buffer.

        Args:
            num_updates (int): The number of training iterations

        Returns:
            list: A list of tuples (critic_loss, actor_loss) recorded for each update
        """
        losses = []
        self.train_iter += 1

        for i in range(num_updates):
            # Sample a mini-batch of transitions from the replay buffer
            data = self.buffer.sample(batch=self.config['batch_size'])
            if data is None:
                break

            states = torch.tensor(np.vstack(data[:, 0]), dtype=torch.float32, device=DEVICE)
            actions = torch.tensor(np.vstack(data[:, 1]), dtype=torch.float32, device=DEVICE)
            rewards = torch.tensor(np.vstack(data[:, 2]), dtype=torch.float32, device=DEVICE)
            next_states = torch.tensor(np.vstack(data[:, 3]), dtype=torch.float32, device=DEVICE)
            dones = torch.tensor(np.vstack(data[:, 4]), dtype=torch.float32, device=DEVICE)

            # Normalize rewards to stabilize training
            reward_std = rewards.std() + 1e-6
            r_normalized = rewards / reward_std

            # Compute target Q-values with noise for smoothing
            with torch.no_grad():
                noise = torch.randn_like(actions) * self.config["policy_noise"]
                noise.clamp_(-self.config["noise_clip"], self.config["noise_clip"])
                next_action = self.policy_target(next_states)
                next_action.add_(noise).clamp_(self.action_low, self.action_high)
                
                q1_next = self.Q_target.Q_value(next_states, next_action)
                q2_next = self.Q_target.Q2_value(next_states, next_action)
                q_next = torch.min(q1_next, q2_next)
                q_target = r_normalized + self.config["discount"] * (1 - dones) * q_next
                q_target = q_target.view(-1, 1)

            # Update the twin critic networks
            q_loss = self.Q.fit(states, actions, q_target)
            
            actor_loss_val = 0.0
            # Delayed actor update: update the actor network every few iterations
            if i % self.config["policy_delay"] == 0:
                self.policy.train()
                a_pred = self.policy(states)
                actor_loss = -self.Q.Q_value(states, a_pred).mean()

                self.policy_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.policy_optimizer.step()
                actor_loss_val = actor_loss.item()

                # If using target networks, softly update them to track the main networks
                if self.config["use_target_net"]:
                    self._soft_update(self.Q_target, self.Q)
                    self._soft_update(self.policy_target, self.policy)

            losses.append((q_loss, actor_loss_val))

        return losses

    def _copy_nets(self):
        """Copy the main networks' weights directly to the target networks."""
        self.Q_target.Q1.load_state_dict(self.Q.Q1.state_dict())
        self.Q_target.Q2.load_state_dict(self.Q.Q2.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    def store_transition(self, transition):
        """Add a new transition (state, action, reward, next_state, done) to the replay buffer."""
        self.buffer.add_transition(transition)

    def state(self):
        """Return the current network weights for checkpointing."""
        return {
            "Q_state": self.Q.state_dict(),
            "policy_state": self.policy.state_dict()
        }

    def restore_state(self, state):
        """Load saved network weights from a checkpoint to restore the agent's state."""
        self.Q.load_state_dict(state["Q_state"])
        self.policy.load_state_dict(state["policy_state"])
        self._copy_nets()

    def reset_noise(self):
        """Restart the exploration noise generator for a new episode."""
        self.noise.reset()

