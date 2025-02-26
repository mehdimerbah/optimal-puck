"""
DDPG - An Implementation of the Deep Deterministic Policy Gradient Algorithm

This module provides an implementation of the DDPG algorithm with:
  - A single Q-function (critic) along with a target critic network.
  - A deterministic policy (actor) and its corresponding target network.
  - Exploration via Ornstein-Uhlenbeck noise.
  - Prioritized experience replay for improved sample efficiency.
  
This implementation is based on the following papers and ideas:
    - "Continuous control with deep reinforcement learning" by Lillicrap et al.
    - "Prioritized Experience Replay" by Schaul et al.
    - "Ornstein-Uhlenbeck Noise" by Uhlenbeck and Ornstein.
"""

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces

from ..baseline.Memory import Memory
from ..baseline.MLP import Feedforward
from ..baseline.PrioritizedMemory import PrioritizedMemory

# Global constant for device selection
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)


class UnsupportedSpace(Exception):
    """Raised when an unsupported observation or action space is provided."""
    pass


class OUNoiseProcess:
    """
    Ornstein-Uhlenbeck noise process for temporally correlated exploration in continuous environments.
    
    Attributes:
        shape (tuple): Dimensions of the noise vector.
        theta (float): Speed of mean reversion.
        dt (float): Time step for the process.
    """
    def __init__(self, shape, theta=0.15, dt=1e-2):
        """
        Initialize the OU noise process.
        
        Args:
            shape (tuple): Shape of the noise vector.
            theta (float): Mean reversion rate (default: 0.15).
            dt (float): Time step for noise generation (default: 0.01).
        """
        self.shape = shape
        self.theta = theta
        self.dt = dt
        self.reset()

    def __call__(self):
        """
        Generate the next noise value.
        
        Returns:
            np.ndarray: Noise vector of the specified shape.
        """
        noise = (self.noise_prev +
                 self.theta * (-self.noise_prev) * self.dt +
                 np.sqrt(self.dt) * np.random.normal(size=self.shape))
        self.noise_prev = noise
        return noise

    def reset(self):
        """Reset the noise process to its initial state."""
        self.noise_prev = np.zeros(self.shape)


class QFunction(Feedforward):
    """
    Q-function approximator for DDPG.
    
    This network estimates Q(s, a) and is constructed as a feedforward network with:
      - Input: Concatenated state and action.
      - Output: A scalar Q-value.
    """
    def __init__(self, observation_dim, action_dim, hidden_sizes, learning_rate=1e-3):
        input_size = observation_dim + action_dim
        output_size = 1
        super().__init__(input_size=input_size,
                         hidden_sizes=hidden_sizes,
                         output_size=output_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-6)
        self.loss_fn = nn.SmoothL1Loss()

    def fit(self, observations, actions, targets):
        """
        Update the Q-network parameters given a batch of transitions.
        
        Args:
            observations (torch.Tensor): Batch of observations.
            actions (torch.Tensor): Batch of actions.
            targets (torch.Tensor): Target Q-values.
        
        Returns:
            float: The computed loss.
        """
        self.train()
        self.optimizer.zero_grad()
        q_pred = self.Q_value(observations, actions)
        loss = self.loss_fn(q_pred, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def Q_value(self, observations, actions):
        """
        Compute Q(s, a) by concatenating state and action.
        
        Args:
            observations (torch.Tensor): Batch of observations.
            actions (torch.Tensor): Batch of actions.
        
        Returns:
            torch.Tensor: Estimated Q-values.
        """
        x = torch.cat([observations, actions], dim=1)
        return self.forward(x)


class DDPGAgent:
    """
    DDPG Agent implementing the actor-critic architecture with target networks and prioritized replay.
    
    Key components:
      - Critic and target critic.
      - Actor and target actor.
      - Replay buffer with prioritized experience replay.
      - Exploration using Ornstein-Uhlenbeck noise.
    """
    def __init__(self, observation_space, action_space, **config):
        """
        Initialize the DDPG agent.
        
        Args:
            observation_space (gym.Space): Environment's observation space (must be Box).
            action_space (gym.Space): Environment's action space (must be Box).
            **config: Override default configuration parameters.
        
        Raises:
            UnsupportedSpace: If the observation or action space is not Box.
        """
        if not isinstance(observation_space, spaces.Box):
            raise UnsupportedSpace(f"Observation space {observation_space} must be of type Box.")
        if not isinstance(action_space, spaces.Box):
            raise UnsupportedSpace(f"Action space {action_space} must be of type Box.")

        self.action_space = action_space
        self.observation_space = observation_space
        self.obs_dim = self.observation_space.shape[0]
        self.act_dim = self.action_space.shape[0]

        # Default configuration parameters for DDPG
        self.config = {
            "noise_scale": 0.1,
            "beta": 0.4,  # Importance sampling correction for prioritized replay
            "soft_update": True,
            "tau": 0.005,
            "discount": 0.99,
            "buffer_size": int(1e6),
            "batch_size": 128,
            "learning_rate_actor": 1e-4,
            "learning_rate_critic": 1e-4,
            "hidden_sizes_actor": [256, 256],
            "hidden_sizes_critic": [256, 256],
            "update_target_every": 100,
            "use_target_net": True,
            "prioritized": True,
            
        }
        self.config.update(config)

        self._setup_memory()
        self._setup_noise()
        self._setup_networks()

        self.train_iter = 0

    def _setup_memory(self):
        """Initialize the replay buffer, using prioritized memory if specified."""
        if self.config.get("prioritized", False):
            self.buffer = PrioritizedMemory(max_size=self.config["buffer_size"], alpha=0.6)
        else:
            self.buffer = Memory(max_size=self.config["buffer_size"])

    def _setup_noise(self):
        """Initialize the Ornstein-Uhlenbeck noise process for exploration."""
        self.noise = OUNoiseProcess(shape=(self.act_dim,))
        self.noise_scale = self.config["noise_scale"]

    def _setup_networks(self):
        """Initialize the critic and actor networks along with their target networks."""
        # Critic and target critic networks
        self.Q = QFunction(
            observation_dim=self.obs_dim,
            action_dim=self.act_dim,
            hidden_sizes=self.config["hidden_sizes_critic"],
            learning_rate=float(self.config["learning_rate_critic"])
        ).to(DEVICE)
        self.Q_target = QFunction(
            observation_dim=self.obs_dim,
            action_dim=self.act_dim,
            hidden_sizes=self.config["hidden_sizes_critic"],
            learning_rate=0.0  # No optimizer for target network
        ).to(DEVICE)

        # Define action bounds and output scaling
        high = torch.tensor(self.action_space.high, dtype=torch.float32, device=DEVICE)
        low = torch.tensor(self.action_space.low, dtype=torch.float32, device=DEVICE)

        self.action_low = low.cpu().numpy()
        self.action_high = high.cpu().numpy()

        output_activation = lambda x: (torch.tanh(x) * (high - low) / 2.0) + (high + low) / 2.0

        # Actor and target actor networks
        self.policy = Feedforward(
            input_size=self.obs_dim,
            hidden_sizes=self.config["hidden_sizes_actor"],
            output_size=self.act_dim,
            activation_fun=nn.ReLU(),
            output_activation=output_activation
        ).to(DEVICE)
        self.policy_target = Feedforward(
            input_size=self.obs_dim,
            hidden_sizes=self.config["hidden_sizes_actor"],
            output_size=self.act_dim,
            activation_fun=nn.ReLU(),
            output_activation=output_activation
        ).to(DEVICE)

        self._copy_nets()

        # Optimizer for the actor network
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=float(self.config["learning_rate_actor"]),
            eps=1e-6
        )

    def _copy_nets(self):
        """Synchronize target networks by directly copying parameters."""
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    

    def _soft_update(self, tau=None):
       """Perform a soft update of target network parameters."""
       if tau is None:
           tau = self.config.get("tau", 0.005)
       # Soft update for policy network 
       for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
           target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
       # Soft update for value network
       for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
           target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def act(self, observation, evaluate=False):
        """
        Given an observation, compute the action. Noise is added during training.
        
        Args:
            observation (np.ndarray): The current state.
            evaluate (bool): If True, no noise is added.
        
        Returns:
            np.ndarray: The chosen action clipped to the environment's bounds.
        """
        self.policy.eval()
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action = self.policy(obs_tensor).cpu().numpy()[0]
        if evaluate:
            return action
        # Add exploration noise for training
        noise = self.noise()
        noisy_action = action + self.noise_scale * noise
        return np.clip(noisy_action, self.action_low, self.action_high)

    def store_transition(self, transition):
        """
        Store a transition in the replay buffer.
        
        Args:
            transition (tuple): A tuple (state, action, reward, next_state, done).
        """
        self.buffer.add_transition(transition)

    def train(self, num_updates=32):
        """
        Perform a series of training updates on the actor and critic networks.
        
        Args:
            num_updates (int): Number of gradient update steps.
        
        Returns:
            list: A list of tuples (critic_loss, actor_loss) for each update.
        """
        losses = []
        self.train_iter += 1

        # Update target networks at fixed intervals (if enabled)
        if self.config["use_target_net"] and not self.config["soft_update"] and (self.train_iter % self.config["update_target_every"] == 0):
            self._copy_nets()

        batch_size = self.config["batch_size"]
        discount = self.config["discount"]
        beta = self.config.get("beta", 0.4)

        for _ in range(num_updates):
            # Sample a batch from the replay buffer
            sample = self.buffer.sample(batch=batch_size, beta=beta) if self.config.get("prioritized", False) else self.buffer.sample(batch=batch_size)
            if sample is None or (self.config.get("prioritized", False) and len(sample[0]) == 0):
                break

            if self.config.get("prioritized", False):
                samples, indices, is_weights = sample
            else:
                samples = sample

            # Unpack transitions: (state, action, reward, next_state, done)
            states_list, actions_list, rewards_list, next_states_list, dones_list = zip(*samples)
            states = torch.tensor(np.vstack(states_list), dtype=torch.float32, device=DEVICE)
            actions = torch.tensor(np.vstack(actions_list), dtype=torch.float32, device=DEVICE)
            rewards = torch.tensor(np.vstack(rewards_list), dtype=torch.float32, device=DEVICE)
            next_states = torch.tensor(np.vstack(next_states_list), dtype=torch.float32, device=DEVICE)
            dones = torch.tensor(np.vstack(dones_list), dtype=torch.float32, device=DEVICE)
            if self.config.get("prioritized", False):
                is_weights_tensor = torch.tensor(is_weights, dtype=torch.float32, device=DEVICE).unsqueeze(1)

            # Compute target Q-values using the target actor and critic
            with torch.no_grad():
                next_actions = self.policy_target(next_states)
                q_next = self.Q_target.Q_value(next_states, next_actions)
                q_target = rewards + discount * (1 - dones) * q_next

            # Update critic network
            q_pred = self.Q.Q_value(states, actions)
            td_errors = torch.abs(q_pred - q_target)
            loss_fn = nn.SmoothL1Loss(reduction='none')
            losses_per_sample = loss_fn(q_pred, q_target)
            if self.config.get("prioritized", False):
                weighted_loss = (is_weights_tensor * losses_per_sample).mean()
            else:
                weighted_loss = losses_per_sample.mean()

            self.Q.optimizer.zero_grad()
            weighted_loss.backward()
            self.Q.optimizer.step()

            # Update actor network: maximize Q-value under current policy
            pred_actions = self.policy(states)
            actor_loss = -self.Q.Q_value(states, pred_actions).mean()
            self.policy_optimizer.zero_grad()
            actor_loss.backward()
            self.policy_optimizer.step()

            # Update priorities if using prioritized replay
            if self.config.get("prioritized", False):
                new_priorities = (td_errors.detach().cpu().numpy() + 1e-6).flatten()
                self.buffer.update_priorities(indices, new_priorities)

            losses.append((weighted_loss.item(), actor_loss.item()))

        # Soft update of target networks
        if self.config.get("soft_update", False):
            self._soft_update()

        return losses

    def state(self):
        """
        Retrieve the current state of the networks for checkpointing.
        
        Returns:
            dict: A dictionary with the critic and actor network states.
        """
        return {
            "Q_state": self.Q.state_dict(),
            "policy_state": self.policy.state_dict()
        }

    def restore_state(self, state):
        """
        Restore network parameters from a checkpoint.
        
        Args:
            state (dict): A dictionary containing saved network states.
        """
        self.Q.load_state_dict(state["Q_state"])
        self.policy.load_state_dict(state["policy_state"])
        self._copy_nets()

    def reset_noise(self):
        """Reset the exploration noise process at the start of a new episode."""
        self.noise.reset()
