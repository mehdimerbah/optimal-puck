import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal
from gymnasium import spaces

from ..baseline.memory import Memory
from ..baseline.feedforward import Feedforward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

class UnsupportedSpace(Exception):
    """Exception raised when observation or action space is unsupported."""
    pass

class OUNoise:
    """Ornstein-Uhlenbeck noise for exploration in continuous environments."""
    def __init__(self, shape, theta=0.15, dt=1e-2):
        self.shape = shape
        self.theta = theta
        self.dt = dt
        self.noise_prev = np.zeros(self.shape)
        self.reset()

    def __call__(self):
        noise = (
            self.noise_prev +
            self.theta * (-self.noise_prev) * self.dt +
            np.sqrt(self.dt) * np.random.normal(size=self.shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self):
        self.noise_prev = np.zeros(self.shape)

class QFunction(Feedforward):
    """
    A Q-function approximator for continuous actions:
    Q(s,a) -> scalar
    Inherits from a generic Feedforward MLP but ensures it has:
     - input_size = obs_dim + act_dim
     - output_size = 1
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
        Train the Q-function on a batch of transitions (s, a, Q_target).
        observations, actions, targets are torch.Tensor
        """
        self.train()
        self.optimizer.zero_grad()

        # Forward pass
        q_pred = self.Q_value(observations, actions)
        loss = self.loss_fn(q_pred, targets)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def Q_value(self, observations, actions):
        """
        Compute Q(s,a) by concatenating obs & actions before feeding the MLP.
        """
        x = torch.cat([observations, actions], dim=1)
        return self.forward(x)

class DDPGAgent:
    """
    A DDPG Agent, containing:
      - A Q-function (critic) + target Q-function
      - A policy (actor) + target policy
      - A replay buffer (Memory)
      - Methods for acting, storing transitions, and training.
    """
    def __init__(self, observation_space, action_space, **config):
        # Validate spaces
        if not isinstance(observation_space, spaces.Box):
            raise UnsupportedSpace(f"Observation space {observation_space} must be Box.")
        if not isinstance(action_space, spaces.Box):
            raise UnsupportedSpace(f"Action space {action_space} must be Box.")

        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.config = {
            "noise_scale": 0.1,                  
            "discount": 0.99, 
            "epsilon": 0.05,     
            "buffer_size": int(1e6),   
            "batch_size": 128,           
            "learning_rate_actor": 1e-4, 
            "learning_rate_critic": 1e-4,
            "hidden_sizes_actor": [256, 256],
            "hidden_sizes_critic": [256, 256],
            "update_target_every": 100,
            "use_target_net": True
        }
        self.config.update(config)

        # Initialize replay buffer
        self.buffer = Memory(max_size=self.config["buffer_size"])

        # Noise process - ABLATION OF NOISE : Testing the effect of removing the noise on the exploration process
        self.action_noise = OUNoise((self.act_dim))
        self.noise_scale = self.config.get("noise_scale", 0.2)
        #self.epsilon = self.config["epsilon"]

        # Q (critic) and target Q
        self.Q = QFunction(
            observation_dim=self.obs_dim,
            action_dim=self.act_dim,
            hidden_sizes=self.config["hidden_sizes_critic"],
            learning_rate=float(self.config["learning_rate_critic"])
        ).to(device)
        self.Q_target = QFunction(
            observation_dim=self.obs_dim,
            action_dim=self.act_dim,
            hidden_sizes=self.config["hidden_sizes_critic"],
            learning_rate=0.0  # target net doesn't need an optimizer
        ).to(device)

        
        # Define action space bounds
        high = torch.tensor(action_space.high, dtype=torch.float32, device=device)
        low  = torch.tensor(action_space.low, dtype=torch.float32, device=device)
        self.high = high.cpu().numpy()
        self.low = low.cpu().numpy()

        # Output activation: scale tanh outputs to [low, high]
        output_activation = lambda x: (torch.tanh(x) * (high - low) / 2.0) + (high + low) / 2.0

        # Policy (actor) and target policy
        self.policy = Feedforward(
            input_size=self.obs_dim,
            hidden_sizes=self.config["hidden_sizes_actor"],
            output_size=self.act_dim,
            activation_fun=nn.ReLU(),
            output_activation=output_activation
        ).to(device)
        self.policy_target = Feedforward(
            input_size=self.obs_dim,
            hidden_sizes=self.config["hidden_sizes_actor"],
            output_size=self.act_dim,
            activation_fun=nn.ReLU(),
            output_activation=output_activation
        ).to(device)

        # Copy initial weights to target networks
        self._copy_nets()

        # Actor optimizer
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=float(self.config["learning_rate_actor"]),
            eps=1e-6
        )

        self.train_iter = 0

    def _copy_nets(self):
        """Synchronize target networks with main networks."""
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    def reset(self):
        """Reset noise process at the beginning of each episode."""
        self.action_noise.reset()


    def act(self, observation, evaluate=False):
        """
        Return action for the given observation, adding OU noise
        if we are not in evaluate mode.
        """
        self.policy.eval()

        # Convert observation -> torch tensor
        obs_tensor = torch.tensor(observation, dtype=torch.float32).to(device).unsqueeze(0)

        with torch.no_grad():
            policy_action = self.policy(obs_tensor)

        # Convert back to numpy array
        policy_action = policy_action.cpu().numpy()[0]

        if evaluate:
            # No noise in evaluation
            return policy_action
        else:
            # Add OU noise
            noise = self.action_noise()
            noisy_action = policy_action + self.noise_scale * noise

            # clip the action to env bounds
            clipped_action = np.clip(noisy_action, self.low, self.high)
            return clipped_action

    # def act(self, observation, epsilon=None, evaluate=False):
    #     """
    #     Return action for the given observation, choose an action with eps-greedy strategy.
    #     """
    #     self.policy.eval()

    #     # Ensure `obs_tensor` is on the correct device
    #     obs_tensor = torch.tensor(observation, dtype=torch.float32).to(device).unsqueeze(0)
        
    #     with torch.no_grad():
    #         policy_action = self.policy(obs_tensor)  # still on GPU
    #     policy_action = policy_action.cpu().numpy()[0] # Now switching to CPU
        
    #     # Decide if we explore or exploit
    #     if np.random.rand() < (epsilon if epsilon is not None else self.epsilon) and not evaluate:
    #         # Random action in continuous space
    #         random_action = np.random.uniform(low=self.low, high=self.high)
    #         action = random_action
    #     else:
    #         # Use the deterministic action from policy
    #         action = policy_action

    #     return action

    def store_transition(self, transition):
        """Add a transition (s, a, r, s_next, done) to the replay buffer."""
        self.buffer.add_transition(transition)

    def train(self, num_updates=32):
        """
        Perform gradient updates on the Q-network (critic) and Policy (actor).
        """
        losses = []
        self.train_iter += 1

        # Optional target network update
        if self.config["use_target_net"] and (self.train_iter % self.config["update_target_every"] == 0):
            self._copy_nets()

        # Shortcut
        batch_size = self.config["batch_size"]
        discount = self.config["discount"]

        for _ in range(num_updates):
            # Sample from replay
            data = self.buffer.sample(batch=batch_size)
            if data is None:
                # Not enough data in buffer yet
                break

            # Prepare torch tensors
            states  = torch.tensor(np.vstack(data[:, 0]), dtype=torch.float32, device=device)
            actions = torch.tensor(np.vstack(data[:, 1]), dtype=torch.float32, device=device)
            rewards = torch.tensor(np.vstack(data[:, 2]), dtype=torch.float32, device=device)
            next_states = torch.tensor(np.vstack(data[:, 3]), dtype=torch.float32, device=device)
            dones   = torch.tensor(np.vstack(data[:, 4]), dtype=torch.float32, device=device)

            # Update Critic
            with torch.no_grad():
                # Target action from policy_target
                next_actions = self.policy_target(next_states)
                # Q-value from Q_target
                q_next = self.Q_target.Q_value(next_states, next_actions)
                q_target = rewards + discount * (1 - dones) * q_next

            q_loss = self.Q.fit(states, actions, q_target)

            # Update Actor
            pred_actions = self.policy(states)
            actor_loss = -self.Q.Q_value(states, pred_actions).mean()
            self.policy_optimizer.zero_grad()
            actor_loss.backward()
            self.policy_optimizer.step()

            losses.append((q_loss, actor_loss.item()))

        return losses

    def state(self):
        """Return model parameters for checkpointing."""
        return {
            "Q_state": self.Q.state_dict(),
            "policy_state": self.policy.state_dict()
        }

    def restore_state(self, state):
        """Load model parameters from checkpoint."""
        self.Q.load_state_dict(state["Q_state"])
        self.policy.load_state_dict(state["policy_state"])
        self._copy_nets()
