"""
Twin Delayed Deep Deterministic Policy Gradient (TD3) Implementation
Based on the paper "Addressing Function Approximation Error in Actor-Critic Methods"
by Fujimoto et al. (2018).

Key Features:
- Twin Q-networks to reduce overestimation bias
- Delayed policy updates
- Target policy smoothing
- Colored noise exploration
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import optparse
from pathlib import Path

from ..baseline_alt.memory import Memory
from ..baseline_alt.feedforward import Feedforward

# Global constants
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

class UnsupportedSpace(Exception):
    """Exception raised when observation or action space is unsupported."""
    pass

class ColoredNoiseProcess:
    """
    Implements temporally correlated noise for exploration using FFT method.
    
    Generates colored noise with specified power spectrum characteristics:
    - β = 0: White noise (uncorrelated)
    - β = 1: Pink noise (1/f power spectrum)
    - β = 2: Red/Brown noise (1/f² power spectrum)
    """
    
    def __init__(self, size, beta=1.0, dt=0.01, sigma=1.0, episode_length=2000):
        """
        Args:
            size (int): Dimension of the noise vector (number of action dimensions)
            beta (float): Color parameter controlling temporal correlation
            dt (float): Time step for noise generation
            sigma (float): Scale of the noise
            episode_length (int): Maximum number of steps per episode
        """
        self.size = size
        self.beta = beta
        self.dt = dt
        self.sigma = sigma
        self.episode_length = episode_length
        self.eps = 1e-8  # Numerical stability constant
        self.reset()
    
    def reset(self):
        """Generates new noise sequences for a fresh episode."""
        self.noise_sequences = self._generate_colored_noise(
            n_steps=self.episode_length,
            n_dims=self.size
        )
        self.step_index = 0
        
    def _generate_colored_noise(self, n_steps, n_dims):
        """
        Generates colored noise using the FFT method.
        
        Args:
            n_steps (int): Number of time steps
            n_dims (int): Number of action dimensions
            
        Returns:
            ndarray: Colored noise array of shape (n_steps, n_dims)
        """
        # Base white noise generation
        white = np.random.normal(0., 1., (n_steps, n_dims))
        
        # Frequency domain setup
        freqs = np.fft.fftfreq(n_steps)
        freqs[0] = float('inf')  # Handle DC component
        
        # Apply power spectrum scaling
        Sf = np.abs(freqs)[:, np.newaxis] ** (-self.beta/2)
        Sf[0] = 0  # Remove DC component
        
        # FFT transform and apply spectrum
        white_fft = np.fft.fft(white, axis=0)
        colored_f = Sf * white_fft
        
        # Transform back to time domain
        colored = np.real(np.fft.ifft(colored_f, axis=0))
        
        # Normalize with numerical stability
        for d in range(n_dims):
            mean = np.mean(colored[:, d])
            std = np.std(colored[:, d])
            
            if std < self.eps:
                std = 1.0
                print(f"Warning: Using default scaling because of near-zero variance.")
            
            colored[:, d] = self.sigma * (colored[:, d] - mean) / std
        
        return colored
    
    def __call__(self):
        """Returns the current noise value and advances the step counter."""
        if self.step_index >= self.episode_length:
            self.reset()
            
        noise = self.noise_sequences[self.step_index]
        self.step_index += 1
        return noise

class TwinQFunction(nn.Module):
    """Twin Q-networks for TD3's double Q-learning."""
    def __init__(self, observation_dim, action_dim, hidden_sizes=[128, 128, 64],
                 learning_rate=0.0003, is_target=False):
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
        if self.is_target:
            raise ValueError("Cannot train target networks.")

        self.train()
        self.optimizer.zero_grad()

        # Q1, Q2 predictions
        q1_pred = self.Q1(torch.cat([observations, actions], dim=1))
        q2_pred = self.Q2(torch.cat([observations, actions], dim=1))

        # Compute loss for both Q-networks
        loss1 = self.loss_fn(q1_pred, targets)
        loss2 = self.loss_fn(q2_pred, targets)
        total_loss = loss1 + loss2

        # Backpropagation and optimization
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.Q1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.Q2.parameters(), max_norm=1.0)
        self.optimizer.step()

        return total_loss.item()

    def Q_value(self, observations, actions):
        return self.Q1(torch.cat([observations, actions], dim=1))

    def Q2_value(self, observations, actions):
        return self.Q2(torch.cat([observations, actions], dim=1))

class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.
    
    Implements key TD3 features:
    1. Twin Critics: Two Q-networks to reduce overestimation bias
    2. Delayed Policy Updates: Updates policy less frequently than critics
    3. Target Policy Smoothing: Adds noise to target actions
    4. Colored Noise Exploration: Uses temporally correlated noise for exploration
    """
    
    def __init__(self, observation_space, action_space, **config):
        """
        Args:
            observation_space (gym.Space): Environment observation space
            action_space (gym.Space): Environment action space
            **config: Configuration parameters
        """
        # Validate input spaces
        if not isinstance(observation_space, spaces.Box):
            raise UnsupportedSpace("Observation space must be Box.")
        if not isinstance(action_space, spaces.Box):
            raise UnsupportedSpace("Action space must be Box.")

        # Store space dimensions
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.action_space = action_space

        # Store action bounds on device
        self.action_low = torch.from_numpy(action_space.low).to(DEVICE)
        self.action_high = torch.from_numpy(action_space.high).to(DEVICE)

        # Default configuration
        self.config = {
            # Exploration
            "eps": 0.1,                # Exploration noise scale
            "noise_beta": 1.0,         # Noise color (0=white, 1=pink, 2=red)
            "warmup_steps": 25000,     # Number of random steps for initial exploration

            # TD3 specific
            "policy_noise": 0.2,       # Target policy smoothing noise
            "noise_clip": 0.5,         # Noise clipping range
            "policy_delay": 2,         # Policy update frequency
            "polyak": 0.995,           # Target network update rate
            "discount": 0.99,          # Reward discount factor

            # Architecture
            "buffer_size": int(1e6),
            "batch_size": 256,         # Updated to 256 as per user config
            "learning_rate_actor": 0.0003,   # Actor learning rate
            "learning_rate_critic": 0.0003,  # Critic learning rate
            "hidden_sizes_actor": [128,128],
            "hidden_sizes_critic": [128,128,64],
            "use_target_net": True,
            "max_steps": 2000
        }
        self.config.update(config)

        # Initialize components
        self._setup_noise()
        self._setup_memory()
        self._setup_networks()
        self.train_iter = 0

    def _setup_noise(self):
        """Initializes exploration noise process."""
        self.noise = ColoredNoiseProcess(
            size=self.act_dim,
            beta=self.config['noise_beta'],
            sigma=self.config['eps'],
            episode_length=self.config['max_steps']
        )

    def _setup_memory(self):
        """Initializes replay buffer."""
        self.buffer = Memory(max_size=self.config["buffer_size"])

    def _setup_networks(self):
        """Initializes actor and critic networks."""
        # Setup critic networks (twin Q-networks)
        self.Q = TwinQFunction(
            observation_dim=self.obs_dim,
            action_dim=self.act_dim,
            hidden_sizes=self.config["hidden_sizes_critic"],
            learning_rate=self.config["learning_rate_critic"],
            is_target=False
        ).to(DEVICE)

        # Setup target critic networks without optimizer
        self.Q_target = TwinQFunction(
            observation_dim=self.obs_dim,
            action_dim=self.act_dim,
            hidden_sizes=self.config["hidden_sizes_critic"],
            learning_rate=0,
            is_target=True
        ).to(DEVICE)

        # Setup actor network
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

        # Setup target actor network without optimizer
        self.policy_target = Feedforward(
            input_size=self.obs_dim,
            hidden_sizes=self.config["hidden_sizes_actor"],
            output_size=self.act_dim,
            activation_fun=nn.ReLU(),
            output_activation=output_activation
        ).to(DEVICE)

        # Initialize target networks
        self._copy_nets()

        # Initialize separate optimizer for actor only
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config["learning_rate_actor"],
            eps=1e-6
        )

    def act(self, observation, eps=None):
        """
        Selects action using the current policy and exploration noise.
        
        Args:
            observation: Environment observation
            eps: Optional override for exploration noise scale
            
        Returns:
            ndarray: Selected action
        """
        self.policy.eval()
        obs_tensor = torch.from_numpy(observation.astype(np.float32)).to(DEVICE)
        with torch.no_grad():
            action = self.policy(obs_tensor).cpu().numpy()
        
        # Add exploration noise if not evaluating
        if eps is not None:
            self.noise.sigma = eps
        if eps != 0:  # Only add noise during training
            noise = self.noise()
            action = np.clip(
                action + noise,
                self.action_space.low,
                self.action_space.high
            )
        
        return action

    def _soft_update(self, target, source):
        """
        Updates target network parameters using Polyak averaging.
        
        θ_target = ρ * θ_target + (1 - ρ) * θ_source
        
        Args:
            target: Target network
            source: Source network
        """
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    self.config["polyak"] * target_param.data +
                    (1.0 - self.config["polyak"]) * param.data
                )

    def train(self, num_updates=32):
        """
        Performs training iterations using sampled transitions.

        Args:
            num_updates (int): Number of training iterations

        Returns:
            list: Training losses
        """
        losses = []
        self.train_iter += 1

        for i in range(num_updates):
            # Sample and prepare batch
            data = self.buffer.sample(batch=self.config['batch_size'])
            if data is None:
                break

            states = torch.tensor(np.vstack(data[:, 0]), dtype=torch.float32, device=DEVICE)
            actions = torch.tensor(np.vstack(data[:, 1]), dtype=torch.float32, device=DEVICE)
            rewards = torch.tensor(np.vstack(data[:, 2]), dtype=torch.float32, device=DEVICE)
            next_states = torch.tensor(np.vstack(data[:, 3]), dtype=torch.float32, device=DEVICE)
            dones = torch.tensor(np.vstack(data[:, 4]), dtype=torch.float32, device=DEVICE)

            # Normalize rewards
            r_mean = rewards.mean()
            r_std = rewards.std() + 1e-8
            r_normalized = (rewards - r_mean) / r_std

            # Compute target Q-values
            with torch.no_grad():
                # Add clipped noise to target actions (policy smoothing)
                noise = torch.randn_like(actions) * self.config["policy_noise"]
                noise = torch.clamp(
                    noise,
                    -self.config["noise_clip"],
                    self.config["noise_clip"]
                )
                
                next_action = self.policy_target(next_states) + noise
                next_action = torch.clamp(
                    next_action,
                    self.action_low,
                    self.action_high
                )

                # Use minimum of twin Q-values
                q1_next = self.Q_target.Q_value(next_states, next_action)
                q2_next = self.Q_target.Q2_value(next_states, next_action)
                q_next = torch.min(q1_next, q2_next)
                
                # Compute targets
                q_target = r_normalized + self.config["discount"] * (1 - dones) * q_next

            # Update critic
            q_target = q_target.view(-1, 1)
            q_loss = self.Q.fit(states, actions, q_target)
            losses.append(q_loss)

            # Delayed policy updates
            if i % self.config["policy_delay"] == 0:
                # Update actor using actor_optimizer
                self.policy.train()
                a_pred = self.policy(states)
                # Q_value returns shape
                actor_loss = -self.Q.Q_value(states, a_pred).mean()

                self.policy_optimizer.zero_grad()
                actor_loss.backward()
                # Implement gradient clipping for actor
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.policy_optimizer.step()
                losses.append(actor_loss.item())

                # Update target networks
                if self.config["use_target_net"]:
                    self._soft_update(self.Q_target, self.Q)
                    self._soft_update(self.policy_target, self.policy)

        return losses

    def _copy_nets(self):
        """Performs hard update of target networks (used for initialization)."""
        self.Q_target.Q1.load_state_dict(self.Q.Q1.state_dict())
        self.Q_target.Q2.load_state_dict(self.Q.Q2.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    def store_transition(self, transition):
        """Stores transition in replay buffer."""
        self.buffer.add_transition(transition)

    def state(self):
        """Returns current agent state for saving."""
        return {
            "Q_state": self.Q.state_dict(),
            "policy_state": self.policy.state_dict()
        }

    def restore_state(self, state):
        """Restores agent state from saved state."""
        self.Q.load_state_dict(state["Q_state"])
        self.policy.load_state_dict(state["policy_state"])
        self._copy_nets()

    def reset_noise(self):
        """Resets exploration noise process for new episode."""
        self.noise.reset()

def main():
    """Main training script."""
    # Parse command line arguments
    parser = optparse.OptionParser()
    parser.add_option('-e', '--env', type='string', dest='env_name',
                     default="Pendulum-v1", help='Environment name')
    parser.add_option('-n', '--eps', type='float', dest='eps',
                     default=0.1, help='Exploration noise scale')
    parser.add_option('-t', '--train', type='int', dest='train',
                     default=8, help='Training iterations per episode')
    parser.add_option('-l', '--lr_actor', type='float', dest='lr_actor',
                     default=0.0003, help='Actor learning rate')
    parser.add_option('-c', '--lr_critic', type='float', dest='lr_critic',
                     default=0.0003, help='Critic learning rate')
    parser.add_option('-m', '--maxepisodes', type='float', dest='max_episodes',
                     default=2000, help='Maximum training episodes')
    parser.add_option('-s', '--seed', type='int', dest='seed',
                     default=42, help='Random seed')
    parser.add_option('-w', '--warmup', type='int', dest='warmup_steps',
                     default=25000, help='Warm-up steps with random actions')
    
    opts, _ = parser.parse_args()

    # Environment setup
    env_name = opts.env_name
    env = gym.make(env_name)

    # Training parameters
    max_episodes = int(opts.max_episodes)
    max_timesteps = env.spec.max_episode_steps if hasattr(env.spec, 'max_episode_steps') else 2000
    train_iter = opts.train
    log_interval = 20
    render = False

    # Set random seeds
    if opts.seed is not None:
        torch.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        env.action_space.seed(opts.seed)
        env.observation_space.seed(opts.seed)

    # Initialize agent
    agent = TD3Agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        eps=opts.eps,
        learning_rate_actor=opts.lr_actor,
        learning_rate_critic=opts.lr_critic,
        warmup_steps=opts.warmup_steps
    )

    # Training metrics
    rewards = []
    lengths = []
    losses = []
    timestep = 0

    # Warm-up period with random actions
    print("Starting warm-up period with random actions...")
    ob, _ = env.reset()
    for _ in range(opts.warmup_steps):
        action = env.action_space.sample()
        ob_new, reward, done, trunc, _ = env.step(action)
        agent.store_transition((ob, action, reward, ob_new, done))
        ob = ob_new if not (done or trunc) else env.reset()[0]
    print(f"Warm-up complete. Collected {opts.warmup_steps} random transitions.")

    # Training loop
    for i_episode in range(1, max_episodes + 1):
        ob, _info = env.reset()
        agent.reset_noise()
        total_reward = 0

        # Episode loop
        for t in range(max_timesteps):
            timestep += 1
            if render:
                env.render()

            # Select and execute action
            action = agent.act(ob)
            ob_new, reward, done, trunc, _info = env.step(action)
            total_reward += reward

            # Store experience
            agent.store_transition((ob, action, reward, ob_new, done))
            
            if done or trunc:
                break
            ob = ob_new

        # Train after each episode
        losses_batch = agent.train(train_iter)
        losses.extend(losses_batch)
        rewards.append(total_reward)
        lengths.append(t)

        # Periodic saving
        if i_episode % 500 == 0:
            print("########## Saving checkpoint... ##########")
            ckpt_path = Path("results") / f"TD3_{env_name}_{i_episode}-eps{opts.eps}-t{train_iter}-l{opts.lr_actor}-s{opts.seed}.pth"
            ckpt_path.parent.mkdir(exist_ok=True)
            torch.save(agent.state(), ckpt_path)
            save_statistics(i_episode)

        # Logging
        if i_episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))
            avg_loss = np.mean(losses[-log_interval:]) if len(losses) >= log_interval else np.mean(losses)
            print(f"Episode {i_episode}\t "
                  f"avg length: {avg_length}\t "
                  f"avg reward: {avg_reward:.2f}\t "
                  f"avg loss: {avg_loss:.4f}")

    # Final save
    save_statistics(i_episode)
    env.close()

if __name__ == '__main__':
    main()
