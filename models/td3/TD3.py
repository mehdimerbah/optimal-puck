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
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import optparse
import pickle
from pathlib import Path

from ..baseline.memory import Memory
from ..baseline.feedforward import Feedforward

# Global constants
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

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
            
            colored[:, d] = (colored[:, d] - mean) / (std + self.eps)
        
        return colored * self.sigma
    
    def sample(self):
        """Returns the next noise vector in the sequence."""
        if self.step_index >= self.episode_length:
            self.reset()
        
        noise = self.noise_sequences[self.step_index]
        self.step_index += 1
        return noise
    
    def __call__(self):
        """Convenience method for sampling."""
        return self.sample()

class TwinQFunction(Feedforward):
    """Twin Q-networks for TD3's double Q-learning."""
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100,100],
                 learning_rate=0.0002):
        input_size = observation_dim + action_dim
        output_size = 1
        super().__init__(input_size=input_size, 
                        hidden_sizes=hidden_sizes, 
                        output_size=output_size)
        
        # Second Q-network
        self.Q2 = Feedforward(input_size=input_size,
                            hidden_sizes=hidden_sizes,
                            output_size=output_size)
        
        self.optimizer = torch.optim.Adam(list(self.parameters()) + 
                                        list(self.Q2.parameters()),
                                        lr=learning_rate,
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss()

    def fit(self, observations, actions, targets):
        self.train()
        self.optimizer.zero_grad()
        
        # Get Q-values from both networks
        pred1 = self.Q_value(observations, actions)
        pred2 = self.Q2_value(observations, actions)
        
        # Compute loss for both Q-networks
        loss1 = self.loss(pred1, targets)
        loss2 = self.loss(pred2, targets)
        total_loss = loss1 + loss2
        
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    def Q_value(self, observations, actions):
        x = torch.cat([observations, actions], dim=1)
        return self.forward(x)

    def Q2_value(self, observations, actions):
        x = torch.cat([observations, actions], dim=1)
        return self.Q2(x)

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
        if not isinstance(observation_space, spaces.box.Box):
            raise ValueError('Observation space must be continuous (Box)')
        if not isinstance(action_space, spaces.box.Box):
            raise ValueError('Action space must be continuous (Box)')

        # Store space dimensions
        self._obs_dim = observation_space.shape[0]
        self._action_n = action_space.shape[0]
        self._action_space = action_space
        
        # Default configuration
        self._config = {
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
            "batch_size": 128,
            "learning_rate_actor": 0.00001,
            "learning_rate_critic": 0.0001,
            "hidden_sizes_actor": [128,128],
            "hidden_sizes_critic": [128,128,64],
            "use_target_net": True,
            "max_steps": 2000
        }
        self._config.update(config)
        
        # Initialize components
        self._setup_noise()
        self._setup_memory()
        self._setup_networks()
        self._setup_training()
    
    def _setup_noise(self):
        """Initializes exploration noise process."""
        self.noise = ColoredNoiseProcess(
            size=self._action_n,
            beta=self._config['noise_beta'],
            sigma=self._config['eps'],
            episode_length=self._config['max_steps']
        )
    
    def _setup_memory(self):
        """Initializes replay buffer."""
        self.buffer = Memory(max_size=self._config["buffer_size"])
    
    def _setup_networks(self):
        """Initializes actor and critic networks."""
        # Setup critic networks (twin Q-networks)
        self.Q = TwinQFunction(
            observation_dim=self._obs_dim,
            action_dim=self._action_n,
            hidden_sizes=self._config["hidden_sizes_critic"],
            learning_rate=self._config["learning_rate_critic"]
        )
        
        self.Q_target = TwinQFunction(
            observation_dim=self._obs_dim,
            action_dim=self._action_n,
            hidden_sizes=self._config["hidden_sizes_critic"],
            learning_rate=0
        )
        
        # Setup actor network
        high = torch.from_numpy(self._action_space.high)
        low = torch.from_numpy(self._action_space.low)
        output_activation = lambda x: (torch.tanh(x) * (high - low) / 2) + (high + low) / 2
        
        self.policy = Feedforward(
            input_size=self._obs_dim,
            hidden_sizes=self._config["hidden_sizes_actor"],
            output_size=self._action_n,
            activation_fun=torch.nn.ReLU(),
            output_activation=output_activation
        )
        
        self.policy_target = Feedforward(
            input_size=self._obs_dim,
            hidden_sizes=self._config["hidden_sizes_actor"],
            output_size=self._action_n,
            activation_fun=torch.nn.ReLU(),
            output_activation=output_activation
        )
        
        # Initialize target networks
        self._copy_nets()
    
    def _setup_training(self):
        """Initializes training-related components."""
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self._config["learning_rate_actor"],
            eps=0.000001
        )
        self.train_iter = 0
    
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
        action = self.policy(obs_tensor).detach().cpu().numpy()
        
        # Add exploration noise
        if eps is not None:
            self.noise.sigma = eps
        noise = self.noise()
        
        return np.clip(
            action + noise,
            self._action_space.low,
            self._action_space.high
        )

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
                    self._config["polyak"] * target_param.data +
                    (1.0 - self._config["polyak"]) * param.data
                )

    def train(self, iter_fit=32):
        """
        Performs training iterations using sampled transitions.
        
        Args:
            iter_fit (int): Number of training iterations
            
        Returns:
            list: Training losses
        """
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32)).to(DEVICE)
        losses = []

        for i in range(iter_fit):
            # Sample and prepare batch
            data = self.buffer.sample(batch=self._config['batch_size'])
            s = to_torch(np.stack(data[:, 0]))
            a = to_torch(np.stack(data[:, 1]))
            r = to_torch(np.stack(data[:, 2])[:, None])
            s_next = to_torch(np.stack(data[:, 3]))
            done = to_torch(np.stack(data[:, 4])[:, None])

            # Compute target Q-values
            with torch.no_grad():
                # Add clipped noise to target actions (policy smoothing)
                noise = torch.randn_like(a) * self._config["policy_noise"]
                noise = torch.clamp(
                    noise,
                    -self._config["noise_clip"],
                    self._config["noise_clip"]
                )
                
                next_action = self.policy_target(s_next) + noise
                next_action = torch.clamp(
                    next_action,
                    torch.from_numpy(self._action_space.low),
                    torch.from_numpy(self._action_space.high)
                )

                # Use minimum of twin Q-values
                q1_next = self.Q_target.Q_value(s_next, next_action)
                q2_next = self.Q_target.Q2_value(s_next, next_action)
                q_next = torch.min(q1_next, q2_next)
                
                # Compute targets
                q_target = r + self._config["discount"] * (1 - done) * q_next

            # Update critic
            q_loss = self.Q.fit(s, a, q_target)
            losses.append(q_loss)

            # Delayed policy updates
            if i % self._config["policy_delay"] == 0:
                # Update actor
                a_pred = self.policy(s)
                actor_loss = -self.Q.Q_value(s, a_pred).mean()
                
                self.optimizer.zero_grad()
                actor_loss.backward()
                self.optimizer.step()
                losses.append(actor_loss.item())
                
                # Update target networks
                if self._config["use_target_net"]:
                    self._soft_update(self.Q_target, self.Q)
                    self._soft_update(self.policy_target, self.policy)

        return losses

    def _copy_nets(self):
        """Performs hard update of target networks (used for initialization)."""
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    def store_transition(self, transition):
        """Stores transition in replay buffer."""
        self.buffer.add_transition(transition)

    def state(self):
        """Returns current agent state for saving."""
        return (self.Q.state_dict(), self.policy.state_dict())

    def restore_state(self, state):
        """Restores agent state from saved state."""
        self.Q.load_state_dict(state[0])
        self.policy.load_state_dict(state[1])
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
                     default=32, help='Training iterations per episode')
    parser.add_option('-l', '--lr', type='float', dest='lr',
                     default=0.0001, help='Actor learning rate')
    parser.add_option('-m', '--maxepisodes', type='float', dest='max_episodes',
                     default=2000, help='Maximum training episodes')
    parser.add_option('-s', '--seed', type='int', dest='seed',
                     default=42, help='Random seed')
    parser.add_option('-w', '--warmup', type='int', dest='warmup_steps',
                     default=25000, help='Warm-up steps with random actions')
    
    opts, _ = parser.parse_args()

    # Environment setup
    env_name = opts.env_name
    env = gym.make(env_name, continuous=True) if env_name == "LunarLander-v2" else gym.make(env_name)

    # Training parameters
    max_episodes = int(opts.max_episodes)
    max_timesteps = 2000
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
        learning_rate_actor=opts.lr,
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
            ckpt_path = Path("results") / f"TD3_{env_name}_{i_episode}-eps{opts.eps}-t{train_iter}-l{opts.lr}-s{opts.seed}.pth"
            ckpt_path.parent.mkdir(exist_ok=True)
            torch.save(agent.state(), ckpt_path)
            save_statistics(i_episode)

        # Logging
        if i_episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))
            print(f"Episode {i_episode}\t "
                  f"avg length: {avg_length}\t "
                  f"avg reward: {avg_reward:.2f}")

    # Final save
    save_statistics(i_episode)
    env.close()

if __name__ == '__main__':
    main()
