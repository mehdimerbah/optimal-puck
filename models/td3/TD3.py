import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import optparse
import pickle

from ..baseline.memory import Memory
from ..baseline.feedforward import Feedforward

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

class UnsupportedSpace(Exception):
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)

# Similar Q-network as DDPG but we need two for TD3
class TwinQFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100,100],
                 learning_rate=0.0002):
        input_size = observation_dim + action_dim
        output_size = 1
        super().__init__(input_size=input_size, 
                        hidden_sizes=hidden_sizes, 
                        output_size=output_size)
        
        # Second Q-network for TD3's twin delayed update
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

class TD3Agent(object):
    def __init__(self, observation_space, action_space, **userconfig):
        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space incompatible')
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace('Action space incompatible')

        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_space = action_space
        self._action_n = action_space.shape[0]
        
        # TD3 specific configurations
        self._config = {
            "eps": 0.1,                # Exploration noise
            "policy_noise": 0.2,       # Target policy smoothing noise
            "noise_clip": 0.5,         # Noise clipping range
            "policy_delay": 2,         # Delayed policy updates
            "polyak": 0.995,           # Polyak coefficient
            "discount": 0.99,
            "buffer_size": int(1e6),
            "batch_size": 128,
            "learning_rate_actor": 0.00001,
            "learning_rate_critic": 0.0001,
            "hidden_sizes_actor": [128,128],
            "hidden_sizes_critic": [128,128,64],
            "use_target_net": True
        }
        self._config.update(userconfig)
        self._eps = self._config['eps']

        self.buffer = Memory(max_size=self._config["buffer_size"])

        # Twin Q Networks
        self.Q = TwinQFunction(observation_dim=self._obs_dim,
                              action_dim=self._action_n,
                              hidden_sizes=self._config["hidden_sizes_critic"],
                              learning_rate=self._config["learning_rate_critic"])
        
        # Target Q Networks
        self.Q_target = TwinQFunction(observation_dim=self._obs_dim,
                                     action_dim=self._action_n,
                                     hidden_sizes=self._config["hidden_sizes_critic"],
                                     learning_rate=0)

        # Actor network with same structure as DDPG
        high, low = torch.from_numpy(self._action_space.high), torch.from_numpy(self._action_space.low)
        output_activation = lambda x: (torch.tanh(x) * (high - low) / 2) + (high + low) / 2

        self.policy = Feedforward(input_size=self._obs_dim,
                                hidden_sizes=self._config["hidden_sizes_actor"],
                                output_size=self._action_n,
                                activation_fun=torch.nn.ReLU(),
                                output_activation=output_activation)
        
        self.policy_target = Feedforward(input_size=self._obs_dim,
                                       hidden_sizes=self._config["hidden_sizes_actor"],
                                       output_size=self._action_n,
                                       activation_fun=torch.nn.ReLU(),
                                       output_activation=output_activation)

        self._copy_nets()
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                        lr=self._config["learning_rate_actor"],
                                        eps=0.000001)
        self.train_iter = 0

    def act(self, observation, eps=None):
        self.policy.eval()
        obs_tensor = torch.from_numpy(observation.astype(np.float32)).to(device)
        action = self.policy(obs_tensor).detach().cpu().numpy()
        
        # Add exploration noise
        noise = np.random.normal(0, self._eps if eps is None else eps, 
                               size=self._action_n)
        return np.clip(action + noise, 
                      self._action_space.low, 
                      self._action_space.high)

    def _soft_update(self, target, source):
        """Soft update target network parameters using Polyak averaging."""
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    self._config["polyak"] * target_param.data + 
                    (1.0 - self._config["polyak"]) * param.data
                )

    def train(self, iter_fit=32):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32)).to(device)
        losses = []
        self.train_iter += 1

        for i in range(iter_fit):
            # Sample from replay buffer
            data = self.buffer.sample(batch=self._config['batch_size'])
            s = to_torch(np.stack(data[:, 0]))
            a = to_torch(np.stack(data[:, 1]))
            r = to_torch(np.stack(data[:, 2])[:, None])
            s_next = to_torch(np.stack(data[:, 3]))
            done = to_torch(np.stack(data[:, 4])[:, None])

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = torch.randn_like(a) * self._config["policy_noise"]
                noise = torch.clamp(noise, 
                                  -self._config["noise_clip"], 
                                  self._config["noise_clip"])
                
                next_action = self.policy_target(s_next) + noise
                next_action = torch.clamp(next_action, 
                                        torch.from_numpy(self._action_space.low), 
                                        torch.from_numpy(self._action_space.high))

                # Get minimum Q-value between two target critics
                q1_next = self.Q_target.Q_value(s_next, next_action)
                q2_next = self.Q_target.Q2_value(s_next, next_action)
                q_next = torch.min(q1_next, q2_next)
                
                # Compute target Q-value
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
                
                if self._config["use_target_net"]:
                    # Soft update target networks after policy update
                    self._soft_update(self.Q_target, self.Q)
                    self._soft_update(self.policy_target, self.policy)

        return losses

    def _copy_nets(self):
        """Hard update target networks (only used for initialization)."""
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def state(self):
        return (self.Q.state_dict(), self.policy.state_dict())

    def restore_state(self, state):
        self.Q.load_state_dict(state[0])
        self.policy.load_state_dict(state[1])
        self._copy_nets()

def main():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env', action='store', type='string',
                         dest='env_name', default="Pendulum-v1",
                         help='Environment (default %default)')
    optParser.add_option('-n', '--eps', action='store',  type='float',
                         dest='eps', default=0.1,
                         help='Policy exploration noise (default %default)')
    optParser.add_option('-t', '--train', action='store', type='int',
                         dest='train', default=32,
                         help='Number of training batches per episode (default %default)')
    optParser.add_option('-l', '--lr', action='store', type='float',
                         dest='lr', default=0.0001,
                         help='Learning rate for actor/policy (default %default)')
    optParser.add_option('-m', '--maxepisodes', action='store', type='float',
                         dest='max_episodes', default=2000,
                         help='Number of episodes (default %default)')
    optParser.add_option('-s', '--seed', action='store',  type='int',
                         dest='seed', default=42,
                         help='Random seed (default %default)')
    opts, args = optParser.parse_args()

    ################## Hyperparameters ##################
    env_name        = opts.env_name
    if env_name == "LunarLander-v2":
        env = gym.make(env_name, continuous=True)
    else:
        env = gym.make(env_name)

    render           = False
    log_interval     = 20            # print avg reward every N episodes
    max_episodes     = int(opts.max_episodes)  # max training episodes
    max_timesteps    = 2000          # max timesteps in one episode

    train_iter       = opts.train    # number of mini-batch updates after each episode
    eps              = opts.eps      # exploration noise
    lr               = opts.lr       # learning rate for the actor
    random_seed      = opts.seed
    ####################################################

    # Set random seeds for reproducibility
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        env.action_space.seed(random_seed)
        env.observation_space.seed(random_seed)

    # Instantiate TD3 agent
    td3 = TD3Agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        eps=eps,
        learning_rate_actor=lr
    )

    # Logging variables
    rewards = []
    lengths = []
    losses = []
    timestep = 0

    def save_statistics(i_episode):
        save_path = f"./results/TD3_{env_name}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}-stat.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump({
                "rewards": rewards,
                "lengths": lengths,
                "eps": eps,
                "train_iter": train_iter,
                "lr": lr,
                "losses": losses,
                "episode": i_episode
            }, f)

    # Training loop
    for i_episode in range(1, max_episodes+1):
        ob, _info = env.reset()
        total_reward = 0

        for t in range(max_timesteps):
            timestep += 1
            if render:
                env.render()

            action = td3.act(ob)
            (ob_new, reward, done, trunc, _info) = env.step(action)
            total_reward += reward

            # Store transition in replay buffer
            td3.store_transition((ob, action, reward, ob_new, done))

            ob = ob_new
            if done or trunc:
                break

        # After each episode, run some training iterations
        losses_batch = td3.train(train_iter)
        losses.extend(losses_batch)

        rewards.append(total_reward)
        lengths.append(t)

        # Save a checkpoint every 500 episodes
        if i_episode % 500 == 0:
            print("########## Saving a checkpoint... ##########")
            ckpt_path = f'./results/TD3_{env_name}_{i_episode}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}.pth'
            torch.save(td3.state(), ckpt_path)
            save_statistics(i_episode)

        # Logging
        if i_episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))
            print(
                f"Episode {i_episode}\t "
                f"avg length: {avg_length}\t "
                f"avg reward: {avg_reward:.2f}"
            )

    # Final save
    save_statistics(i_episode)
    env.close()

if __name__ == '__main__':
    main()
