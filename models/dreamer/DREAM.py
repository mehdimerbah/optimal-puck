"""
DREAM.py

Defines DreamerV3Agent for a (Box,Box) environment (like HockeyEnv).
"""

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces

from models.baseline.memory import Memory
from models.baseline.feedforward import Feedforward  # or your MLP class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

class UnsupportedSpace(Exception):
    """Exception for unsupported action/observation spaces."""
    pass

class WorldModel(nn.Module):
    """
    Simplified world model:
      - encoder: obs -> latent
      - forward: (latent, action) -> (next_latent, reward, done_logit)
    """
    def __init__(self, obs_dim, act_dim, latent_dim=32, hidden_size=64):
        super().__init__()
        self.encoder = Feedforward(
            input_size=obs_dim,
            hidden_sizes=[hidden_size],
            output_size=latent_dim,
            activation_fun=nn.ReLU()
        )
        self.transition = Feedforward(
            input_size=latent_dim + act_dim,
            hidden_sizes=[hidden_size],
            output_size=latent_dim,
            activation_fun=nn.ReLU()
        )
        self.reward_head = Feedforward(
            input_size=latent_dim + act_dim,
            hidden_sizes=[hidden_size],
            output_size=1,
            activation_fun=nn.ReLU()
        )
        self.done_head = Feedforward(
            input_size=latent_dim + act_dim,
            hidden_sizes=[hidden_size],
            output_size=1,
            activation_fun=nn.ReLU()
        )

    def encode_observation(self, obs):
        return self.encoder(obs)

    def forward(self, latent, action):
        x = torch.cat([latent, action], dim=1)
        next_latent = self.transition(x)
        reward = self.reward_head(x)
        done_logit = self.done_head(x)
        return next_latent, reward, done_logit

class Actor(nn.Module):
    """
    Actor network -> action in [-1, 1]. We'll rescale externally.
    """
    def __init__(self, latent_dim, act_dim, hidden_size=64):
        super().__init__()
        self.net = Feedforward(
            input_size=latent_dim,
            hidden_sizes=[hidden_size, hidden_size],
            output_size=act_dim,
            activation_fun=nn.ReLU(),
            output_activation=nn.Tanh()
        )

    def forward(self, latent):
        return self.net(latent)

class Critic(nn.Module):
    """
    Critic -> value estimate V(latent).
    """
    def __init__(self, latent_dim, hidden_size=64):
        super().__init__()
        self.net = Feedforward(
            input_size=latent_dim,
            hidden_sizes=[hidden_size, hidden_size],
            output_size=1,
            activation_fun=nn.ReLU()
        )

    def forward(self, latent):
        return self.net(latent)

class DreamerV3Agent:
    def __init__(self, observation_space, action_space, **config):
        if not isinstance(observation_space, spaces.Box):
            raise UnsupportedSpace("Observation space must be Box.")
        if not isinstance(action_space, spaces.Box):
            raise UnsupportedSpace("Action space must be Box.")

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        defaults = {
            "latent_dim": 32,
            "wm_hidden_size": 64,
            "actor_hidden_size": 64,
            "critic_hidden_size": 64,
            "world_model_lr": 1e-3,
            "actor_lr": 1e-4,
            "critic_lr": 1e-4,
            "discount": 0.99,
            "buffer_size": 1e5,
            "batch_size": 64,
            "grad_updates_per_step": 1,
        }
        defaults.update(config)
        self.config = defaults

        self.world_model = WorldModel(
            obs_dim=obs_dim,
            act_dim=act_dim,
            latent_dim=self.config["latent_dim"],
            hidden_size=self.config["wm_hidden_size"]
        ).to(device)

        self.actor = Actor(
            latent_dim=self.config["latent_dim"],
            act_dim=act_dim,
            hidden_size=self.config["actor_hidden_size"]
        ).to(device)

        self.critic = Critic(
            latent_dim=self.config["latent_dim"],
            hidden_size=self.config["critic_hidden_size"]
        ).to(device)

        # optimizers
        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=float(self.config["world_model_lr"]))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(self.config["actor_lr"]))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(self.config["critic_lr"]))

        self.discount = self.config["discount"]
        self.batch_size = int(self.config["batch_size"])

        self.buffer = Memory(max_size=int(self.config["buffer_size"]))

        # For rescaling actions from [-1,1] to [low,high]
        high = torch.tensor(action_space.high, dtype=torch.float32, device=device)
        low  = torch.tensor(action_space.low, dtype=torch.float32, device=device)
        self.action_scale = (high - low) / 2.0
        self.action_bias  = (high + low) / 2.0

    def reset(self):
        pass  # For RNN states etc. Not used here.

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            latent = self.world_model.encode_observation(obs_t)
            a_t = self.actor(latent)
            # rescale to real environment bounds
            a_t = a_t * self.action_scale + self.action_bias
        return a_t.cpu().numpy()[0]

    def store_transition(self, transition: tuple):
        self.buffer.add_transition(transition)

    def train(self, num_updates=32):
        """
        Performs num_updates gradient steps. Each step:
          1) Train world model
          2) Train critic
          3) Train actor
        Returns a list of (wm_loss, actor_loss, critic_loss)
        """
        import torch.nn.functional as F

        losses_list = []
        for _ in range(num_updates):
            batch = self.buffer.sample(batch=self.batch_size)
            if batch is None:
                break

            states  = torch.tensor(np.vstack(batch[:, 0]), dtype=torch.float32, device=device)
            actions = torch.tensor(np.vstack(batch[:, 1]), dtype=torch.float32, device=device)
            rewards = torch.tensor(np.vstack(batch[:, 2]), dtype=torch.float32, device=device)
            next_states = torch.tensor(np.vstack(batch[:, 3]), dtype=torch.float32, device=device)
            dones   = torch.tensor(np.vstack(batch[:, 4]), dtype=torch.float32, device=device)

            # 1) World model
            latents = self.world_model.encode_observation(states)
            pred_next_latent, pred_reward, pred_done = self.world_model(latents, actions)
            with torch.no_grad():
                next_latents = self.world_model.encode_observation(next_states)

            loss_latent = F.mse_loss(pred_next_latent, next_latents)
            loss_reward = F.mse_loss(pred_reward, rewards)
            loss_done   = F.binary_cross_entropy_with_logits(pred_done, dones)
            wm_loss = loss_latent + loss_reward + loss_done

            self.wm_optimizer.zero_grad()
            wm_loss.backward()
            self.wm_optimizer.step()

            # 2) Critic
            latents_critic = self.world_model.encode_observation(states).detach()
            next_latents_critic = self.world_model.encode_observation(next_states).detach()

            value = self.critic(latents_critic)
            with torch.no_grad():
                target_value = rewards + (1 - dones) * self.discount * self.critic(next_latents_critic)

            critic_loss = F.mse_loss(value, target_value)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # 3) Actor
            latents_actor = self.world_model.encode_observation(states).detach()
            imagined_actions = self.actor(latents_actor)
            imagined_next_latent, _, _ = self.world_model(latents_actor, imagined_actions)
            actor_loss = - self.critic(imagined_next_latent).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            losses_list.append((wm_loss.item(), actor_loss.item(), critic_loss.item()))
        return losses_list

    def state(self) -> dict:
        """
        Return state dict for checkpointing.
        """
        return {
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }

    def restore_state(self, state: dict):
        self.world_model.load_state_dict(state["world_model"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
