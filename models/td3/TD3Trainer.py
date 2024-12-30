import os
import json
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
from datetime import datetime
from .TD3 import TD3Agent

class TD3Trainer:
    def __init__(self, env_name, training_config, model_config, experiment_path):
        self.env_name = env_name
        self.training_config = training_config
        self.model_config = model_config
        self.experiment_path = Path(experiment_path)
        
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
        
        # Create directories for saving
        self.model_dir = self.experiment_path / 'models'
        self.model_dir.mkdir(exist_ok=True)
        
        self.log_dir = self.experiment_path / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        
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
        episode_rewards = []
        best_eval_reward = float('-inf')
        metrics = {'train_rewards': [], 'eval_rewards': [], 'losses': []}

        for episode in range(self.training_config['max_episodes']):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_losses = []

            for t in range(self.max_steps):
                # Select action and add exploration noise
                action = self.agent.act(state)
                
                # Execute action
                next_state, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward
                
                # Store transition
                self.agent.store_transition((state, action, reward, next_state, done))
                
                # Train agent
                if self.agent.buffer.size >= self.model_config['batch_size']:
                    losses = self.agent.train(iter_fit=self.training_config['train_iter'])
                    episode_losses.extend(losses)
                
                if done or truncated:
                    break
                    
                state = next_state

            # Log training progress
            episode_rewards.append(episode_reward)
            metrics['train_rewards'].append(episode_reward)
            metrics['losses'].extend(episode_losses)
            
            # Evaluate and log every eval_interval episodes
            if (episode + 1) % self.training_config['eval_interval'] == 0:
                eval_reward = self.evaluate_policy()
                metrics['eval_rewards'].append(eval_reward)
                
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.save_checkpoint(episode + 1, metrics)
                
                print(f"Episode {episode + 1}: Train reward: {episode_reward:.2f}, Eval reward: {eval_reward:.2f}")
            
            # Regular progress logging
            elif (episode + 1) % self.training_config['log_interval'] == 0:
                print(f"Episode {episode + 1}: Train reward: {episode_reward:.2f}")
            
            # Save checkpoint every save_interval episodes
            if (episode + 1) % self.training_config['save_interval'] == 0:
                self.save_checkpoint(episode + 1, metrics)

        # Save final checkpoint
        self.save_checkpoint(self.training_config['max_episodes'], metrics)
        
        # Close environments
        self.env.close()
        self.eval_env.close()
        
        return metrics
