from models.ddpg import DDPGAgent
import gymnasium as gym
import numpy as np
import torch




class DDPGTrainer:
    def __init__(self, env_name, training_config, model_config, experiment_path):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.training_config = training_config
        self.model_config = model_config
        self.experiment_path = experiment_path
        self.agent = self._initialize_agent()
        self.rewards = []
        self.lengths = []
        self.losses = []
        self.timestep = 0
        self._initialize_seed()

    def _initialize_agent(self):
         # Example, adjust based on your imports
        return DDPGAgent(
            self.env.observation_space,
            self.env.action_space,
            eps=self.model_config['eps'],
            discount=self.model_config['discount'],
            learning_rate_actor=self.model_config['learning_rate_actor'],
            update_target_every=self.model_config['update_target_every']
        )

    def _initialize_seed(self):
        seed = self.training_config.get('seed', 42)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def train(self):
        max_episodes = self.training_config['max_episodes']
        max_timesteps = self.training_config['max_timesteps']
        log_interval = self.training_config['log_interval']
        save_interval = self.training_config['save_interval']
        render = self.training_config['render']
        save_path = self.experiment_path / 'results'

        save_path.mkdir(parents=True, exist_ok=True)

        for i_episode in range(1, max_episodes + 1):
            obs, _ = self.env.reset()
            self.agent.reset()
            total_reward = 0

            for t in range(max_timesteps):
                self.timestep += 1
                action = self.agent.act(obs)
                next_obs, reward, done, trunc, _ = self.env.step(action)
                self.agent.store_transition((obs, action, reward, next_obs, done))
                obs = next_obs
                total_reward += reward

                if render:
                    self.env.render()

                if done or trunc:
                    break

            self.losses.extend(self.agent.train(self.training_config['train_iter']))
            self.rewards.append(total_reward)
            self.lengths.append(t)

            if i_episode % save_interval == 0:
                self._save_checkpoint(i_episode, save_path)

            if i_episode % log_interval == 0:
                self._log_progress(i_episode, log_interval)

        return self._final_metrics()

    def _save_checkpoint(self, episode, save_path):
        torch.save(
            self.agent.state(),
            save_path / f"checkpoint_ep{episode}.pth"
        )

    def _log_progress(self, episode, log_interval):
        avg_reward = np.mean(self.rewards[-log_interval:])
        avg_length = int(np.mean(self.lengths[-log_interval:]))
        print(f"Episode {episode} \t Avg Length: {avg_length} \t Avg Reward: {avg_reward}")

    def _final_metrics(self):
        return {
            "average_reward": np.mean(self.rewards),
            "average_length": np.mean(self.lengths),
            "final_loss": self.losses[-1] if self.losses else None
        }
