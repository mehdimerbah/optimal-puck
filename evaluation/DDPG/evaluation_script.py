## for each experiment, load the checkpoints and evaluate the agent against the weak and strong opponent
import os
import warnings 
warnings.filterwarnings("ignore")
import numpy as np
import torch
from hockey.hockey_env import HockeyEnv_BasicOpponent, Mode, HockeyEnv, BasicOpponent
import pickle
from models.ddpg.DDPG import DDPGAgent


init_env = HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=False)


model_config = {
    "noise_scale": 0.1, 
    "beta": 0.4,
    "soft_update": True,                 
    "discount": 0.95,      
    "buffer_size": int(1e6),     
    "batch_size": 128,            
    "learning_rate_actor": 1e-4, 
    "learning_rate_critic": 1e-4,
    "hidden_sizes_actor": [256, 256],
    "hidden_sizes_critic": [256, 256],
    "update_target_every": 100,
    "use_target_net": True,
    "prioritized": True
}

ddpg_agent = DDPGAgent(
            observation_space=init_env.observation_space,
            action_space= init_env.action_space,
            **model_config
            )

experiments_path = "experiment_evaluation"
win_stats = {}
experiments_list = [f"{experiments_path}/{experiment}" for experiment in os.listdir(experiments_path)]
experiments_list = sorted(experiments_list, key=lambda x: int(x.split("_")[-2]))


for experiment in experiments_list:
    checkpoints_path = f"{experiment}/results/training/saved_models/"
    checkpoints = os.listdir(checkpoints_path)
    win_stats[experiment] = {}
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0][2:]))

    print(F"###################### {experiment} #######################")

    for checkpoint in checkpoints:
        checkpoint_path = f"{checkpoints_path}/{checkpoint}"
        model = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        ddpg_agent.restore_state(model)
        print("Current Checkpoint:", checkpoint)

        # Evaluate against the weak opponent
        eval_env = HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=True)
        
        num_test_episodes = 100
        cumulative_rewards = []
        win_count = 0
        for ep in range(num_test_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                # set simple deterministic policy
                full_action = ddpg_agent.act(obs, evaluate=True)
                action = full_action[:4]
                obs, reward, done, trunc, info = eval_env.step(action)
                episode_reward += reward

                eval_env.render(mode="rgb_array")

                if done or trunc:
                    break
            cumulative_rewards.append(episode_reward)
            if info["winner"] == 1:
                win_count += 1

        eval_env.close()

        win_stats[experiment][checkpoint] = {
            "weak_opponent": {
                "average_cumulative_reward": np.mean(cumulative_rewards),
                "win_rate": win_count / num_test_episodes
            }
        }

        print("Weak Opponent Stats:", win_stats[experiment][checkpoint]["weak_opponent"])

        # Evaluate against the strong opponent
        eval_env = HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=False)
        
        num_test_episodes = 100
        cumulative_rewards = []
        win_count = 0
        for ep in range(num_test_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                # set simple deterministic policy
                full_action = ddpg_agent.act(obs, evaluate=True)
                action = full_action[:4]
                obs, reward, done, trunc, info = eval_env.step(action)
                episode_reward += reward

                eval_env.render(mode="rgb_array")

                if done or trunc:
                    break
            cumulative_rewards.append(episode_reward)
            if info["winner"] == 1:
                win_count += 1

        win_stats[experiment][checkpoint]["strong_opponent"] = {
            "average_cumulative_reward": np.mean(cumulative_rewards),
            "win_rate": win_count / num_test_episodes
        }

        print("Strong Opponent Stats:", win_stats[experiment][checkpoint]["strong_opponent"])

        eval_env.close()

    ## save win stats into pickle file

    with open(f"{experiment}/results/evaluation/win_stats.pkl", "wb") as f:
        pickle.dump(win_stats[experiment], f)

## save all win stats to pickle file

with open(f"{experiments_path}/win_stats.pkl", "wb") as f:
    pickle.dump(win_stats, f)


