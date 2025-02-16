import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
import argparse
import yaml

from hockey.hockey_env import HockeyEnv_BasicOpponent, Mode
from models.dreamer.DREAMTrainer import DreamerV3Trainer


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained agent for a specified experiment')
    parser.add_argument('--experiment-id', type=str, help='experiment id', dest='experiment_id')
    parser.add_argument('--model-name', type=str, help='name of the trained model checkpoint', dest='model_name',
                        required=True)
    args = parser.parse_args()

    experiment_id = args.experiment_id
    experiment_path = Path(f"rl_experiments/experiments/{experiment_id}")
    model_path = experiment_path / "results/training/saved_models" / args.model_name

    with open(experiment_path / "configs/training_config.yaml", "r") as f:
        training_config = yaml.safe_load(f)

    env_name = training_config['environment']['name']
    model_config = training_config['model']['config']

    trainer_map = {
        "Dreamer": DreamerV3Trainer
    }

    trainer = trainer_map[training_config['model']['name']](
        env_name=env_name,
        training_config={},  # No training needed
        model_config=model_config,
        experiment_path=experiment_path
    )

    # Load trained model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    trainer.agent.restore_state(checkpoint)
    print(f"Loaded model from {model_path}")

    # Evaluate model
    num_test_episodes = 50
    cumulative_rewards = []
    win_count = 0

    env = HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=True)
    frames = []

    for ep in range(num_test_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = trainer.agent.act(obs)
            obs, reward, done, trunc, info = env.step(action)
            episode_reward += reward
            frames.append(env.render(mode='rgb_array'))
            if done or trunc:
                break
        cumulative_rewards.append(episode_reward)
        win_count += info.get("winner", 0)

    env.close()

    # Save results
    results_path = experiment_path / "results/evaluation"
    results_path.mkdir(parents=True, exist_ok=True)

    gifs_path = results_path / "gifs"
    plots_path = results_path / "plots"
    gifs_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)

    # Plot and save results
    plt.plot(cumulative_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Rewards over Episodes")
    plt.savefig(plots_path / "cumulative_rewards.png")
    plt.close()

    # Save GIF
    gif_path = gifs_path / "dream_laserhockey_evaluation.gif"
    imageio.mimsave(gif_path, frames, fps=15)
    print(f"Saved evaluation GIF to {gif_path}")

    print("Average cumulative reward:", np.mean(cumulative_rewards))
    print("Win rate:", win_count / num_test_episodes)


if __name__ == "__main__":
    main()

