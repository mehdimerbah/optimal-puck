#!/usr/bin/env python3
"""
dreamer_evaluation.py

This script provides a full evaluation pipeline for Dreamer-style training on the HockeyEnv,
which has 18-dimensional vector observations.

Features:
1) Evaluate the world model for ALL models in 'saved_models':
   - Reconstruction error (MSE, optional SSIM -> placeholder => always NaN for vector data)
   - Multi-step latent prediction error (st_pred vs. st_post)
   - Compare real vs. "dreamed" episodes (placeholder).

2) Evaluate latent space for one chosen model:
   - PCA, KMeans, DBSCAN, nearest neighbor, KDE.

3) Evaluate imagined rollouts for that chosen model:
   - (a) Latent-state error
   - (b) Obs reconstruction error (using the VectorDecoder)
   - (c) Compare real vs. predicted rewards.

4) Evaluate policy performance in dreaming vs. real environment.

Additionally:
 - Optionally generate GIFs against weak/strong opponents.
 - Optionally allow a human to play with the environment.

Everything is saved in subfolders:
  rl_experiments/experiments/<experiment_id>/results/evaluation/
    - "plots/<model_name>" for all evaluation plots
    - "gifs/<model_name>"  for all created GIFs
    - a text summary for the all-model world-model evaluation
"""

import argparse
import time
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
from typing import Dict, Any

# For dimensionality reduction, clustering, neighbors, density
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors, KernelDensity

# SSIM attempt
try:
    from pytorch_msssim import ssim as ssim_func

    HAVE_SSIM = True
except ImportError:
    print("[Warning] 'pytorch_msssim' not found. SSIM metrics will be NaN.")
    HAVE_SSIM = False


    def ssim_func(a, b, *args, **kwargs):
        return torch.tensor(float('nan'))

# Laser Hockey environment
import hockey.hockey_env as h_env
from hockey.hockey_env import Mode, HockeyEnv_BasicOpponent, HumanOpponent, BasicOpponent

# Import your Dreamer trainer
from models.dreamer.DREAMTrainer import DreamerV3Trainer
#from models.dreamer.DREAM import VectorDecoder



def _create_model_output_dirs(base_dir: Path, model_name: str) -> Dict[str, Path]:
    """
    Creates "plots/<model_name>" and "gifs/<model_name>" subdirectories under base_dir
    """
    plots_dir = base_dir / "plots" / model_name
    gifs_dir = base_dir / "gifs" / model_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    gifs_dir.mkdir(parents=True, exist_ok=True)
    return {"plots": plots_dir, "gifs": gifs_dir}


def evaluate_world_model_all(
        trainer_class,
        experiment_path: Path,
        model_config: Dict[str, Any],
        env_name: str,
        num_eval_steps: int = 100,
        device: str = 'cpu'
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the world model for all .pth in 'saved_models'.

    1) Reconstruction error if a decoder is present (MSE, SSIM => mostly NaN for vector).
    2) Multi-step latent prediction error (st_pred vs. st_post).
    3) Compare real vs. dreamed => placeholder.
    """
    saved_models_dir = experiment_path / "results" / "training" / "saved_models"
    model_files = list(saved_models_dir.glob("*.pth"))
    model_files.sort()
    if len(model_files) == 0:
        print(f"No checkpoints found in {saved_models_dir}")
        return {}

    env_eval = HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=True)
    results_dict = {}

    for model_file in model_files:
        print(f"\n[WorldModelEval] Evaluating world model: {model_file.name}")

        trainer = trainer_class(env_name=env_name, training_config={}, model_config=model_config,
                                experiment_path=experiment_path)
        checkpoint = torch.load(model_file, map_location=device)
        trainer.agent.restore_state(checkpoint)

        # Short real rollout
        obs_list = []
        next_obs_list = []
        actions_list = []

        obs, _ = env_eval.reset()
        for step in range(num_eval_steps):
            action = trainer.agent.act(obs, sample=False)
            next_obs, reward, done, trunc, info = env_eval.step(action)

            obs_list.append(obs)
            next_obs_list.append(next_obs)
            actions_list.append(action)

            obs = next_obs
            if done or trunc:
                break

        # Reconstruction placeholders
        reconstruction_mse = float('nan')
        reconstruction_ssim = float('nan')
        # If you truly have a decoder and want to do an MSE over 18-dim, you'd sample e.g.
        #   decode(...) => obs_pred => compare with real obs. (Omitted here.)

        # Multi-step latent error
        multi_step_error = 0.0
        count_ms = 0
        state = trainer.agent.world_model.init_state(batch_size=1)

        with torch.no_grad():
            for i in range(len(obs_list) - 1):
                current_obs = torch.tensor(obs_list[i], dtype=torch.float32, device=device).unsqueeze(0)
                act_t = torch.tensor(actions_list[i], dtype=torch.float32, device=device).unsqueeze(0)

                next_state, stats, r_pred, done_logit, st_pred = trainer.agent.world_model(state, act_t, current_obs)
                # get real next posterior
                real_next_obs = torch.tensor(obs_list[i + 1], dtype=torch.float32, device=device).unsqueeze(0)
                real_next_state, _ = trainer.agent.world_model.rssm(next_state, torch.zeros_like(act_t), real_next_obs)
                # st_pred vs real_next_state[2]
                err = torch.mean((st_pred - real_next_state[2]) ** 2).item()
                multi_step_error += err
                count_ms += 1

                state = next_state

        if count_ms > 0:
            multi_step_error /= count_ms

        # Compare real vs. dreamed => placeholder
        dreamed_vs_real_score = float('nan')

        results_dict[model_file.name] = {
            "reconstruction_mse": reconstruction_mse,
            "reconstruction_ssim": reconstruction_ssim,
            "multi_step_error": multi_step_error,
            "dreamed_vs_real_score": dreamed_vs_real_score
        }

    env_eval.close()
    return results_dict


def evaluate_latent_space(trainer, output_dir: Path, device='cpu'):
    """
    Evaluate latent space with PCA, KMeans, DBSCAN, NN, KDE.
    Saves plots in output_dir.
    """
    print("\n[LatentSpaceEval] Evaluating the latent space...")

    env_eval = HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=True)
    obs, _ = env_eval.reset()
    latents = []
    state = trainer.agent.world_model.init_state(batch_size=1)
    max_steps = 200

    with torch.no_grad():
        for _ in range(max_steps):
            d1, d2, st = state
            lat_vec = torch.cat([d1, d2, st], dim=-1).cpu().numpy().squeeze(0)
            latents.append(lat_vec)

            action = trainer.agent.act(obs, sample=False)
            next_obs, reward, done, trunc, info = env_eval.step(action)
            if done or trunc:
                break

            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            next_state, _, _, _, _ = trainer.agent.world_model(state, act_t, obs_t)

            state = next_state
            obs = next_obs

    env_eval.close()
    latents = np.array(latents)
    if latents.shape[0] < 2:
        print("Not enough latent points for analysis. Skipping.")
        return

    # PCA
    pca = PCA(n_components=2)
    lat_2d = pca.fit_transform(latents)
    plt.figure(figsize=(8, 6))
    plt.scatter(lat_2d[:, 0], lat_2d[:, 1], s=12, alpha=0.7, label="Latent points")
    plt.title("PCA of Latent States")
    plt.grid(True)
    plt.legend()
    pca_path = output_dir / "latent_space_pca.png"
    plt.savefig(pca_path)
    plt.close()

    # KMeans
    kmeans = KMeans(n_clusters=3, n_init="auto")
    k_labels = kmeans.fit_predict(lat_2d)
    plt.figure(figsize=(8, 6))
    plt.scatter(lat_2d[:, 0], lat_2d[:, 1], c=k_labels, s=12, cmap='viridis')
    plt.title("KMeans (k=3) in Latent Space")
    km_path = output_dir / "latent_space_kmeans.png"
    plt.savefig(km_path)
    plt.close()

    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    db_labels = dbscan.fit_predict(lat_2d)
    plt.figure(figsize=(8, 6))
    plt.scatter(lat_2d[:, 0], lat_2d[:, 1], c=db_labels, s=12, cmap='plasma')
    plt.title("DBSCAN in Latent Space")
    db_path = output_dir / "latent_space_dbscan.png"
    plt.savefig(db_path)
    plt.close()

    # NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(lat_2d)
    idx_rand = np.random.randint(0, len(lat_2d))
    query = lat_2d[idx_rand].reshape(1, -1)
    dist, inds = neigh.kneighbors(query)

    plt.figure(figsize=(8, 6))
    plt.scatter(lat_2d[:, 0], lat_2d[:, 1], s=12, alpha=0.4, label="All latents")
    plt.scatter(lat_2d[idx_rand, 0], lat_2d[idx_rand, 1], c='red', s=40, label="Query")
    nn_points = lat_2d[inds[0]]
    plt.scatter(nn_points[:, 0], nn_points[:, 1], c='green', s=40, label="Neighbors")
    plt.title("Nearest Neighbors in Latent Space")
    plt.legend()
    nn_path = output_dir / "latent_space_nearest_neighbors.png"
    plt.savefig(nn_path)
    plt.close()

    # KDE
    kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
    kde.fit(lat_2d)
    x_min, x_max = lat_2d[:, 0].min() - 0.5, lat_2d[:, 0].max() + 0.5
    y_min, y_max = lat_2d[:, 1].min() - 0.5, lat_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    points = np.vstack([xx.ravel(), yy.ravel()]).T
    log_dens = kde.score_samples(points)
    dens = np.exp(log_dens).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, dens, cmap='Blues', alpha=0.6)
    plt.scatter(lat_2d[:, 0], lat_2d[:, 1], s=12, color='red', alpha=0.7)
    plt.colorbar().set_label("Density")
    plt.title("KDE of Latent Space")
    kde_path = output_dir / "latent_space_kde.png"
    plt.savefig(kde_path)
    plt.close()

    print("Latent space evaluation done.")


def evaluate_imagined_rollouts(trainer, output_dir: Path, device='cpu'):
    """
    Evaluate 'dreamed' future:
      - (a) Latent MSE: st_pred vs. st_post
      - (b) Decoder-based reconstruction MSE (vector env), SSIM placeholder => NaN
      - (c) Compare real vs. predicted rewards
    """
    print("\n[ImaginedRolloutsEval] Start evaluating imagined rollouts...")

    env_eval = HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=True)
    obs, _ = env_eval.reset()

    rollout_len = 50
    real_obs_list = []
    lat_post_list = []
    lat_pred_list = []
    obs_pred_list = []  # store decoded obs
    real_reward_list = []
    pred_reward_list = []

    state_post = trainer.agent.world_model.init_state(batch_size=1)

    with torch.no_grad():
        for t in range(rollout_len):
            real_obs_list.append(obs)

            # pick action from latents
            d1, d2, st = state_post
            feat = torch.cat([d1, d2, st], dim=-1)
            action_torch, _ = trainer.agent.actor(feat, sample=False)
            if not trainer.agent.is_discrete:
                action_np = action_torch.cpu().numpy().squeeze(0)
            else:
                action_np = int(action_torch.cpu().numpy()[0])

            # step real env
            next_obs, real_r, done, trunc, info = env_eval.step(action_np)
            real_reward_list.append(real_r)

            # posterior update => st_post
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if trainer.agent.is_discrete:
                oh = torch.nn.functional.one_hot(action_torch.long(), num_classes=trainer.agent.act_dim).float()
                act_in = oh
            else:
                act_in = action_torch

            next_state_post, stats, r_pred, done_logit, st_pred = trainer.agent.world_model(state_post, act_in, obs_t)
            lat_post_list.append(next_state_post[2].cpu().numpy().squeeze(0))
            lat_pred_list.append(st_pred.cpu().numpy().squeeze(0))
            pred_reward_list.append(r_pred.item())

            # decode st_pred => predicted observation
#            if trainer.agent.world_model.use_decoder and trainer.agent.world_model.decoder is not None:
#                # we decode with st_pred to see open-loop
#                d1_n, d2_n = next_state_post[0], next_state_post[1]
#                latent_for_decode = torch.cat([d1_n, d2_n, st_pred], dim=-1)  # shape [1, deter*2 + stoch]
#                obs_dec = trainer.agent.world_model.decoder(latent_for_decode)  # shape [1, obs_dim]
#                obs_pred_list.append(obs_dec.cpu().numpy().squeeze(0))  # (obs_dim,)
#            else:
#                obs_pred_list.append(None)

            state_post = next_state_post
            obs = next_obs
            if done or trunc:
                break

    env_eval.close()

    # Convert to arrays
    real_obs_arr = np.array(real_obs_list)  # shape [T, obs_dim]
    lat_post_arr = np.array(lat_post_list)  # shape [T, stoch_dim]
    lat_pred_arr = np.array(lat_pred_list)  # shape [T, stoch_dim]
    obs_pred_arr = obs_pred_list  # list of arrays or None
    real_reward_arr = np.array(real_reward_list)
    pred_reward_arr = np.array(pred_reward_list)

    # (a) Latent MSE
    latent_mse_each = np.mean((lat_pred_arr - lat_post_arr) ** 2, axis=1)
    plt.figure()
    plt.plot(latent_mse_each, label="Latent MSE (st_pred vs. st_post)")
    plt.xlabel("Time")
    plt.ylabel("MSE")
    plt.title("Latent Prediction Error Over Time")
    plt.legend()
    lat_mse_path = output_dir / "imagined_latent_mse.png"
    plt.savefig(lat_mse_path)
    plt.close()

    # (b) Obs reconstruction MSE + SSIM placeholder
    have_deco = any([o is not None for o in obs_pred_arr])
    if have_deco:
        obs_mse_list = []
        obs_ssim_list = []
        for t in range(len(obs_pred_arr)):
            pred_obs_vec = obs_pred_arr[t]
            if pred_obs_vec is None:
                obs_mse_list.append(np.nan)
                obs_ssim_list.append(np.nan)
                continue
            # real_obs_arr[t] => shape [obs_dim]
            # pred_obs_vec    => shape [obs_dim]
            # MSE
            mse_val = np.mean((pred_obs_vec - real_obs_arr[t]) ** 2)
            obs_mse_list.append(mse_val)
            # SSIM is typically for images, so we just set it to NaN or keep a placeholder
            obs_ssim_list.append(np.nan)

        obs_mse_list = np.array(obs_mse_list)
        plt.figure()
        plt.plot(obs_mse_list, label="Obs MSE (decoded vs. real)")
        plt.xlabel("Time")
        plt.ylabel("MSE")
        plt.title("Observation Reconstruction Error Over Time")
        plt.legend()
        obs_mse_path = output_dir / "imagined_obs_mse.png"
        plt.savefig(obs_mse_path)
        plt.close()
    else:
        print("No decoded observations found. Skipping obs-based MSE/SSIM plots.")

    # (c) real vs. predicted reward
    length = min(len(real_reward_arr), len(pred_reward_arr))
    rr = real_reward_arr[:length]
    pr = pred_reward_arr[:length]
    plt.figure()
    plt.plot(rr, label="Real Reward")
    plt.plot(pr, label="Predicted Reward")
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.title("Real vs. Predicted Reward")
    plt.legend()
    rew_plot = output_dir / "imagined_reward_comparison.png"
    plt.savefig(rew_plot)
    plt.close()

    print("Imagined rollouts evaluation done.")


def evaluate_dream_policy(trainer, output_dir: Path):
    """
    Compare policy performance in real env vs. purely latent dream rollout.
    """
    print("\n[DreamPolicyEval] Evaluating dream policy performance...")

    # 1) Real environment
    env_eval = HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=True)
    obs, _ = env_eval.reset()
    real_return = 0.0
    for _ in range(100):
        action = trainer.agent.act(obs, sample=False)
        obs, rew, done, trunc, info = env_eval.step(action)
        real_return += rew
        if done or trunc:
            break
    env_eval.close()

    # 2) Dream environment
    dream_return = 0.0
    state = trainer.agent.world_model.init_state(batch_size=1)
    with torch.no_grad():
        for _ in range(100):
            d1, d2, st = state
            lat = torch.cat([d1, d2, st], dim=-1)
            act, _ = trainer.agent.actor(lat, sample=False)

            if trainer.agent.is_discrete:
                oh = torch.nn.functional.one_hot(act.long(), num_classes=trainer.agent.act_dim).float()
                act_in = oh
            else:
                act_in = act

            feat = torch.cat([d1, d2, st, act_in], dim=-1)
            r_pred = trainer.agent.world_model.reward_head(feat)
            dream_return += r_pred.item()

            dummy_obs = torch.zeros((1, trainer.agent.obs_dim), dtype=torch.float32, device=lat.device)
            next_state, _, _, _, _ = trainer.agent.world_model(
                state, act_in, dummy_obs
            )
            state = next_state

    action_sens_info = "Action sensitivity test not implemented. Placeholder."

    print(f"Real Return: {real_return:.2f}, Dream Return: {dream_return:.2f}")
    print(action_sens_info)

    res_txt = output_dir / "dream_policy_eval.txt"
    with open(res_txt, "w") as f:
        f.write(f"Real Return: {real_return:.2f}\n")
        f.write(f"Dream Return: {dream_return:.2f}\n")
        f.write(action_sens_info + "\n")
    print(f"Saved dream policy eval -> {res_txt}")


def create_gif_vs_opponent(
        trainer,
        output_dir: Path,
        opponent_weak: bool = True,
        gif_name: str = "eval_opponent.gif",
        num_steps: int = 200
):
    """
    Generate a GIF of agent vs. BasicOpponent (weak or strong).
    """
    env = HockeyEnv_BasicOpponent(mode=Mode.NORMAL, weak_opponent=opponent_weak)
    frames = []
    obs, _ = env.reset()
    trainer.agent.reset()

    for _ in range(num_steps):
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        action = trainer.agent.act(obs, sample=False)
        obs, rew, done, trunc, info = env.step(action)
        if done or trunc:
            break

    env.close()
    gif_path = output_dir / gif_name
    imageio.mimsave(gif_path, frames, fps=15)
    print(f"Saved GIF -> {gif_path} (Opponent = {'Weak' if opponent_weak else 'Strong'})")


def play_against_human():
    """
    Let a human use keyboard to control one player vs. BasicOpponent.
    """
    print("\n[HumanPlay] Opening HockeyEnv for a human vs BasicOpponent...")

    env = h_env.HockeyEnv()
    player1 = HumanOpponent(env=env, player=1)
    player2 = BasicOpponent(weak=False)
    obs, info = env.reset()

    env.render()
    time.sleep(1)

    obs_agent2 = env.obs_agent_two()
    for _ in range(1000):
        time.sleep(0.02)
        env.render()
        a1 = player1.act(obs)
        a2 = player2.act(obs_agent2)
        obs, r, d, t, info = env.step(np.hstack([a1, a2]))
        obs_agent2 = env.obs_agent_two()
        if d:
            break
    env.close()
    print("Human vs agent match ended.")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Dreamer Evaluation for HockeyEnv (Vector Observations)')
    parser.add_argument('--experiment-id', type=str, required=True,
                        help='Experiment folder name under rl_experiments/experiments/<id>')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name of single model checkpoint, e.g. dreamer_checkpoint_ep200.pth')
    parser.add_argument('--num-steps-worldmodel', type=int, default=100,
                        help='Number of steps for short environment rollouts to evaluate world model.')
    parser.add_argument('--evaluate-latents', action='store_true',
                        help='Run latent-space evaluation for the chosen model.')
    parser.add_argument('--evaluate-imagination', action='store_true',
                        help='Run imagination rollout evaluation for the chosen model.')
    parser.add_argument('--evaluate-dream-policy', action='store_true',
                        help='Compare real vs. dream policy performance for the chosen model.')
    parser.add_argument('--play-human', action='store_true',
                        help='Enable a human to play in the HockeyEnv.')
    parser.add_argument('--record-gifs', action='store_true',
                        help='Record GIFs vs. weak/strong opponents.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use: cpu or cuda')
    args = parser.parse_args()

    experiment_id = args.experiment_id
    chosen_model_name = args.model_name
    device = args.device

    # Prepare evaluation folder
    experiment_path = Path(f"rl_experiments/experiments/{experiment_id}")
    evaluation_dir = experiment_path / "results" / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_file = experiment_path / "configs" / "training_config.yaml"
    with open(config_file, "r") as f:
        training_config = yaml.safe_load(f)

    env_name = training_config["environment"]["name"]
    model_config = training_config["model"]["config"]





    # 1) Evaluate world model for all .pth
    print("=== Step 1: Evaluate the world model for ALL saved models ===")
    wm_results = evaluate_world_model_all(
        trainer_class=DreamerV3Trainer,
        experiment_path=experiment_path,
        model_config=model_config,
        env_name=env_name,
        num_eval_steps=args.num_steps_worldmodel,
        device=device
    )



    print("\nWorld Model Evaluation Results (All Models):")
    wm_log = evaluation_dir / "world_model_evaluation_summary.txt"
    with open(wm_log, "w") as f:
        for fname, metrics in wm_results.items():
            line = f"{fname}: {metrics}\n"
            print(line, end='')
            f.write(line)

    # 2) Load single chosen model
    chosen_model_path = experiment_path / "results" / "training" / "saved_models" / chosen_model_name
    if not chosen_model_path.exists():
        print(f"ERROR: Chosen model not found => {chosen_model_path}")
        return

    print(f"\n=== Loading chosen model: {chosen_model_name} ===")
    trainer = DreamerV3Trainer(env_name=env_name, training_config={}, model_config=model_config,
                               experiment_path=experiment_path)
    checkpoint = torch.load(chosen_model_path, map_location=device)
    trainer.agent.restore_state(checkpoint)
    print(f"Loaded model from {chosen_model_path}")



    # Create subfolders for the chosen model
    out_dirs = _create_model_output_dirs(evaluation_dir, chosen_model_name)

    if args.record_gifs:
        create_gif_vs_opponent(trainer, out_dirs["gifs"], opponent_weak=True, gif_name="dreamer_vs_weak.gif")
        create_gif_vs_opponent(trainer, out_dirs["gifs"], opponent_weak=False, gif_name="dreamer_vs_strong.gif")


    # Evaluate latents
    if args.evaluate_latents:
        evaluate_latent_space(trainer, out_dirs["plots"], device=device)

    # Evaluate imagination
    if args.evaluate_imagination:
        evaluate_imagined_rollouts(trainer, out_dirs["plots"], device=device)

    # Evaluate dream policy
    if args.evaluate_dream_policy:
        evaluate_dream_policy(trainer, out_dirs["plots"])

    # Record GIFs
    if args.record_gifs:
        create_gif_vs_opponent(trainer, out_dirs["gifs"], opponent_weak=True, gif_name="dreamer_vs_weak.gif")
        create_gif_vs_opponent(trainer, out_dirs["gifs"], opponent_weak=False, gif_name="dreamer_vs_strong.gif")

    # Human play
    if args.play_human:
        play_against_human()


if __name__ == "__main__":
    main()
