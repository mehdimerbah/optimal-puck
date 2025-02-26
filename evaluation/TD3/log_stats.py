#!/usr/bin/env python3
import argparse
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def running_mean(x, N):
    """
    Compute the exponential moving average (EMA) of array x using a smoothing factor.
    For EMA, the smoothing factor is alpha = 2/(N+1) and the returned array has the same length as x.
    """
    x = np.array(x, dtype=float)
    if len(x) == 0:
        return x
    alpha = 2.0 / (N + 1)
    ema = np.empty_like(x)
    ema[0] = x[0]
    for t in range(1, len(x)):
        ema[t] = alpha * x[t] + (1 - alpha) * ema[t-1]
    return ema

def parse_log(file_path):
    """
    Parse the log file to extract episode numbers and various metrics.
    Returns a dict with arrays for each metric.
    """
    episodes = []
    rewards = []
    reward_std = []
    lengths = []
    overall_win_rate = []
    random_wr = []
    weak_wr = []
    strong_wr = []
    self_wr = []
    phases = []

    # Regex patterns for different metrics
    episode_pattern = re.compile(r"Episode\s+(\d+)")
    reward_pattern = re.compile(r"(?:Avg\s+)?Reward\s*:\s*([-+]?[0-9]*\.?[0-9]+)")
    reward_std_pattern = re.compile(r"(?:Reward\s*Std|Reward_std)\s*:\s*([-+]?[0-9]*\.?[0-9]+)")
    length_pattern = re.compile(r"(?:Avg\s+)?Length\s*:\s*([-+]?[0-9]*\.?[0-9]+)")
    overall_wr_pattern = re.compile(r"Win\s*(?:Rate)?:\s*([-+]?[0-9]*\.?[0-9]+)")
    random_wr_pattern = re.compile(r"Random\s+WR\s*:\s*([-+]?[0-9]*\.?[0-9]+)")
    weak_wr_pattern = re.compile(r"Weak\s+WR\s*:\s*([-+]?[0-9]*\.?[0-9]+)")
    strong_wr_pattern = re.compile(r"Strong\s+WR\s*:\s*([-+]?[0-9]*\.?[0-9]+)")
    self_wr_pattern = re.compile(r"Self\s+WR\s*:\s*([-+]?[0-9]*\.?[0-9]+)")
    phase_pattern = re.compile(r"Phase\s*:\s*(\S+)")

    with open(file_path, 'r') as f:
        for line in f:
            m_ep = episode_pattern.search(line)
            if m_ep:
                try:
                    ep_num = int(m_ep.group(1))
                except ValueError:
                    ep_num = None
            else:
                ep_num = None

            if ep_num is not None:
                episodes.append(ep_num)
                # Initialize placeholders
                rew_val = np.nan
                rew_std_val = np.nan
                len_val = np.nan
                owr_val = np.nan
                rwr_val = np.nan
                wwr_val = np.nan
                stwr_val = np.nan
                swr_val = np.nan
                ph_val = ""

                m = reward_pattern.search(line)
                if m:
                    try:
                        rew_val = float(m.group(1))
                    except ValueError:
                        pass

                m = reward_std_pattern.search(line)
                if m:
                    try:
                        rew_std_val = float(m.group(1))
                    except ValueError:
                        pass

                m = length_pattern.search(line)
                if m:
                    try:
                        len_val = float(m.group(1))
                    except ValueError:
                        pass

                m = overall_wr_pattern.search(line)
                if m:
                    try:
                        owr_val = float(m.group(1))
                    except ValueError:
                        pass

                m = random_wr_pattern.search(line)
                if m:
                    try:
                        rwr_val = float(m.group(1))
                    except ValueError:
                        pass

                m = weak_wr_pattern.search(line)
                if m:
                    try:
                        wwr_val = float(m.group(1))
                    except ValueError:
                        pass

                m = strong_wr_pattern.search(line)
                if m:
                    try:
                        stwr_val = float(m.group(1))
                    except ValueError:
                        pass

                m = self_wr_pattern.search(line)
                if m:
                    try:
                        swr_val = float(m.group(1))
                    except ValueError:
                        pass

                m = phase_pattern.search(line)
                if m:
                    ph_val = m.group(1)

                rewards.append(rew_val)
                reward_std.append(rew_std_val)
                lengths.append(len_val)
                overall_win_rate.append(owr_val)
                random_wr.append(rwr_val)
                weak_wr.append(wwr_val)
                strong_wr.append(stwr_val)
                self_wr.append(swr_val)
                phases.append(ph_val)

    data_dict = {
        'episodes': np.array(episodes, dtype=float),
        'rewards': np.array(rewards, dtype=float),
        'reward_std': np.array(reward_std, dtype=float),
        'lengths': np.array(lengths, dtype=float),
        'overall_win_rate': np.array(overall_win_rate, dtype=float),
        'random_wr': np.array(random_wr, dtype=float),
        'weak_wr': np.array(weak_wr, dtype=float),
        'strong_wr': np.array(strong_wr, dtype=float),
        'self_wr': np.array(self_wr, dtype=float),
        'phases': phases
    }
    return data_dict

def parse_pkl(pkl_file, num_episodes):
    """
    Load Q-loss and Actor-loss from a pickle file that contains data["losses"],
    a list or array of [q_loss, actor_loss] values.
    If there are more loss entries than episodes, average them per episode.
    Returns (q_losses, actor_losses) as arrays.
    """
    try:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"[WARNING] Could not load pickle file: {e}")
        return None, None

    losses = data.get("losses", [])
    if not losses:
        print("[INFO] No 'losses' found in the pickle file.")
        return None, None

    try:
        losses_array = np.array(losses, dtype=float)
        total_loss_entries = losses_array.shape[0]
        print(f"[DEBUG] losses_array shape: {losses_array.shape}")

        if losses_array.ndim != 2 or losses_array.shape[1] != 2:
            print("[WARNING] 'losses' is not a Nx2 array. Cannot parse Q/Actor Loss properly.")
            return None, None

        if total_loss_entries < num_episodes:
            print("[INFO] Less loss data than episodes; padding with NaN.")
            padded = np.full((num_episodes, 2), np.nan, dtype=float)
            padded[:total_loss_entries, :] = losses_array
            losses_array = padded
        elif total_loss_entries > num_episodes:
            if total_loss_entries % num_episodes == 0:
                batch_size = total_loss_entries // num_episodes
                losses_per_episode = losses_array.reshape(num_episodes, batch_size, 2).mean(axis=1)
            else:
                batch_size = total_loss_entries // num_episodes
                leftover = total_loss_entries % num_episodes
                print(f"[INFO] Imperfect grouping: {batch_size} updates/ep + leftover {leftover}")
                losses_per_episode = []
                idx = 0
                for _ in range(num_episodes):
                    chunk = losses_array[idx: idx + batch_size]
                    if len(chunk) > 0:
                        losses_per_episode.append(chunk.mean(axis=0))
                    else:
                        losses_per_episode.append([np.nan, np.nan])
                    idx += batch_size
                losses_per_episode = np.array(losses_per_episode)
        else:
            losses_per_episode = losses_array

        q_losses = losses_per_episode[:, 0]
        actor_losses = losses_per_episode[:, 1]
        print(f"[DEBUG] losses_per_episode shape: {losses_per_episode.shape}")
        print(f"[DEBUG] Q-losses: min={np.nanmin(q_losses):.6f}, max={np.nanmax(q_losses):.6f}, mean={np.nanmean(q_losses):.6f}")
        print(f"[DEBUG] Actor-losses: min={np.nanmin(actor_losses):.6f}, max={np.nanmax(actor_losses):.6f}, mean={np.nanmean(actor_losses):.6f}")
        return q_losses, actor_losses
    except Exception as e:
        print(f"[WARNING] Error processing 'losses' data: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Parse log and optional pickle stats, then plot them.")
    parser.add_argument("--log_file", type=str, default="test.log", help="Path to the log file with episode stats.")
    parser.add_argument("--pkl_file", type=str, default="test.pkl", help="Optional path to a pickle file containing Q/Actor losses.")
    parser.add_argument("--output_file", type=str, default="training_plot.png", help="Filename for saving the generated plot.")
    parser.add_argument("--window_size", type=int, default=50, help="Smoothing window size for the plots (EMA window).")
    parser.add_argument("--csv_file", type=str, default="training_data.csv", help="Filename for saving the data as CSV.")
    args = parser.parse_args()

    # 1) Parse log data
    log_data = parse_log(args.log_file)
    episodes = log_data["episodes"]
    rewards = log_data["rewards"]
    reward_std = log_data["reward_std"]
    lengths = log_data["lengths"]
    overall_wr = log_data["overall_win_rate"]
    random_wr = log_data["random_wr"]
    weak_wr = log_data["weak_wr"]
    strong_wr = log_data["strong_wr"]
    self_wr = log_data["self_wr"]

    if len(episodes) == 0:
        print(f"[ERROR] No episodes found in log file: {args.log_file}")
        return

    num_log_episodes = len(episodes)

    # 2) Parse pickle data for Q/Actor loss (optional)
    q_losses, actor_losses = None, None
    if args.pkl_file:
        q_losses, actor_losses = parse_pkl(args.pkl_file, num_log_episodes)

    # 3) Decide subplots; note every metric from the log is logged at a 50-episode step.
    n_plots = 1  # Always have Rewards
    has_q_actor = (q_losses is not None) and (actor_losses is not None)
    has_overall_wr = not np.isnan(overall_wr).all()
    has_any_opponent_wr = not (np.isnan(random_wr).all() and np.isnan(weak_wr).all() and
                               np.isnan(strong_wr).all() and np.isnan(self_wr).all())
    has_lengths = not np.isnan(lengths).all()

    if has_q_actor:
        n_plots += 2  # One subplot for Q-loss and one for Actor-loss
    if has_overall_wr:
        n_plots += 1
    if has_any_opponent_wr:
        n_plots += 1
    if has_lengths:
        n_plots += 1

    # Calculate number of rows needed (3 plots per column)
    n_rows = (n_plots + 1) // 2
    fig, axs = plt.subplots(n_rows, 2, figsize=(16, 6 * n_rows))
    axs = axs.ravel()  # Flatten the array for easier indexing
    
    # Initialize current axis index
    current_ax = 0
    window_size = args.window_size

    # 3a) Plot Rewards
    valid_idx = ~np.isnan(rewards)
    v_ep = episodes[valid_idx]
    v_rw = rewards[valid_idx]
    v_rw_std = reward_std[valid_idx] if not np.isnan(reward_std).all() else None

    # For EMA, the smoothed array has the same length as input, so use the full x-axis
    smoothed_rw = running_mean(v_rw, window_size)
    smoothed_ep = v_ep  # do not slice for EMA

    ax = axs[current_ax]
    ax.plot(v_ep, v_rw, label='Raw Rewards', alpha=0.4)
    ax.plot(smoothed_ep, smoothed_rw,
            label=f'EMA Rewards (window={window_size})', linewidth=2)
    if v_rw_std is not None and len(v_rw_std) == len(v_ep):
        smoothed_rw_std = running_mean(v_rw_std, window_size)
        ax.fill_between(smoothed_ep,
                        smoothed_rw - smoothed_rw_std,
                        smoothed_rw + smoothed_rw_std,
                        color='gray', alpha=0.2, label='Reward Std Dev')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Reward vs Episodes')
    ax.legend()
    ax.grid(True)
    current_ax += 1

    # 3b) Q-Loss & Actor-Loss (if present) using the same episode numbers from the log.
    if has_q_actor:
        valid_idx = ~np.isnan(q_losses)
        v_q = q_losses[valid_idx]
        v_ep_q = episodes[valid_idx]
        smoothed_q = running_mean(v_q, window_size)
        smoothed_q_ep = v_ep_q  # full length for EMA

        ax = axs[current_ax]
        ax.plot(v_ep_q, v_q, label='Raw Q-Loss', alpha=0.4)
        ax.plot(smoothed_q_ep, smoothed_q,
                 label=f'EMA Q-Loss (window={window_size})', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Q-Loss')
        ax.set_title('Q-Loss vs Episodes')
        ax.legend()
        ax.grid(True)
        current_ax += 1

        valid_idx = ~np.isnan(actor_losses)
        v_a = actor_losses[valid_idx]
        v_ep_a = episodes[valid_idx]
        smoothed_a = running_mean(v_a, window_size)
        smoothed_a_ep = v_ep_a

        ax = axs[current_ax]
        ax.plot(v_ep_a, v_a, label='Raw Actor Loss', alpha=0.4)
        ax.plot(smoothed_a_ep, smoothed_a,
                 label=f'EMA Actor Loss (window={window_size})', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Actor Loss')
        ax.set_title('Actor Loss vs Episodes')
        ax.legend()
        ax.grid(True)
        current_ax += 1

    # 3c) Overall Win Rate
    if has_overall_wr:
        valid_idx = ~np.isnan(overall_wr)
        v_ep_wr = episodes[valid_idx]
        v_wr = overall_wr[valid_idx]
        smoothed_wr = running_mean(v_wr, window_size)
        smoothed_wr_ep = v_ep_wr

        ax = axs[current_ax]
        ax.plot(v_ep_wr, v_wr, label='Raw Win Rate', alpha=0.4)
        ax.plot(smoothed_wr_ep, smoothed_wr,
                 label=f'EMA Win Rate (window={window_size})', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate')
        ax.set_title('Overall Win Rate vs Episodes')
        ax.legend()
        ax.grid(True)
        current_ax += 1

    # 3d) Opponent Win Rates
    if has_any_opponent_wr:
        wr_labels = ['Random WR', 'Weak WR', 'Strong WR', 'Self WR']
        wr_arrays = [random_wr, weak_wr, strong_wr, self_wr]
        colors = ['C0', 'C1', 'C2', 'C3']

        # Create a single subplot for all opponent win rates
        ax = axs[current_ax]
        for i, (label_i, arr_i) in enumerate(zip(wr_labels, wr_arrays)):
            valid_idx = ~np.isnan(arr_i)
            if not np.any(valid_idx):
                continue
            v_ep_opp = episodes[valid_idx]
            v_arr = arr_i[valid_idx]
            smoothed_arr = running_mean(v_arr, window_size)
            smoothed_ep = v_ep_opp

            ax.plot(v_ep_opp, v_arr,
                     label=f'Raw {label_i}', alpha=0.3,
                     color=colors[i])
            ax.plot(smoothed_ep, smoothed_arr,
                     label=f'EMA {label_i} (window={window_size})',
                     linewidth=2, color=colors[i])
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate')
        ax.set_title('Opponent Win Rates vs Episodes')
        ax.legend()
        ax.grid(True)
        current_ax += 1

    # 3e) Episode Length
    if has_lengths:
        valid_idx = ~np.isnan(lengths)
        v_ep_len = episodes[valid_idx]
        v_len = lengths[valid_idx]
        smoothed_len = running_mean(v_len, window_size)
        smoothed_len_ep = v_ep_len

        ax = axs[current_ax]
        ax.plot(v_ep_len, v_len, label='Raw Episode Length', alpha=0.4)
        ax.plot(smoothed_len_ep, smoothed_len,
                 label=f'EMA Episode Length (window={window_size})', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Length')
        ax.set_title('Episode Length vs Episodes')
        ax.legend()
        ax.grid(True)
        current_ax += 1

    # Hide any unused subplots
    for i in range(current_ax, len(axs)):
        axs[i].axis('off')

    # 4) Save data to CSV
    data_dict = {
        'episodes': episodes,
        'rewards': rewards,
        'reward_std': reward_std,
        'lengths': lengths,
        'overall_win_rate': overall_wr,
        'random_wr': random_wr,
        'weak_wr': weak_wr,
        'strong_wr': strong_wr,
        'self_wr': self_wr
    }
    
    if q_losses is not None:
        data_dict['q_losses'] = q_losses
    if actor_losses is not None:
        data_dict['actor_losses'] = actor_losses
    
    df = pd.DataFrame(data_dict)
    df.to_csv(args.csv_file, index=False)
    print(f"\nData saved to {args.csv_file}")

    plt.tight_layout()
    plt.savefig(args.output_file)
    print(f"Plot saved to {args.output_file}")
    plt.show()


if __name__ == "__main__":
    import sys
    main()
