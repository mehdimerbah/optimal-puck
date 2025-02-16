"""
DREAM.py

A Dreamer-V3-like script featuring:
 - Multi-layer RSSM (2-layer GRU)
 - Forward-Model Curiosity (ICM)
 - Plan2Explore Ensemble -> Disagreement-based Intrinsic Reward
 - Separate alpha_ext, alpha_int (Dual SAC-Style)
 - Lambda-return for the Critic
 - Slow Target RSSM
 - Prioritized Sequence Replay (PER) with Weighted IS
 - 'kl_loss' fix
 - Adaptive Gradient Clipping (AGC) + Cosine LR
 - Value Normalization for Critic's lambda-returns
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from gymnasium import spaces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

# --------------------------------------------------------------------------------
# 0) Additional utilities: Weighted PER with Weighted IS, AGC, etc.
# --------------------------------------------------------------------------------

def adaptive_gradient_clipping(parameters, clip_factor=0.05, eps=1e-3):
    """
    Applies AGC (Adaptive Gradient Clipping) to a list of parameters.
    Clamps grad magnitudes based on parameter norms.
    This is a simpler version of the concept used in fairscale or jax's optax.
    """
    parameters = [p for p in parameters if p.grad is not None]
    for p in parameters:
        param_norm = p.data.norm(2)
        grad_norm = p.grad.data.norm(2)
        if param_norm == 0 or grad_norm == 0:
            continue
        ratio = (grad_norm / (param_norm + eps))
        if ratio > clip_factor:
            p.grad.data.mul_(clip_factor / (ratio + eps))

# --------------------------------------------------------------------------------
# 1) Exceptions and RunningMeanStd
# --------------------------------------------------------------------------------
class UnsupportedSpace(Exception):
    pass

class RunningMeanStd:
    """
    Tracks mean and variance in a streaming fashion.
    Used for e.g. normalizing rewards or returns.
    """
    def __init__(self, epsilon=1e-5, shape=()):
        self.epsilon = epsilon
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64)

    def update(self, x):
        x = np.asarray(x)
        self.count += x.shape[0]
        delta = x.mean(axis=0) - self.mean
        self.mean += delta * (x.shape[0] / self.count)
        delta2 = x.mean(axis=0) - self.mean
        self.M2 += (x - self.mean).T @ (x - self.mean)

    @property
    def var(self):
        return self.M2 / max(self.count - 1, 1)

    @property
    def std(self):
        return np.sqrt(self.var + self.epsilon)

    def normalize(self, x):
        return (x - self.mean) / (self.std + 1e-8)

# --------------------------------------------------------------------------------
# 2) Prioritized Sequence Replay (Episode-level, Weighted IS version)
# --------------------------------------------------------------------------------
class PrioritizedSequenceMemory:
    """
    A more refined PER approach:
      - Each episode has a priority
      - We store an array of priorities
      - We sample episodes according to p_i^alpha
      - Weighted IS correction factor for the sampled data
    """
    def __init__(self, max_size=100000, seq_len=50, alpha=0.6, beta=0.4):
        self.max_size = int(max_size)
        self.seq_len = seq_len
        self.alpha = alpha
        self.beta = beta
        self.episodes = []
        self.priorities = []
        self.cur_episode = []
        self.cur_priority = 0.0
        self.size = 0
        self.eps = 1e-6

    def add_transition(self, transition, priority=None):
        """
        transition = {obs, action, reward, next_obs, done}
        priority is optional
        """
        p = priority if priority is not None else 1.0
        self.cur_priority = max(self.cur_priority, p)
        self.cur_episode.append(transition)
        self.size += 1
        if transition["done"]:
            # finalize the episode
            self.episodes.append(self.cur_episode)
            self.priorities.append(self.cur_priority)
            self.cur_episode = []
            self.cur_priority = 0.0
            # evict oldest if memory is full
            while self.size > self.max_size and self.episodes:
                oldest = self.episodes.pop(0)
                oldest_p = self.priorities.pop(0)
                self.size -= len(oldest)

    def __len__(self):
        return len(self.episodes)

    def _sample_episode_index(self):
        if not self.episodes:
            return None, None
        ps = np.array(self.priorities, dtype=np.float32)
        ps_alpha = (ps + self.eps) ** self.alpha
        total = ps_alpha.sum()
        probs = ps_alpha / total
        idx = np.random.choice(len(self.episodes), p=probs)
        # Weighted importance-sampling factor
        # w_i = (1/(N*P(i)))^beta
        N = len(self.episodes)
        weight = (1.0 / (N*probs[idx]))**self.beta
        return idx, weight

    def sample_subsequence(self):
        """
        Sample a random subsequence from one randomly-chosen episode with PER weighting
        Returns: seq, weight
        """
        if not self.episodes:
            return None, None
        idx, w = self._sample_episode_index()
        if idx is None:
            return None, None
        ep = self.episodes[idx]
        L = len(ep)
        if L < self.seq_len:
            pad_len = self.seq_len - L
            seq = ep + [ep[-1]]*pad_len
        else:
            start_idx = np.random.randint(0, L - self.seq_len +1)
            seq = ep[start_idx:start_idx+self.seq_len]
        return seq, w

class PrioritizedSequenceDataset(Dataset):
    def __init__(self, replay_memory: PrioritizedSequenceMemory):
        self.replay_memory = replay_memory
        self.seq_len = replay_memory.seq_len

    def __len__(self):
        return max(1, len(self.replay_memory))

    def __getitem__(self, idx):
        seq, weight = self.replay_memory.sample_subsequence()
        if seq is None:
            return None
        batch_dict = {}
        keys = seq[0].keys()
        for k in keys:
            batch_dict[k] = np.array([step[k] for step in seq], dtype=np.float32)
        # we also store 'weight' for WeightedIS in the training step
        batch_dict['IS_weight'] = np.array(weight, dtype=np.float32)
        return batch_dict

# --------------------------------------------------------------------------------
# 3) Basic MLP
# --------------------------------------------------------------------------------
class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_fun=nn.ReLU(), output_activation=None):
        super().__init__()
        layers = []
        in_dim = input_size
        for hs in hidden_sizes:
            layers.append(nn.Linear(in_dim, hs))
            layers.append(activation_fun)
            in_dim = hs
        layers.append(nn.Linear(in_dim, output_size))
        if output_activation is not None:
            layers.append(output_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------------------------------------
# 4) Multi-layer RSSM + slow target
# --------------------------------------------------------------------------------
class RSSM(nn.Module):
    """
    A 2-layer GRU-based stochastic recurrent model
    """
    def __init__(self, obs_dim, act_dim, deter_dim=64, stoch_dim=32):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        self.rnn1 = nn.GRUCell(stoch_dim + act_dim, deter_dim)
        self.rnn2 = nn.GRUCell(deter_dim, deter_dim)

        self.prior_net = Feedforward(deter_dim*2, [64], 2*stoch_dim)
        self.post_net  = Feedforward(deter_dim*2 + obs_dim, [64], 2*stoch_dim)

        self.init_d1 = nn.Parameter(torch.zeros(deter_dim))
        self.init_d2 = nn.Parameter(torch.zeros(deter_dim))
        self.init_st = nn.Parameter(torch.zeros(stoch_dim))

    def init_state(self, batch_size):
        d1 = self.init_d1.unsqueeze(0).expand(batch_size, -1)
        d2 = self.init_d2.unsqueeze(0).expand(batch_size, -1)
        st = self.init_st.unsqueeze(0).expand(batch_size, -1)
        return (d1, d2, st)

    def forward(self, prev_state, action, obs):
        (d1, d2, st) = prev_state
        x1 = torch.cat([st, action], dim=-1)
        d1_next = self.rnn1(x1, d1)
        d2_next = self.rnn2(d1_next, d2)
        catd = torch.cat([d1_next, d2_next], dim=-1)

        prior_stats = self.prior_net(catd)
        pm, plv = torch.chunk(prior_stats, 2, dim=-1)

        post_inp = torch.cat([catd, obs], dim=-1)
        post_stats= self.post_net(post_inp)
        mm, mlv = torch.chunk(post_stats, 2, dim=-1)

        # sample
        st_next = mm + torch.exp(0.5*mlv)*torch.randn_like(mm)
        nxt = (d1_next, d2_next, st_next)
        stats = (pm, plv, mm, mlv)
        return nxt, stats

    def kl_loss(self, prior_stats, post_stats):
        """
        prior_stats = (pm, plv), post_stats = (mm, mlv)
        kl( post || prior )
        """
        pm, plv = prior_stats
        mm, mlv = post_stats
        kl = 0.5 * (
            plv - mlv
            + (torch.exp(mlv) + (mm - pm)**2)/torch.exp(plv)
            - 1.0
        )
        return kl.sum(dim=-1)

def soft_update_rssm(source: RSSM, target: RSSM, tau=0.01):
    """
    Soft-update from source to target with factor tau
    """
    for sp, tp in zip(source.parameters(), target.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

class SlowRSSM(nn.Module):
    """
    Contains an RSSM inside, so we can do load_state_dict or partial updates
    from a main RSSM.
    """
    def __init__(self, obs_dim, act_dim, deter_dim=64, stoch_dim=32):
        super().__init__()
        self.rssm = RSSM(obs_dim, act_dim, deter_dim, stoch_dim)

    def load_from_rssm(self, world_model):
        """
        Load parameters from the main world_model.rssm into this slow rssm.
        """
        self.rssm.load_state_dict(world_model.rssm.state_dict())

    def forward(self, *args, **kwargs):
        return self.rssm(*args, **kwargs)

    def init_state(self, *args, **kwargs):
        return self.rssm.init_state(*args, **kwargs)

    def kl_loss(self, *args, **kwargs):
        return self.rssm.kl_loss(*args, **kwargs)


# --------------------------------------------------------------------------------
# 5) WorldModel with ICM + reward/done heads
# --------------------------------------------------------------------------------
class RSSMWorldModel(nn.Module):
    """
    RSSM + forward model (ICM) + reward/done
    """
    def __init__(self, obs_dim, act_dim, deter_dim=64, stoch_dim=32):
        super().__init__()
        self.rssm = RSSM(obs_dim, act_dim, deter_dim, stoch_dim)
        in_dim = (2*deter_dim + stoch_dim + act_dim)
        self.reward_head = Feedforward(in_dim, [64], 1)
        self.done_head   = Feedforward(in_dim, [64], 1)
        self.forward_model = Feedforward(in_dim, [64], stoch_dim)

    def init_state(self, batch_size):
        return self.rssm.init_state(batch_size)

    def forward(self, prev_state, action, obs):
        next_state, stats = self.rssm(prev_state, action, obs)
        d1, d2, st = next_state
        feat = torch.cat([d1, d2, st, action], dim=-1)
        r_pred = self.reward_head(feat)
        done_logit = self.done_head(feat)
        stoch_pred = self.forward_model(feat)
        return next_state, stats, r_pred, done_logit, stoch_pred

# --------------------------------------------------------------------------------
# 6) Plan2Explore ensemble
# --------------------------------------------------------------------------------
class Plan2ExploreModel(nn.Module):
    """
    K deterministic networks => measure disagreement => intrinsic
    """
    def __init__(self, in_dim, out_dim, ensemble_size=5, hidden_size=64):
        super().__init__()
        self.ensemble = nn.ModuleList([
            Feedforward(in_dim, [hidden_size], out_dim) for _ in range(ensemble_size)
        ])
        self.K = ensemble_size

    def forward(self, x):
        preds = []
        for net in self.ensemble:
            preds.append(net(x))
        return preds

    @torch.no_grad()
    def disagreement(self, x):
        preds = self.forward(x)
        stack = torch.stack(preds, dim=0) # [K,B,out_dim]
        var = torch.var(stack, dim=0, unbiased=False) # [B, out_dim]
        return var.mean(dim=-1, keepdim=True) # [B,1]

# --------------------------------------------------------------------------------
# 7) Actor & Critic
# --------------------------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, latent_dim, action_space, hidden_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.is_discrete = isinstance(action_space, spaces.Discrete)
        if self.is_discrete:
            self.act_dim = action_space.n
            self.net = Feedforward(latent_dim, [hidden_size, hidden_size], self.act_dim)
        else:
            self.act_dim = action_space.shape[0]
            self.net = Feedforward(latent_dim, [hidden_size, hidden_size],
                                   self.act_dim, output_activation=nn.Tanh())

    def forward(self, latent, sample=True):
        out = self.net(latent)
        if self.is_discrete:
            dist = torch.distributions.Categorical(logits=out)
            act = dist.sample() if sample else torch.argmax(out, dim=-1)
            return act, dist
        else:
            dist = None
            act = out # in [-1,1]
            return act, dist

class Critic(nn.Module):
    def __init__(self, latent_dim, hidden_size=64):
        super().__init__()
        self.net = Feedforward(latent_dim, [hidden_size, hidden_size], 1)
    def forward(self, latent):
        return self.net(latent).squeeze(-1)

# --------------------------------------------------------------------------------
# 8) The DreamerV3Agent with:
#    - SlowTarget RSSM
#    - PER
#    - alpha_ext, alpha_int
#    - lambda_return with a running mean std on returns
#    - AGC-optimized training + Cosine LR
# --------------------------------------------------------------------------------

def soft_update_params(source: nn.Module, target: nn.Module, tau=0.01):
    for sp, tp in zip(source.parameters(), target.parameters()):
        tp.data.copy_(tau*sp.data + (1.0 - tau)*tp.data)

class DreamerV3Agent:
    def __init__(self, observation_space, action_space, **config):
        defaults = {
            "deter_dim": 64, "stoch_dim": 32,
            "wm_lr": 1e-3, "actor_lr":1e-4, "critic_lr":1e-4,
            "alpha_lr_ext":1e-4, "alpha_lr_int":1e-4,
            "discount": 0.99, "lambda_": 0.95,
            "buffer_size":1e5, "seq_len":50, "batch_size":8,
            "max_train_steps":1e6, "actor_horizon":5,
            "icm_scale": 0.2, "plan2explore_size":5, "plan2explore_scale":1.0,
            "target_kl":1.0, "trust_region_scale":0.05,
            "tau_rssm":0.01,   # soft-update rate for slow target
            "use_per": True,
            "per_alpha": 0.6,
            "per_beta": 0.4,
            "agc_clip": 0.05,  # AGC factor
        }
        defaults.update(config)
        self.cfg = defaults

        if not isinstance(observation_space, spaces.Box):
            raise UnsupportedSpace("Obs must be Box")

        self.obs_dim = observation_space.shape[0]
        self.action_space = action_space
        if isinstance(action_space, spaces.Box):
            self.is_discrete = False
            self.act_dim = action_space.shape[0]
        else:
            self.is_discrete = True
            self.act_dim = action_space.n

        # 1) World Model + slow
        self.world_model = RSSMWorldModel(
            obs_dim=self.obs_dim, act_dim=self.act_dim,
            deter_dim=self.cfg["deter_dim"],
            stoch_dim=self.cfg["stoch_dim"]
        ).to(device)
        self.slow_rssm = SlowRSSM(
            self.obs_dim, self.act_dim,
            self.cfg["deter_dim"], self.cfg["stoch_dim"]
        ).to(device)
        # load
        self.slow_rssm.rssm.load_state_dict(self.world_model.rssm.state_dict())

        # 2) Plan2Explore
        in_dim = (2*self.cfg["deter_dim"] + self.cfg["stoch_dim"] + self.act_dim)
        out_dim= self.cfg["stoch_dim"]
        self.plan2explore = Plan2ExploreModel(in_dim, out_dim, self.cfg["plan2explore_size"]).to(device)

        # 3) Actor/Critic
        latent_dim = (2*self.cfg["deter_dim"] + self.cfg["stoch_dim"])
        self.actor = Actor(latent_dim, self.action_space).to(device)
        self.critic= Critic(latent_dim).to(device)

        # 4) alpha_ext / alpha_int
        self.log_alpha_ext = nn.Parameter(torch.zeros([]))
        self.log_alpha_int = nn.Parameter(torch.zeros([]))
        self.opt_alpha_ext = torch.optim.Adam([self.log_alpha_ext], lr=self.cfg["alpha_lr_ext"])
        self.opt_alpha_int = torch.optim.Adam([self.log_alpha_int], lr=self.cfg["alpha_lr_int"])

        # 5) Build optimizers with AGC -> we can define a small wrapper
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.wm_params = list(self.world_model.parameters())
        self.ens_params= list(self.plan2explore.parameters())
        self.actor_params = list(self.actor.parameters())
        self.critic_params= list(self.critic.parameters())

        # We'll use normal Adam here, but manually apply AGC afterwards
        self.wm_opt = torch.optim.Adam(self.wm_params, lr=self.cfg["wm_lr"])
        self.ens_opt= torch.optim.Adam(self.ens_params,self.cfg["wm_lr"])
        self.actor_opt=torch.optim.Adam(self.actor_params,lr=self.cfg["actor_lr"])
        self.critic_opt=torch.optim.Adam(self.critic_params,lr=self.cfg["critic_lr"])

        self.wm_sched = CosineAnnealingLR(self.wm_opt, self.cfg["max_train_steps"], eta_min=1e-5)
        self.ens_sched= CosineAnnealingLR(self.ens_opt,self.cfg["max_train_steps"], eta_min=1e-5)
        self.actor_sched= CosineAnnealingLR(self.actor_opt,self.cfg["max_train_steps"], eta_min=1e-5)
        self.critic_sched=CosineAnnealingLR(self.critic_opt,self.cfg["max_train_steps"],eta_min=1e-5)

        # 6) Replay
        if self.cfg["use_per"]:
            self.buffer = PrioritizedSequenceMemory(
                max_size=int(self.cfg["buffer_size"]),
                seq_len=int(self.cfg["seq_len"]),
                alpha=self.cfg["per_alpha"],
                beta=self.cfg["per_beta"]
            )
            self.dataset= PrioritizedSequenceDataset(self.buffer)
        else:
            raise NotImplementedError("Non-PER buffer not implemented in this example.")
        self.dataloader = DataLoader(self.dataset,batch_size=int(self.cfg["batch_size"]),
                                     shuffle=True, drop_last=True)

        # 7) Reward RMS & Return RMS for Critic normalizing
        self.reward_rms = RunningMeanStd(shape=())
        self.return_rms = RunningMeanStd(shape=()) # Value normalization

        self.discount = self.cfg["discount"]
        self.lambda_  = self.cfg["lambda_"]
        self.global_step=0

        # Action scaling if continuous
        if not self.is_discrete:
            high = torch.tensor(self.action_space.high, dtype=torch.float32, device=device)
            low  = torch.tensor(self.action_space.low,  dtype=torch.float32, device=device)
            self.action_scale = (high - low)/2.
            self.action_bias  = (high + low)/2.
        else:
            self.action_scale = None
            self.action_bias  = None

    @property
    def alpha_ext(self):
        return torch.exp(self.log_alpha_ext)
    @property
    def alpha_int(self):
        return torch.exp(self.log_alpha_int)

    def reset(self):
        pass

    @torch.no_grad()
    def act(self, obs, sample=True):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        # No RNN mgmt for inference example
        d1, d2, st = self.world_model.rssm.init_state(1)
        lat = torch.cat([d1, d2, st], dim=-1)
        a, dist = self.actor(lat, sample=sample)
        if not self.is_discrete:
            a = a*self.action_scale + self.action_bias
        return a.cpu().numpy()[0]

    def store_transition(self, transition):
        """
        We can store a disagreement-based priority if we want:
        For now, we pass priority=1.
        """
        prio = 1.0
        self.buffer.add_transition(transition, priority=prio)
        self.reward_rms.update(np.array([transition["reward"]], dtype=np.float32))

    def train(self, num_updates=1):
        logs=[]
        loader_iter= iter(self.dataloader)
        for _ in range(num_updates):
            try:
                batch= next(loader_iter)
            except StopIteration:
                loader_iter= iter(self.dataloader)
                try:
                    batch= next(loader_iter)
                except StopIteration:
                    break
            if batch is None:
                break

            obs     = batch["obs"].to(device)
            actions = batch["action"].to(device)
            rewards = batch["reward"].to(device)
            dones   = batch["done"].to(device)
            IS_w    = batch["IS_weight"].to(device) # Weighted IS from PER
            B, T= obs.shape[0], obs.shape[1]

            # 1) reward normalization
            rew_mean= torch.tensor(self.reward_rms.mean,dtype=torch.float32,device=device)
            rew_std= torch.tensor(self.reward_rms.std, dtype=torch.float32,device=device)
            normed_rewards= (rewards - rew_mean)/(rew_std+1e-8)

            # ----------------------------------------------------------------
            # WORLD MODEL update (RSSM + ICM)
            # ----------------------------------------------------------------
            self.wm_opt.zero_grad()
            wm_loss_val=0.0
            state= self.world_model.init_state(B)
            for t in range(T):
                a_t= actions[:, t, :]
                o_t= obs[:, t, :]
                next_st, stats, r_pred, done_logit, st_pred= self.world_model(state, a_t, o_t)
                pm,plv, mm,mlv= stats
                klv= self.world_model.rssm.kl_loss((pm,plv), (mm,mlv)).mean()
                r_loss= F.mse_loss(r_pred, normed_rewards[:, t:t+1])
                d_loss= F.binary_cross_entropy_with_logits(done_logit, dones[:, t:t+1])
                st_next= next_st[2].detach()
                icm_err= F.mse_loss(st_pred, st_next)
                step_loss= klv + r_loss + d_loss + self.cfg["icm_scale"]*icm_err
                # Weighted IS
                step_loss= (step_loss*IS_w).mean()
                step_loss.backward(retain_graph=(t<T-1))
                wm_loss_val+= step_loss.item()
                state= next_st
            # AGC
            adaptive_gradient_clipping(self.wm_params, clip_factor=self.cfg["agc_clip"])
            self.wm_opt.step()
            # slow update
            soft_update_rssm(self.world_model.rssm, self.slow_rssm.rssm, tau=self.cfg["tau_rssm"])

            # ----------------------------------------------------------------
            # Plan2Explore => ensemble
            # ----------------------------------------------------------------
            self.ens_opt.zero_grad()
            ens_loss_val=0.0
            stt= self.world_model.init_state(B)
            for t in range(T):
                a_t= actions[:, t, :]
                o_t= obs[:, t, :]
                nxt, stt_stats= self.world_model.rssm(stt, a_t, o_t)
                st_next= nxt[2].detach()
                feat_c= torch.cat([nxt[0], nxt[1], st_next, a_t], dim=-1)
                preds= self.plan2explore(feat_c)
                e_loss= 0.0
                for outp in preds:
                    e_loss+= F.mse_loss(outp, st_next)
                e_loss= (e_loss*IS_w).mean() # Weighted
                e_loss.backward(retain_graph=(t<T-1))
                ens_loss_val+= e_loss.item()
                stt= nxt
            adaptive_gradient_clipping(self.ens_params, clip_factor=self.cfg["agc_clip"])
            self.ens_opt.step()

            # ----------------------------------------------------------------
            # Critic => lambda-return with return normalization
            # ----------------------------------------------------------------
            with torch.no_grad():
                lat_seq= []
                s0= self.world_model.init_state(B)
                for t in range(T):
                    nxt, stt_stats= self.world_model.rssm(s0, actions[:, t, :], obs[:, t, :])
                    lat_seq.append(torch.cat([nxt[0], nxt[1], nxt[2]], dim=-1).unsqueeze(1))
                    s0= nxt
                lat_seq= torch.cat(lat_seq, dim=1) # [B,T, lat_dim]

                # Intrinsic => plan2explore
                s0= self.world_model.init_state(B)
                int_rews_list=[]
                for t in range(T):
                    d1c,d2c, stc= s0
                    feat_c= torch.cat([d1c,d2c, stc, actions[:,t,:]], dim=-1)
                    disagree= self.plan2explore.disagreement(feat_c).squeeze(-1)
                    int_rews_list.append(disagree)
                    nx, stt_s= self.world_model.rssm(s0, actions[:,t,:], obs[:,t,:])
                    s0= nx
                int_rews= torch.stack(int_rews_list, dim=1)

                extr_scaled= self.alpha_ext*normed_rewards
                int_scaled = self.alpha_int*int_rews
                total_rw= extr_scaled + int_scaled

                # Lambda-return
                dones_mask = (1. - dones)
                values = self.critic(lat_seq).detach()
                G = torch.zeros_like(total_rw).to(device)
                lam = self.cfg["lambda_"]
                gam = self.cfg["discount"]
                for t in reversed(range(T)):
                    if t == T - 1:
                        G[:, t] = total_rw[:, t] + gam * dones_mask[:, t] * values[:, t]
                    else:
                        G[:, t] = total_rw[:, t] + gam * dones_mask[:, t] * (
                                (1. - lam) * values[:, t + 1] + lam * G[:, t + 1]
                        )
                # Return normalization
                # we store all G => update RMS
                self.return_rms.update(G.detach().cpu().numpy().reshape(-1))

                # Convert mean and std to Torch tensors on the same device as G
                meanR_t = torch.tensor(self.return_rms.mean, device=G.device, dtype=G.dtype)
                stdR_t = torch.tensor(self.return_rms.std, device=G.device, dtype=G.dtype)

                Rnorm = (G - meanR_t) / (stdR_t + 1e-8)

            vals_c = self.critic(lat_seq)  # [B,T]
            # Weighted IS for critic
            critic_losses = (F.mse_loss(vals_c, Rnorm, reduction='none') * IS_w.view(-1, 1)).mean()
            self.critic_opt.zero_grad()
            critic_losses.backward()
            adaptive_gradient_clipping(self.critic_params, clip_factor=self.cfg["agc_clip"])
            self.critic_opt.step()

            # ----------------------------------------------------------------
            # Actor => multi-step
            # ----------------------------------------------------------------
            actor_loss= self._multi_step_actor_loss(lat_seq.reshape(B*T, -1))
            self.actor_opt.zero_grad()
            actor_loss.backward()
            adaptive_gradient_clipping(self.actor_params, clip_factor=self.cfg["agc_clip"])
            self.actor_opt.step()

            # ----------------------------------------------------------------
            # trust region => KL => alpha_ext, alpha_int
            # minimal demonstration
            mean_kl= klv
            diff= mean_kl - self.cfg["target_kl"]
            hinge= F.relu(diff).detach()
            alpha_loss_ext= self.log_alpha_ext*hinge*self.cfg["trust_region_scale"]
            alpha_loss_int= self.log_alpha_int*hinge*self.cfg["trust_region_scale"]
            self.opt_alpha_ext.zero_grad()
            alpha_loss_ext.backward(retain_graph=True)
            self.opt_alpha_ext.step()
            self.opt_alpha_int.zero_grad()
            alpha_loss_int.backward()
            self.opt_alpha_int.step()

            # ----------------------------------------------------------------
            # LR sched
            # ----------------------------------------------------------------
            self.global_step+=1
            self.wm_sched.step()
            self.ens_sched.step()
            self.actor_sched.step()
            self.critic_sched.step()

            logs.append({
                "wm_loss": wm_loss_val/T,
                "ens_loss": ens_loss_val/T,
                "critic_loss": critic_losses.item(),
                "actor_loss": actor_loss.item(),
                "alpha_ext": float(self.alpha_ext.item()),
                "alpha_int": float(self.alpha_int.item()),
                "mean_kl": mean_kl.item(),
            })
        return logs

    def _multi_step_actor_loss(self, init_latents):
        """
        Multi-step imagination using the slow RSSM prior for stable rollouts.
        Summation of (value + alpha_int * KL) along horizon, then negative.
        """
        B_ = init_latents.shape[0]
        d= self.cfg["deter_dim"]
        s= self.cfg["stoch_dim"]
        d1= init_latents[:, :d]
        d2= init_latents[:, d:2*d]
        st= init_latents[:, 2*d:2*d+s]

        horizon= self.cfg["actor_horizon"]
        discount= 1.0
        total_val= 0.0
        for _ in range(horizon):
            lat= torch.cat([d1,d2,st], dim=-1)
            act, dist= self.actor(lat, sample=True)
            if not self.is_discrete:
                act_in= act
            else:
                oh= F.one_hot(act.long(), num_classes=self.act_dim).float()
                act_in= oh

            # use slow RSSM prior
            catd= torch.cat([d1,d2], dim=-1)
            prior_out= self.slow_rssm.rssm.prior_net(catd)
            pm, plv= torch.chunk(prior_out,2,dim=-1)
            st_next= pm + torch.exp(0.5*plv)*torch.randn_like(pm)

            # RNN steps
            x1= torch.cat([st, act_in], dim=-1)
            d1_next= self.slow_rssm.rssm.rnn1(x1, d1)
            d2_next= self.slow_rssm.rssm.rnn2(d1_next, d2)

            lat_next= torch.cat([d1_next, d2_next, st_next], dim=-1)
            val= self.critic(lat_next).mean()
            # alpha_int * KL
            kl_intr= 0.5*(plv.exp() + pm.pow(2) - plv -1).sum(dim=-1).mean()
            rew= val + self.alpha_int*kl_intr
            total_val+= discount*rew
            discount*= self.cfg["discount"]
            # update states
            d1= d1_next
            d2= d2_next
            st= st_next

        return -total_val

    def state(self):
        return {
            "world_model": self.world_model.state_dict(),
            "slow_rssm": self.slow_rssm.state_dict(),
            "plan2explore": self.plan2explore.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_alpha_ext": self.log_alpha_ext.data.clone(),
            "log_alpha_int": self.log_alpha_int.data.clone(),
            "opt_alpha_ext": self.opt_alpha_ext.state_dict(),
            "opt_alpha_int": self.opt_alpha_int.state_dict(),
            "wm_opt": self.wm_opt.state_dict(),
            "ens_opt": self.ens_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "global_step": self.global_step
        }

    def restore_state(self, ckpt):
        self.world_model.load_state_dict(ckpt["world_model"])
        self.slow_rssm.load_state_dict(ckpt["slow_rssm"])
        self.plan2explore.load_state_dict(ckpt["plan2explore"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.log_alpha_ext.data.copy_(ckpt["log_alpha_ext"])
        self.log_alpha_int.data.copy_(ckpt["log_alpha_int"])

        self.opt_alpha_ext.load_state_dict(ckpt["opt_alpha_ext"])
        self.opt_alpha_int.load_state_dict(ckpt["opt_alpha_int"])
        self.wm_opt.load_state_dict(ckpt["wm_opt"])
        self.ens_opt.load_state_dict(ckpt["ens_opt"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.global_step= ckpt["global_step"]


