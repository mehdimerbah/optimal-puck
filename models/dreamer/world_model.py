
################################################################
# OPTIONAL
################################################################
class RewardEMA:
    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)
    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as torchd

from models.dreamer.tools import (
    weight_init,
    uniform_weight_init,
    symlog,
    TanhBijector,
    ContDist,
    OneHotDist,
    Bernoulli,
    MSEDist,
    SafeTruncatedNormal,
    SampleDist,
    DiscDist,
    UnnormalizedHuber,
    SymlogDist,
    static_scan,
   # Optimizer,
    RequiresGrad,
)

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as torchd


################################################################
# Utilities
################################################################

def weight_init(m):
    """He initialization for linear layers (fan_in)."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

def uniform_weight_init(scale):
    """Returns a function that initializes weights uniformly in [-scale, scale]."""
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -scale, scale)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return _init

class RequiresGrad:
    """
    Context manager that enables grad for a module's parameters
    (even if they are globally set to `requires_grad=False`).
    Used for short sub-blocks, e.g. in training steps.
    """
    def __init__(self, module):
        self.module = module
        self.prev_state = None

    def __enter__(self):
        self.prev_state = []
        for p in self.module.parameters():
            self.prev_state.append(p.requires_grad)
            p.requires_grad_(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p, req in zip(self.module.parameters(), self.prev_state):
            p.requires_grad_(req)

class Optimizer:
    """
    Minimal wrapper for an optimizer that handles:
      - Optional gradient clipping
      - Optional weight decay
      - Autocast scaling if needed
    """
    def __init__(self, name, params, lr, eps, clip, weight_decay,
                 opt="adam", use_amp=False):
        self.name = name
        self.clip = clip
        self.use_amp = use_amp
        if opt == "adam":
            self.opt = torch.optim.Adam(
                params, lr=lr, eps=eps, weight_decay=weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {opt} not supported.")

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def __call__(self, loss, parameters, retain_graph=True):
        self.opt.zero_grad()
        self.scaler.scale(loss).backward(retain_graph=retain_graph)
        if self.clip:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(parameters, self.clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        return {f"{self.name}_loss": float(loss.detach().cpu())}


################################################################
# Distributions
################################################################



class BinaryDist:
    """Wrapper for Bernoulli(logits=...)."""
    def __init__(self, logits, absmax=None):
        self._dist = torchd.Bernoulli(logits=logits)
        self.absmax = absmax

    @property
    def mean(self):
        return self._dist.probs

    def sample(self, sample_shape=()):
        return self._dist.sample(sample_shape)

    def mode(self):
        return (self._dist.probs > 0.5).float()

    def log_prob(self, x):
        return self._dist.log_prob(x)

    def entropy(self):
        return self._dist.entropy()



################################################################
# Simple MLP
################################################################

class MLP(nn.Module):
    def __init__(self,
                 inp_dim,
                 shape,
                 layers,
                 units,
                 act="ReLU",
                 norm=True,
                 dist="normal",
                 std=1.0,
                 min_std=0.1,
                 max_std=1.0,
                 device="cuda",
                 name="MLP"):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = shape
        self._dist = dist
        self._std_str = std  # could be "learned" or float
        self._min_std = min_std
        self._max_std = max_std
        self._device = device

        act_fn = getattr(nn, act) if isinstance(act, str) else act
        self.layers = nn.Sequential()
        hidden_dim = inp_dim
        for i in range(layers):
            lin = nn.Linear(hidden_dim, units, bias=False)
            self.layers.add_module(f"{name}_lin{i}", lin)
            if norm:
                self.layers.add_module(f"{name}_ln{i}", nn.LayerNorm(units, eps=1e-3))
            self.layers.add_module(f"{name}_act{i}", act_fn())
            hidden_dim = units
        self.layers.apply(weight_init)

        out_dim = int(np.prod(self._shape))
        self.mean_layer = nn.Linear(hidden_dim, out_dim)
        self.mean_layer.apply(uniform_weight_init(1.0))

        if isinstance(std, str) and std == "learned":
            self.std_layer = nn.Linear(hidden_dim, out_dim)
            self.std_layer.apply(uniform_weight_init(1.0))
        else:
            self.std_layer = None

    def forward(self, x):
        print(f"[DEBUG {self.__class__.__name__}] (ID={id(self)}) Input shape: {x.shape}")
        h = self.layers(x)
        mean = self.mean_layer(h)
        if self.std_layer is not None:
            raw_std = self.std_layer(h)
            return self._build_dist(self._dist, mean, raw_std, self._shape)
        else:
            return self._build_dist(self._dist, mean, self._std_str, self._shape)

    def _build_dist(self, dist_type, mean, std_val, out_shape):
        if dist_type == "none":
            return mean  # raw output

        elif dist_type == "mse":
            # Return raw predictions to be used in an MSE loss
            return mean

        elif dist_type == "normal":
            # If std_val is a string 'learned', we already handled that above
            if isinstance(std_val, str):
                raise ValueError("std_val='learned' must be handled separately.")
            if not torch.is_tensor(std_val):
                std_val = torch.tensor([std_val], device=mean.device)
            # Limit range of std:
            std = (self._max_std - self._min_std) * torch.sigmoid(std_val + 2.0) + self._min_std
            # Optional saturating the mean, e.g. via tanh:
            mean = torch.tanh(mean)
            dist = torchd.Independent(torchd.Normal(mean, std), 1)
            return ContDist(dist)

        elif dist_type == "binary":
            # Interpreted as logits for Bernoulli
            return BinaryDist(mean)

        else:
            raise NotImplementedError(f"Unknown distribution type: {dist_type}")


################################################################
# GRUCell (Deterministic Recurrent Unit)
################################################################




class GRUCell(nn.Module):
    def __init__(self, inp_size, hidden_size, act="SiLU", norm=True, update_bias=-1.0):
        """
        Einfacher GRU-Cell, der ein 'inp_size + hidden_size' -> '3*hidden_size'-Mapping nutzt.
        Robust gegenüber in-place Modifikationen und unsachgemäßen Aktivierungsfunktionen.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.update_bias = update_bias

        # Aktivierung: Falls ein String übergeben wird, versuche die entsprechende nn-Klasse zu holen.
        if isinstance(act, str):
            try:
                act_fn = getattr(nn, act)
                self._act = act_fn()  # Instanz der Aktivierungsfunktion
            except Exception as e:
                raise ValueError(f"Fehler beim Abrufen der Aktivierungsfunktion '{act}': {e}")
        elif callable(act):
            self._act = act
        else:
            raise TypeError("Parameter 'act' muss entweder ein String oder eine callable Funktion sein.")

        # Erstelle die lineare Schicht (mit optionaler LayerNorm)
        layers = [nn.Linear(inp_size + hidden_size, 3 * hidden_size, bias=False)]
        if norm:
            layers.append(nn.LayerNorm(3 * hidden_size, eps=1e-3))
        self.fc = nn.Sequential(*layers)

    def forward(self, x, h):
        """
        x: Tensor der Form (B, inp_size)
        h: Tensor der Form (B, hidden_size)
        Gibt den neuen Hidden-State h_new zurück.
        """
        # Klone Eingaben, um Seiteneffekte (in-place Modifikationen) zu vermeiden
        x_safe = x.clone()
        h_safe = h.clone()

        # Eingaben zusammenfügen
        combined = torch.cat([x_safe, h_safe], dim=-1)
        gates = self.fc(combined)  # (B, 3*hidden_size)

        # Splitte die Gates in Reset, Candidate und Update
        reset, cand, update = torch.split(gates, self.hidden_size, dim=-1)

        # Wende Sigmoid an (out-of-place Operationen)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update + self.update_bias)

        # Prüfe, ob die Aktivierungsfunktion callable ist
        if not callable(self._act):
            raise TypeError("Die Aktivierungsfunktion ist nicht callable.")

        # Berechne den Candidate-Zustand – dabei wird cand explizit geklont,
        # um in-place Modifikationen zu vermeiden.
        candidate_input = reset * cand.clone()
        candidate = self._act(candidate_input)

        # Neuer Hidden-State als Mischung aus Candidate und bisherigem Zustand
        h_new = update * candidate + (1 - update) * h_safe
        return h_new

class RSSM(nn.Module):
    """
    Vereinfachtes RSSM für 1D-Daten:
      - stoch: dimension des stochastischen Zustands
      - deter: dimension des deterministischen (GRU) Zustands
      - obs_dim: dimension der Beobachtung
      - act_dim: dimension der Aktion
    """
    def __init__(
        self,
        stoch_dim=32,
        deter_dim=32,
        obs_dim=18,
        act_dim=4,
        hidden=32,
        norm=True,
        min_std=0.1,
        device="cuda",
        initial="learned",
        depth=1,
        act="SiLU",
    ):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden
        self._device = device
        self._min_std = min_std
        self._initial = initial
        self._depth = depth

        # 1) Netz für Prior-Eingang: (stoch + action) -> hidden
        self.prior_in = nn.Sequential(
            nn.Linear(stoch_dim + act_dim, hidden, bias=False),
            nn.LayerNorm(hidden, eps=1e-3) if norm else nn.Identity(),
            getattr(nn, act)() if isinstance(act, str) else act
        )

        # 2) GRU-Cell, um deter zu aktualisieren
        self.gru_cell = GRUCell(hidden, deter_dim, act=act, norm=norm)

        # 3) Netz für Prior-Output: (deter) -> hidden -> suff. stats -> stoch
        self.prior_out = nn.Sequential(
            nn.Linear(deter_dim, hidden, bias=False),
            nn.LayerNorm(hidden, eps=1e-3) if norm else nn.Identity(),
            getattr(nn, act)() if isinstance(act, str) else act,
        )
        self.prior_stats = nn.Linear(hidden, 2 * stoch_dim)

        # 4) Netz für Posterior: (deter + obs) -> hidden -> stoch
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + obs_dim, hidden, bias=False),
            nn.LayerNorm(hidden, eps=1e-3) if norm else nn.Identity(),
            getattr(nn, act)() if isinstance(act, str) else act
        )
        self.post_stats = nn.Linear(hidden, 2 * stoch_dim)

        # Optional: initial deter state
        if self._initial == "learned":
            self.initial_deter = nn.Parameter(torch.zeros((1, deter_dim)), requires_grad=True)
        elif self._initial == "zeros":
            self.initial_deter = None
        else:
            raise NotImplementedError("Unknown initial mode")

    def initial_state(self, batch_size):
        """
        Erzeuge initialen RSSM-State (deter + stoch).
        """
        device = self._device
        if self._initial == "learned":
            deter_init = torch.tanh(self.initial_deter).expand(batch_size, -1)
        else:
            deter_init = torch.zeros(batch_size, self.deter_dim, device=device)

        # Erzeuge initial stoch durch prior_out
        h = self.prior_out(deter_init)  # (B, hidden)
        stats = self._stats_fn(self.prior_stats(h))  # dict: mean, std
        dist = self._build_dist(stats["mean"], stats["std"])
        stoch_init = dist.mean  # oder .sample()

        return {
            "deter": deter_init,           # (B, deter_dim)
            "stoch": stoch_init,           # (B, stoch_dim)
            "mean": stats["mean"],         # (B, stoch_dim)
            "std": stats["std"]            # (B, stoch_dim)
        }

    def observe(self, obs, action, is_first, state=None, sample=True):
        """
        Berechnet Posterior und Prior über eine Folge von T Schritten.
        obs: (B, T, obs_dim)
        action: (B, T, act_dim)
        is_first: (B, T) 0/1 -> Reset Markierungen
        state: optional Startzustand
        Returns: post, prior (dicts mit shape (B,T,...) pro key).
        """
        B, T, _ = obs.shape
        if state is None:
            state = self.initial_state(B)

        post = {k: [] for k in state.keys()}
        prior = {k: [] for k in state.keys()}
        prev_state = state

        for t in range(T):
            # Reset, falls is_first=1
            mask = (is_first[:, t] > 0.5)
            if mask.any():
                init_s = self.initial_state(mask.sum())
                # state-keys: deter, stoch, mean, std
                # wir müssen die betroffenen Einträge ersetzen
                idxs = mask.nonzero(as_tuple=True)[0]
                for k in prev_state.keys():
                    prev_state[k][idxs] = init_s[k]

                # action auch ggf. nullen
                action[:, t] = torch.where(
                    mask.unsqueeze(-1),
                    torch.zeros_like(action[:, t]),
                    action[:, t]
                )

            pr = self.prior_step(prev_state, action[:, t], sample=sample)
            po = self.post_step(pr, obs[:, t], sample=sample)

            for k in po.keys():
                post[k].append(po[k].unsqueeze(1))
            for k in pr.keys():
                prior[k].append(pr[k].unsqueeze(1))

            prev_state = po  # Posterior wird next state

        # Stapeln entlang der Zeitachse
        post = {k: torch.cat(v, dim=1) for k, v in post.items()}
        prior = {k: torch.cat(v, dim=1) for k, v in prior.items()}
        return post, prior

    def prior_step(self, prev_state, action, sample=True):
        """
        p(s_t | s_{t-1}, a_{t-1}).
        """
        x = torch.cat([prev_state["stoch"], action], dim=-1)  # (B, stoch_dim+act_dim)
        h = self.prior_in(x)  # (B, hidden)

        # GRU Update
        deter = prev_state["deter"]
        for _ in range(self._depth):
            deter = self.gru_cell(h, deter)

        # Stats aus deter
        h2 = self.prior_out(deter)
        stats = self._stats_fn(self.prior_stats(h2))
        dist = self._build_dist(stats["mean"], stats["std"])
        if sample:
            stoch = dist.sample()
        else:
            stoch = dist.mean

        return {
            "deter": deter,
            "stoch": stoch,
            "mean": stats["mean"],
            "std": stats["std"]
        }

    def post_step(self, prior_state, obs, sample=True):
        """
        Posterior: q(s_t | deter, obs).
        """
        x = torch.cat([prior_state["deter"], obs], dim=-1)
        h = self.post_net(x)  # (B, hidden)
        stats = self._stats_fn(self.post_stats(h))
        dist = self._build_dist(stats["mean"], stats["std"])

        if sample:
            stoch = dist.sample()
        else:
            stoch = dist.mean

        return {
            "deter": prior_state["deter"],
            "stoch": stoch,
            "mean": stats["mean"],
            "std": stats["std"]
        }



    def get_feat(self, state):
        feat = torch.cat([state["stoch"], state["deter"]], dim=-1)
        print("[DEBUG RSSM] get_feat output shape:", feat.shape)
        return feat

    def _stats_fn(self, out):
        """
        out: (B, 2*stoch_dim)
        Splittet in mean, std und wendet min_std an.
        """
        mean, std_ = torch.split(out, self.stoch_dim, dim=-1)
        # Softplus etc. – hier mal Softplus
        std_ = F.softplus(std_) + self._min_std
        return {"mean": mean, "std": std_}

    def _build_dist(self, mean, std):
        """
        Baut eine Normalverteilung (Independent) aus mean und std.
        """
        dist = torchd.Normal(mean, std)
        return torchd.Independent(dist, 1)

    def get_dist(self, stats):
        """
        Baut eine Independent Normalverteilung aus den gegebenen Statistikwerten.
        stats: Dict mit den Schlüsseln "mean" und "std", jeweils mit Shape (B, …, stoch_dim)
        """
        mean, std = stats["mean"], stats["std"]
        base_dist = torchd.Normal(mean, std)
        return torchd.Independent(base_dist, 1)

    def kl_loss(self, post, prior, free=1.0, dyn_scale=1.0, rep_scale=1.0):
        """
        Berechnet den KL-Verlust zwischen Posterior und Prior.

        post, prior: Dicts mit Schlüsseln "mean" und "std" (Shape: (B, T, stoch_dim)).
        free: Minimalwert (free bits) für den KL-Verlust.
        dyn_scale, rep_scale: Skalierungsfaktoren für zusätzliche Verluste.

        Returns:
            kl: der eigentliche KL-Verlust (B, T)
            rep_loss: z. B. ein Verlust zur Repräsentation (hier rep_scale * kl)
            dyn_loss: z. B. ein dynamischer Verlust (hier dyn_scale * kl)
            rep_loss2: Platzhalter, z. B. gleich rep_loss
        """
        # Erzeuge die Verteilungen:
        dist_post = torchd.Normal(post["mean"], post["std"])
        dist_post = torchd.Independent(dist_post, 1)
        dist_prior = torchd.Normal(prior["mean"], prior["std"])
        dist_prior = torchd.Independent(dist_prior, 1)

        # Berechne den KL-Verlust:
        kl = torchd.kl.kl_divergence(dist_post, dist_prior)
        # Free bits trick:
        kl = torch.clamp(kl, min=free)

        # Berechne zusätzliche Verluste (hier als einfache Skalierungen):
        rep_loss = rep_scale * kl
        dyn_loss = dyn_scale * kl
        rep_loss2 = rep_loss  # z. B. identisch als Platzhalter

        return kl, rep_loss, dyn_loss, rep_loss2


################################################################
# WorldModel
################################################################

class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super().__init__()
        self._config = config
        self._step   = step
        self._use_amp= (config["precision"] == 16)

        # Example obs_dim=18
        self.obs_dim= obs_space.shape[0]
        self.embed_size = config["encoder"]["embed_dim"]

        # 1) Encoder
        self.encoder = MLP(
            inp_dim=self.obs_dim,
            shape=self.embed_size,  # e.g. 128
            layers=config["encoder"]["layers"],
            units=192,
            act=config["act"],
            norm=config["norm"],
            dist="none",  # direct features
            std=1.0,
            min_std=0.1,
            max_std=1.0,
            device=config["device"],
            name="Encoder"
        ).to(config["device"])

        self.dynamics = RSSM(
            stoch_dim=config["stoch_dim"],
            deter_dim=config["dyn_deter"],
            obs_dim=self.embed_size,  # Dimension der eingebetteten Beobachtungen (z.B. 18 für 1D-Daten)
            act_dim=config["num_actions"],
            hidden=config["dyn_hidden"],
            norm=config["norm"],
            min_std=config["dyn_min_std"],
            initial=config["initial"],
            depth=1,
            act=config["act"],
            device=config["device"]
        ).to(config["device"])

        # Feature-size from RSSM
        feat_size = config["dyn_stoch"] + config["dyn_deter"]

        # 3) Heads
        self.heads = nn.ModuleDict()

        # Obs Decoder
        self.heads["obs_decoder"] = MLP(
            inp_dim=feat_size,
            shape=18,
            layers=config["decoder"]["layers"],
            units=512,
            act="SiLU",
            norm=True,
            dist="mse",  # will do MSE with ground truth
            std=1.0,
            min_std=0.1,
            max_std=1.0,
            device=config["device"],
            name="ObsDecoder"
        ).to(config["device"])

        # Reward
        self.heads["reward"] = MLP(
            inp_dim=feat_size,
            shape=1,
            layers=config["reward_head"]["layers"],
            units=192,
            act="SiLU",
            norm=True,
            dist="normal",  # Normal => log_prob can be used
            std=1.0,
            min_std=0.1,
            max_std=1.0,
            device=config["device"],
            name="Reward"
        ).to(config["device"])

        # Continuation
        self.heads["cont"] = MLP(
            inp_dim=feat_size,
            shape=1,
            layers=config["cont_head"]["layers"],
            units=192,
            act="SiLU",
            norm=True,
            dist="binary",
            std=1.0,  # unused for binary
            min_std=0.1,
            max_std=1.0,
            device=config["device"],
            name="Cont"
        ).to(config["device"])

        # Which heads receive gradient from the Dreamer loss
        for n in config["grad_heads"]:
            if n not in self.heads:
                raise ValueError(f"Grad head {n} not found in heads: {list(self.heads.keys())}")

        # 4) Optimizer
        self._model_opt = Optimizer(
            "model",
            self.parameters(),
            config["model_lr"],
            config["opt_eps"],
            config["grad_clip"],
            config["weight_decay"],
            opt=config["opt"],
            use_amp=self._use_amp
        )

        # Scales for each head’s loss
        self._scales = {
            "reward": config["reward_head"]["loss_scale"],
            "cont":   config["cont_head"]["loss_scale"]
        }

        total_params = sum(p.numel() for p in self.parameters())
        print(f"[WorldModel] Created with {total_params} parameters total.")

    def preprocess(self, data):
        dev= self._config["device"]
        out= {}
        for k,v in data.items():
            out[k] = v.clone().detach().to(device=dev, dtype=torch.float32)
        # discount or is_terminal => cont
        if "discount" in out:
            out["discount"] *= self._config["discount"]
            out["discount"]  = out["discount"].unsqueeze(-1)
        if "is_terminal" in out:
            out["cont"] = (1.0 - out["is_terminal"]).unsqueeze(-1)
        return out

    # Beispielhafter Ausschnitt aus der _train-Methode in world_model.py:

    def _train(self, data):
        data = self.preprocess(data)
        with RequiresGrad(self):
            # Verwende torch.amp.autocast statt torch.cuda.amp.autocast
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                # 1) Encode: (B,T,embed_dim)
                embed = self.encoder(data["obs"])
                # 2) RSSM: Berechne Posterior und Prior
                post, prior = self.dynamics.observe(embed, data["action"], data["is_first"])
                # 3) KL-Verlust
                kl_free = self._config["kl_free"]
                dyn_scale = self._config["dyn_scale"]
                rep_scale = self._config["rep_scale"]
                kl_loss, rep_loss, dyn_loss, rep_loss2 = self.dynamics.kl_loss(post, prior, kl_free, dyn_scale,
                                                                               rep_scale)
                # 4) Extrahiere Features
                feat = self.dynamics.get_feat(post)
                # 5) Wende alle Heads an:
                preds = {}
                for name, head in self.heads.items():
                    grad = (name in self._config["grad_heads"])
                    finp = feat if grad else feat.detach()
                    preds[name] = head(finp)
                # 6) Berechne die Verluste für jeden Head:
                losses = {}
                for name, output in preds.items():
                    # Wähle das passende Ziel:
                    if name == "obs_decoder":
                        truth = data["obs"]  # Form: (B,T,obs_dim)
                    else:
                        truth = data[name]  # z.B. reward oder cont (Form: (B,T) oder (B,T,1))
                    if truth.ndim == 2:
                        truth = truth.unsqueeze(-1)  # (B,T,1)
                    if self.heads[name]._dist == "mse":
                        # Bei MSE arbeiten wir direkt mit Tensoren:
                        while truth.ndim < output.ndim:
                            truth = truth.unsqueeze(-1)
                        loss_val = F.mse_loss(output, truth, reduction="none")
                    else:
                        # Hier greifen wir auf den Mittelwert der zugrunde liegenden Distribution zu:
                        while truth.ndim < output._dist.mean.ndim:
                            truth = truth.unsqueeze(-1)
                        loss_val = -output.log_prob(truth)
                    # Falls nötig, reduziere über die letzte Dimension:
                    if loss_val.ndim == 3:
                        if loss_val.shape[-1] != 1:
                            loss_val = loss_val.mean(dim=-1)
                        else:
                            loss_val = loss_val.squeeze(-1)
                    losses[name] = loss_val
                # 7) Skaliere die Verluste:
                scaled_losses = {k: self._scales.get(k, 1.0) * v for k, v in losses.items()}
                heads_loss = sum(scaled_losses.values())
                model_loss = heads_loss + kl_loss
            final_loss = torch.mean(model_loss)
            metrics = self._model_opt(final_loss, self.parameters())
        metrics["kl_loss"] = float(kl_loss.mean().detach().cpu())
        metrics["rep_loss"] = float(rep_loss.mean().detach().cpu())
        metrics["dyn_loss"] = float(dyn_loss.mean().detach().cpu())
        with torch.amp.autocast("cuda", enabled=self._use_amp):
            post_ent = self.dynamics.get_dist(post).entropy()
            prior_ent = self.dynamics.get_dist(prior).entropy()
        metrics["prior_ent"] = float(post_ent.mean().detach().cpu())
        metrics["post_ent"] = float(prior_ent.mean().detach().cpu())

        return post, metrics





