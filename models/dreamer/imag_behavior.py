import torch
import copy
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

from models.dreamer.tools import (
    Optimizer,
    RequiresGrad,
    static_scan,
    lambda_return,
    tensorstats
)
from .world_model import MLP, RewardEMA, WorldModel


import torch
from torch import nn

class CustomOptimizer:
    """
    Kapselt einen PyTorch-Optimizer und optional AMP-GradScaler.
    Bietet:
      - zero_grad()
      - step()
      - backward(loss)
      - grad_clipping
      - mit oder ohne AMP (fp16).
    """

    def __init__(
        self,
        parameters,
        lr=1e-3,
        eps=1e-5,
        grad_clip=None,
        weight_decay=0.0,
        opt="adam",
        use_amp=False
    ):
        """
        Parameters
        ----------
        parameters: iterable
            Parameter deines Modells (z.B. model.parameters()).
        lr: float
            Learning Rate.
        eps: float
            Epsilon für den Optimizer.
        grad_clip: float oder None
            Maximaler Gradient-Norm. Falls None, kein Clipping.
        weight_decay: float
            Weight Decay, falls gewünscht.
        opt: str
            Name des Optimizers ("adam", "sgd" etc.)
        use_amp: bool
            Ob Automatic Mixed Precision genutzt wird.
        """
        self.grad_clip = grad_clip
        self.use_amp = use_amp

        # AMP GradScaler (für FP16-Training)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # PyTorch-Optimizer anlegen
        if opt.lower() == "adam":
            self._opt = torch.optim.Adam(
                parameters, lr=lr, eps=eps, weight_decay=weight_decay
            )
        elif opt.lower() == "sgd":
            self._opt = torch.optim.SGD(
                parameters, lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        else:
            raise NotImplementedError(f"Optimizer {opt} nicht implementiert.")

    def zero_grad(self):
        """Setzt die Gradienten aller Parameter auf 0."""
        self._opt.zero_grad()

    def step(self):
        """Führt einen Optimizer-Schritt aus (Parameterupdate)."""
        self._opt.step()

    def backward(self, loss, retain_graph=False):
        """Ruft loss.backward() mit optionalem AMP-Scaling."""
        if self.use_amp:
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

    def clip_grad(self):
        """Clipt Gradienten, falls grad_clip gesetzt."""
        if self.grad_clip is not None:
            if self.use_amp:
                # unscale_ entfernt das Skalieren, bevor wir clippen
                self.scaler.unscale_(self._opt)
            torch.nn.utils.clip_grad_norm_(
                self._opt.param_groups[0]["params"], self.grad_clip
            )

    def update(self, loss, retain_graph=False):
        """
        Kürzere "all-in-one"-Methode:
          1) zero_grad
          2) backward
          3) clip_grad
          4) step (mit/ohne AMP).
        Falls du die Steps manuell steuern willst,
        kannst du auch jede Methode einzeln aufrufen.
        """
        self.zero_grad()
        self.backward(loss, retain_graph=retain_graph)
        self.clip_grad()

        if self.use_amp:
            # Bei AMP muss man scaler.step() und scaler.update() statt _opt.step() machen:
            self.scaler.step(self._opt)
            self.scaler.update()
        else:
            self.step()


class ImagBehavior(nn.Module):
    """
    Trainingsklasse für die "Imagination":
     - Actor (Policy) auf dem RSSM-Feature
     - Critic (Value)
     - Nutzt WorldModel für imaginative Rollouts
    """
    def __init__(self, config, world_model):
        super().__init__()
        self._config = config
        self._world_model = world_model
        self._use_amp = (config["precision"] == 16)

        # Feature-Size aus RSSM


        # Actor
        # Angenommen, du hast in der neuen RSSM:
        # stoch_dim = 8, deter_dim = 10  => get_feat liefert 18
        inp_dim = config.get("actor_inp_dim", 18)  # Setze hier 18 ein oder berechne es: stoch_dim + deter_dim

        feat_size = config["dyn_stoch"] + config["dyn_deter"]

        # Actor – korrekt, wenn er mit 96D arbeitet:
        self.actor = MLP(
            inp_dim=feat_size,  # also 96
            shape=config["num_actions"],
            layers=config["actor"]["layers"],
            units=192,
            act=config["act"],
            norm=config["norm"],
            dist="normal",
            std=1.0,
            min_std=0.1,
            max_std=1.0,
            device=config["device"],
            name="Actor"
        ).to(config["device"])

        # Critic (Value) – passe den inp_dim an, damit er ebenfalls 96D erwartet:
        self.value = MLP(
            inp_dim=feat_size,  # also 96, NICHT 18
            shape=1,
            layers=config["critic"]["layers"],
            units=config["units"],
            act=config["act"],
            norm=config["norm"],
            dist="normal",
            std=1.0,
            min_std=0.1,
            max_std=1.0,
            device=config["device"],
            name="Value"
        ).to(config["device"])



        def print_actor_architecture(actor):
            print("Actor-Architektur:")
            print(actor)
            print("\nParameter-Shapes:")
            for name, param in actor.named_parameters():
                print(f"{name}: {param.shape}")

        # Beispielaufruf – z. B. direkt nach der Initialisierung des Actors:
        print_actor_architecture(self.actor)



        # Optional Slow-Target
        if config["critic"].get("slow_target", False):
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0

        # Optimizer
        opt_args = dict(wd=config["weight_decay"], opt=config["opt"], use_amp=self._use_amp)
        self.actor_opt = CustomOptimizer(
            parameters=self.actor.parameters(),
            lr=config["actor"]["lr"],
            eps=config["actor"]["eps"],
            grad_clip=config["actor"]["grad_clip"],
            weight_decay=config["weight_decay"],
            opt=config["opt"],  # z.B. "adam"
            use_amp=self._use_amp,
        )

        self.critic_opt = CustomOptimizer(
            parameters=self.value.parameters(),
            lr=config["critic"]["lr"],
            eps=config["critic"]["eps"],
            grad_clip=config["critic"]["grad_clip"],
            weight_decay=config["weight_decay"],
            opt=config["opt"],
            use_amp=self._use_amp
        )

        # Optionale RewardEMA
        if config.get("reward_EMA", False):
            self.register_buffer("ema_vals", torch.zeros(2, device=config["device"]))
            self.reward_ema = RewardEMA(device=config["device"])

        # Debug-Hooks (optional; können entfernt werden)
        from functools import partial
        def print_grad(grad, name, param):
            print(f"[DEBUG] Parameter '{name}' grad computed, version: {param._version}")

        for name, param in self.actor.named_parameters():
            param.register_hook(partial(print_grad, name=name, param=param))
        for name, param in self.value.named_parameters():
            param.register_hook(partial(print_grad, name=name, param=param))

    def log_tensor_debug(tensor, name, num_samples=5):
        # Extrahiere CPU-Werte und wandle in numpy um
        t_cpu = tensor.detach().cpu()
        print(f"[DEBUG] {name}: shape = {t_cpu.shape}, "
              f"min = {t_cpu.min().item():.4f}, max = {t_cpu.max().item():.4f}, "
              f"mean = {t_cpu.mean().item():.4f}, "
              f"sample = {t_cpu.view(-1)[:num_samples].numpy()}")

    def _train(self, start, objective):
        """
        start: Posterior-State (Dictionary) aus dem WorldModel
        objective: Funktor, z. B. reward=lambda feat, state, act: ...
        """
        torch.autograd.set_detect_anomaly(True)
        print("[DEBUG] _train() gestartet.")

        self._update_slow_target()
        metrics = {}

        def log_tensor_debug(tensor, name, num_samples=5):
            # Extrahiere CPU-Werte und wandle in numpy um
            t_cpu = tensor.detach().cpu()
            print(f"[DEBUG] {name}: shape = {t_cpu.shape}, "
                  f"min = {t_cpu.min().item():.4f}, max = {t_cpu.max().item():.4f}, "
                  f"mean = {t_cpu.mean().item():.4f}, "
                  f"sample = {t_cpu.view(-1)[:num_samples].numpy()}")

        # ============================
        # Gemeinsame Vorwärtsrechnung
        # ============================
        with torch.amp.autocast("cuda", enabled=self._use_amp):
            # 1) Rollout in der Imagination
            imag_feat, imag_state, imag_action = self._imagine(start, self._config["imag_horizon"])
            log_tensor_debug(imag_feat, "imag_feat")
            #log_tensor_debug(imag_state, "imag_state")
            log_tensor_debug(imag_action, "imag_action")

            # 2) Rewards berechnen (z. B. aus dem Objective)
            reward = objective(imag_feat, imag_state, imag_action)
            log_tensor_debug(reward, "reward")

            # 3) Actor
            actor_out = self.actor(imag_feat)
            actor_ent = actor_out.entropy()
            log_tensor_debug(actor_ent, "actor_entropy")

            # 4) Targets für den Value berechnen (Lamda-Return etc.)
            target, weights, base = self._compute_target(imag_feat, imag_state, reward)
            # Logge einen Auszug des Target-Tensors (z. B. das erste Element jedes Zeitschritts)
            if isinstance(target, list) and target:
                target_tensor = torch.stack([t.detach() for t in target], dim=1)
                log_tensor_debug(target_tensor, "target")

            # 5) Actor-Loss
            actor_loss, actor_metrics = self._compute_actor_loss(
                imag_feat, imag_action, target, weights, base
            )
            actor_loss = actor_loss - self._config["actor"]["entropy"] * actor_ent[:-1, ..., None]
            actor_loss = actor_loss.mean()
            log_tensor_debug(actor_loss, "actor_loss")

            # 6) Critic (Value) auf den *detachten* Features
            value_input = imag_feat.detach()  # Entkopplung vom Actor-Graph
            value_dist = self.value(value_input[:-1])
            tar_stack_value = torch.stack([t.detach() for t in target], dim=1)  # (T, B, 1)
            value_loss = -value_dist.log_prob(tar_stack_value)
            if self._config["critic"].get("slow_target", False):
                slow_dist = self._slow_value(value_input[:-1])
                value_loss = value_loss - value_dist.log_prob(slow_dist.mode().detach())
            value_loss = (weights[:-1] * value_loss[..., None]).mean()
            log_tensor_debug(value_loss, "value_loss")

            # 7) Gesamtverlust
            total_loss = actor_loss + value_loss


        # ============================
        # Gemeinsamer Backward-Pass
        # ============================
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        print(total_loss.grad_fn)
        actor_loss.backward()
        self.actor_opt.clip_grad()
        self.critic_opt.clip_grad()
        self.actor_opt.step()
        self.critic_opt.step()

        # ============================
        # Logging / Metriken
        # ============================
        metrics.update(actor_metrics)
        metrics.update(tensorstats(self.value(value_input[:-1]).mode(), "value"))
        metrics.update(tensorstats(torch.stack(target, dim=1), "target"))
        metrics.update(tensorstats(reward, "imag_reward"))
        if self._config["actor"]["dist"] in ["onehot"]:
            metrics.update(tensorstats(torch.argmax(imag_action, dim=-1).float(), "imag_action"))
        else:
            metrics.update(tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = float(actor_ent.mean().detach().cpu())
        metrics["actor_loss"] = float(actor_loss.detach().cpu())
        metrics["critic_loss"] = float(value_loss.detach().cpu())

        print("[DEBUG] _train() abgeschlossen.")
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, horizon):
        # Flache die Zeitdimension ab: (B,T,...) -> (B*T,...)
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start_flat = {k: flatten(v) for k, v in start.items()}
        print("[DEBUG _imagine] start_flat shapes:", {k: v.shape for k, v in start_flat.items()})

        def step(prev, _):
            st = prev[0]
            # Falls st ein Dictionary ist, kann man z.B. den "stoch" oder "deter" Teil debuggen:
            if isinstance(st, dict):
                if "stoch" in st:
                    print("[DEBUG _imagine: step] st['stoch'] shape:", st["stoch"].shape)
                if "deter" in st:
                    print("[DEBUG _imagine: step] st['deter'] shape:", st["deter"].shape)
            else:
                print("[DEBUG _imagine: step] st shape:", st.shape)

            # Hole das Feature aus dem Zustand
            feat = self._world_model.dynamics.get_feat(st)
            print("[DEBUG _imagine: step] feat shape:", feat.shape)

            # Actor-Forward-Pass (verwende clone, um in-place Probleme zu vermeiden)
            policy_out = self.actor(feat.detach())
            print("[DEBUG _imagine: step] policy_out type:", type(policy_out))

            # Sample Aktion
            act = policy_out.sample()
            print("[DEBUG _imagine: step] act shape:", act.shape)

            # Berechne den nächsten Zustand mit dem prior_step
            next_st = self._world_model.dynamics.prior_step(st, act, sample=True)
            if isinstance(next_st, dict):
                print("[DEBUG _imagine: step] next_st keys:", next_st.keys())
                for key, value in next_st.items():
                    print(f"[DEBUG _imagine: step] next_st[{key}] shape:", value.shape)
            else:
                print("[DEBUG _imagine: step] next_st shape:", next_st.shape)

            return (next_st, feat, act)

        # Führe static_scan über den Zeitschritt aus
        scan_out = static_scan(step, [torch.arange(horizon)], (start_flat, None, None))
        states, feats, acts = scan_out

        # Debug-Ausgaben für die finalen Ergebnisse
        if isinstance(states, dict):
            print("[DEBUG _imagine] Final states shapes:", {k: v.shape for k, v in states.items()})
        else:
            print("[DEBUG _imagine] Final states shape:", states.shape)
        print("[DEBUG _imagine] Final feats shape:", feats.shape)
        print("[DEBUG _imagine] Final acts shape:", acts.shape)

        # Klonen, um Views auszuschließen
        if isinstance(states, dict):
            states = {k: v.clone() for k, v in states.items()}
        else:
            states = states.clone()
        feats = feats.clone()
        acts = acts.clone()

        return feats, states, acts



    def _compute_target(self, imag_feat, imag_state, reward):
        print("[DEBUG _compute_target] imag_feat shape:", imag_feat.shape)
        # Wenn ein Discount-Head vorhanden ist:
        if "cont" in self._world_model.heads:
            cont_inp = self._world_model.dynamics.get_feat(imag_state)
            print("[DEBUG _compute_target] cont_inp shape:", cont_inp.shape)
            discount_dist = self._world_model.heads["cont"](cont_inp).mean
            discount = self._config["discount"] * discount_dist
        else:
            discount = self._config["discount"] * torch.ones_like(reward)
        print("[DEBUG _compute_target] discount shape:", discount.shape)

        val = self.value(imag_feat).mode()
        print("[DEBUG _compute_target] value output shape:", val.shape)

        ret = lambda_return(
            reward[1:],  # (T, B, 1)
            val[:-1],  # (T, B, 1)
            discount[1:],  # (T, B, 1)
            bootstrap=val[-1],
            lambda_=self._config["discount_lambda"],
            axis=0
        )
        w = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], dim=0), dim=0
        ).detach()
        print("[DEBUG _compute_target] weights shape:", w.shape)
        base = val[:-1]
        return ret, w, base

    # --------------------------------------------------

    def _compute_actor_loss(self, imag_feat, imag_action, target, weights, baseline):
        """
        Berechnet den Actor-Loss (Policy-Gradient-Anteil) ohne in-place Operationen.
        """
        metrics = {}
        # Klone den Input, um sicherzugehen, dass keine Views weitergegeben werden.
        pol = self.actor(imag_feat.clone())
        print("[DEBUG _compute_actor_loss] Actor input shape:", imag_feat.clone().shape)

        # Staple target: Liste von T Tensoren zu (T, B, 1)
        tar_stack = torch.stack(target, dim=1)
        print("[DEBUG _compute_actor_loss] tar_stack shape:", tar_stack.shape)

        # Berechne Vorteil (adv)
        if self._config.get("reward_EMA", False):
            offset, scale = self.reward_ema(tar_stack, self.ema_vals)
            norm_t = (tar_stack - offset) / scale
            norm_b = (baseline - offset) / scale
            adv = norm_t - norm_b
            metrics.update(tensorstats(norm_t, "normed_target"))
            metrics["EMA_005"] = float(self.ema_vals[0])
            metrics["EMA_095"] = float(self.ema_vals[1])
            print("[DEBUG _compute_actor_loss] adv shape (with EMA):", adv.shape)
        else:
            adv = tar_stack - baseline
            print("[DEBUG _compute_actor_loss] adv shape (without EMA):", adv.shape)

        # Berechne actor_target abhängig von imag_gradient
        if self._config["imag_gradient"] == "dynamics":
            actor_target = adv
            print("[DEBUG _compute_actor_loss] Using dynamics; actor_target shape:", actor_target.shape)
        elif self._config["imag_gradient"] == "reinforce":
            logp = pol.log_prob(imag_action)
            print("[DEBUG _compute_actor_loss] logp shape before slicing:", logp.shape)
            logp = logp[:-1]
            if logp.ndim == 2:
                logp = logp.unsqueeze(-1)
            print("[DEBUG _compute_actor_loss] logp shape after slicing:", logp.shape)
            val_imag = self.value(imag_feat[:-1]).mode()
            print("[DEBUG _compute_actor_loss] val_imag shape:", val_imag.shape)
            adv_ = (tar_stack[:-1] - val_imag).detach()
            actor_target = logp * adv_
            print("[DEBUG _compute_actor_loss] actor_target shape (reinforce):", actor_target.shape)
        elif self._config["imag_gradient"] == "both":
            logp = pol.log_prob(imag_action)[:-1]
            if logp.ndim == 2:
                logp = logp.unsqueeze(-1)
            val_imag = self.value(imag_feat[:-1]).mode()
            adv_ = (tar_stack[:-1] - val_imag).detach()
            reinforce_part = logp * adv_
            mix = self._config["imag_gradient_mix"]
            dyn_part = adv[:-1]
            actor_target = mix * dyn_part + (1.0 - mix) * reinforce_part
            metrics["imag_gradient_mix"] = mix
            print("[DEBUG _compute_actor_loss] actor_target shape (both):", actor_target.shape)
        else:
            raise NotImplementedError(f"Unknown imag_gradient: {self._config['imag_gradient']}")

        # Gewichtung anpassen: weights (T, B) -> (T-1, B, 1)
        w_ = weights[:-1]
        while w_.ndim < actor_target.ndim:
            w_ = w_.unsqueeze(-1)
        print("[DEBUG _compute_actor_loss] weights shape after unsqueeze:", w_.shape)

        loss = -w_ * actor_target
        print("[DEBUG _compute_actor_loss] loss shape:", loss.shape)

        return loss, metrics

    def _update_slow_target(self):
        # Optional: Langsames Update des Value-Netzes
        if self._config["critic"].get("slow_target", False):
            if self._updates % self._config["critic"]["slow_target_update"] == 0:
                mix = self._config["critic"]["slow_target_fraction"]
                for p_s, p_t in zip(self.value.parameters(), self._slow_value.parameters()):
                    p_t.data = mix * p_s.data + (1 - mix) * p_t.data
            self._updates += 1


