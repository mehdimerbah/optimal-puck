"""
DREAM.py

Ein Agent-Skript, das ein WorldModel (mit ImagBehavior) nutzt und verschiedene
Explorations-Module (Random, Plan2Explore) einbindet. Dazu kommt ein Replay-Puffer,
Priorizierte Sequences und ein Agent, der all das zusammenfasst.
"""

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from gymnasium import spaces

# Eventuell wichtig, falls du MuJoCo oder andere Grafiksachen benötigst
os.environ["MUJOCO_GL"] = "osmesa"

# ---------------------------------------------------------
# 1) Imports deiner Module (wenn sie in anderen Dateien liegen)
# ---------------------------------------------------------
# Passe die folgenden Imports an deinen tatsächlichen Pfad an.
# z. B.:
# from models.dreamer.world_model import WorldModel
# from models.dreamer.imag_behavior import ImagBehavior
# from models.dreamer.exploration import Random, Plan2Explore
# from models.dreamer.Replay import Replay

from models.dreamer.world_model import WorldModel
from models.dreamer.imag_behavior import ImagBehavior
from models.dreamer.exploration import Random, Plan2Explore
from models.dreamer.Replay import Replay

# Falls du Tools wie `tools.Every`, `tools.Once` etc. brauchst, importiere sie hier
# from models.dreamer import tools

# Optional: Für Kompilierung via PyTorch 2.0
USE_TORCH_COMPILE = False

# ---------------------------------------------------------
# 2) Device-Einstellungen und Hilfsfunktionen
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)  # optional

def adaptive_gradient_clipping(parameters, clip_factor=0.05, eps=1e-3):
    """
    Optional: AGC (Adaptive Gradient Clipping).
    """
    params = [p for p in parameters if p.grad is not None]
    for p in params:
        param_norm = p.data.norm(2)
        grad_norm = p.grad.data.norm(2)
        if param_norm == 0 or grad_norm == 0:
            continue
        ratio = grad_norm / (param_norm + eps)
        if ratio > clip_factor:
            p.grad.data.mul_(clip_factor / (ratio + eps))

class UnsupportedSpace(Exception):
    pass

# ---------------------------------------------------------
# 3) RunningMeanStd (z. B. für Reward-Normalisierung)
# ---------------------------------------------------------
class RunningMeanStd:
    def __init__(self, epsilon=1e-5, shape=()):
        self.epsilon = epsilon
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64)

    def update(self, x):
        x = np.asarray(x)
        self.count += x.shape[0]
        bmean = x.mean(axis=0)
        delta = bmean - self.mean
        self.mean += delta * (x.shape[0] / self.count)
        delta2 = bmean - self.mean
        self.M2 += ((x - self.mean)**2).sum()

    @property
    def var(self):
        return self.M2 / max(self.count, 1)

    @property
    def std(self):
        return np.sqrt(self.var + self.epsilon)

    def normalize(self, x):
        return (x - self.mean) / (self.std + 1e-8)

# ---------------------------------------------------------
# 4) Replay + PrioritizedSequenceDataset
# ---------------------------------------------------------
class PrioritizedSequenceDataset(Dataset):
    """
    Dataset für Sequenz-Sampling aus dem Replay Buffer mit Priorisierung (PER).
    """
    def __init__(self, replay_memory, obs_dim, act_dim):
        super().__init__()
        self.replay_memory = replay_memory
        self.max_seq_len = replay_memory.length
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def __len__(self):
        return max(1, len(self.replay_memory))

    def __getitem__(self, idx):
        seq, weight = None, None
        # Versuche mehrfach, eine gültige Sequenz zu sampeln
        for _ in range(10):
            seq, weight = self.replay_memory.sample_subsequence(
                max_seq_len=self.max_seq_len
            )
            if seq is not None:
                break
        # Fallback, falls gar keine Sequenz
        if seq is None:
            dummy_obs = np.zeros((self.max_seq_len, self.obs_dim), dtype=np.float32)
            dummy_act = np.zeros((self.max_seq_len, self.act_dim), dtype=np.float32)
            dummy_rew = np.zeros((self.max_seq_len,), dtype=np.float32)
            dummy_done = np.ones((self.max_seq_len,), dtype=np.float32)
            dummy_weight = np.array(1.0, dtype=np.float32)
            batch_dict = {
                "obs": dummy_obs,
                "action": dummy_act,
                "reward": dummy_rew,
                "done": dummy_done,
                "IS_weight": dummy_weight,
            }
            print("[DEBUG] Returning dummy batch (no valid sequence found).")
            return batch_dict

        # Baue den Batch-Dict
        batch_dict = {}
        keys = seq[0].keys()
        for k in keys:
            try:
                batch_dict[k] = np.array([step[k] for step in seq], dtype=np.float32)
            except Exception as e:
                print(f"[DEBUG] Error converting key {k}: {e}")
                batch_dict[k] = [step[k] for step in seq]
        batch_dict["IS_weight"] = np.array(weight, dtype=np.float32)
        return batch_dict

# ---------------------------------------------------------
# 5) Hilfsfunktionen: count_steps, make_dataset, make_env
#    Falls du sie brauchst, kannst du sie hier definieren
# ---------------------------------------------------------
def count_steps(folder):
    """
    Beispiel: Zählt .npz-Dateien in 'folder' und extrahiert Steps,
    falls du Episode-Files speicherst.
    """
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))

def make_dataset(episodes, config):
    """
    Falls du Episoden in Memory hast und daraus
    ein Batch-Dataset bauen willst.
    """
    raise NotImplementedError("Implementiere, wenn du offline/online Daten hast.")

def make_env(config, mode, idx):
    """
    Erstelle deine Umgebung.
    Diese Funktion ist ein Minimalstub:
    """
    raise NotImplementedError("Bitte implementiere, falls du parallele Envs erstellst.")

# ---------------------------------------------------------
# 6) Agent-Klasse DreamerV3Agent
# ---------------------------------------------------------
class DreamerV3Agent:
    """
    Dreamer-ähnlicher Agent, der WorldModel + ImagBehavior + Explorationsmodule nutzt.
    """
    def __init__(self, observation_space, action_space, **config):
        defaults = {
            "model_lr": 1e-3,
            "batch_size": 16,
            "seq_len": 50,
            "buffer_size": 1e5,
            "use_per": True,
            "agc_clip": 0.05,
            "discount": 0.99,
            "lambda_": 0.95,
            "max_train_steps": 1e6,
            "expl_behavior": "random",  # "greedy", "random", "plan2explore"
            "expl_until": 1e5,
        }
        defaults.update(config)
        self.cfg = defaults
        self.device = device

        # Falls du numerische Strings im config hast -> konvertiere
        def convert_numeric_values(d):
            if isinstance(d, dict):
                return {k: convert_numeric_values(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_numeric_values(item) for item in d]
            elif isinstance(d, str):
                try:
                    return int(d)
                except ValueError:
                    try:
                        return float(d)
                    except ValueError:
                        return d
            else:
                return d
        self.cfg = convert_numeric_values(self.cfg)


        # Check Obs
        if not isinstance(observation_space, spaces.Box):
            raise UnsupportedSpace("DreamerV3Agent requires a Box observation space.")
        self.obs_dim = observation_space.shape[0]
        self.action_space = action_space

        # Check Action
        if isinstance(action_space, spaces.Discrete):
            self.is_discrete = True
            self.act_dim = action_space.n
        else:
            self.is_discrete = False
            self.act_dim = action_space.shape[0]

        # Füge num_actions hinzu
        if "num_actions" not in self.cfg:
            self.cfg["num_actions"] = self.act_dim

        # 1) Erzeuge World Model
        self.world_model = WorldModel(
            obs_space=observation_space,
            act_space=action_space,
            step=0,
            config=self.cfg
        ).to(self.device)

        # 2) ImagBehavior
        self.imag_behavior = ImagBehavior(self.cfg, self.world_model).to(self.device)

        # PyTorch 2.0 compile
        if USE_TORCH_COMPILE and hasattr(torch, "compile"):
            self.world_model = torch.compile(self.world_model)
            self.imag_behavior = torch.compile(self.imag_behavior)

        # 3) Explorationsverhalten
        def reward_fn(feat, state, act):
            return self.world_model.heads["reward"](feat).mean()

        expl_map = {
            "greedy": lambda: self.imag_behavior,
            "random": lambda: Random(self.cfg, action_space),
            "plan2explore": lambda: Plan2Explore(self.cfg, self.world_model, reward_fn),
        }
        self._expl_behavior = expl_map[self.cfg["expl_behavior"]]()
        self._expl_behavior.to(self.device)

        # 4) Replay Buffer
        if self.cfg["use_per"]:
            self.buffer = Replay(
                length=int(self.cfg["seq_len"]),
                capacity=int(self.cfg["buffer_size"]),
                directory=self.cfg.get("replay_dir", None),
                chunksize=1024,
                online=False,
                use_priority=True,
                seed=self.cfg.get("seed", 42)
            )
            self.dataset = PrioritizedSequenceDataset(self.buffer, self.obs_dim, self.act_dim)
        else:
            raise NotImplementedError("Only PER is implemented for now.")

        # DataLoader
        from torch.utils.data import DataLoader
        def custom_collate_fn(batch):
            collated = {}
            keys = batch[0].keys()
            for key in keys:
                if key == "IS_weight":
                    collated[key] = torch.stack([torch.tensor(sample[key]) for sample in batch])
                else:
                    max_len = max(sample[key].shape[0] for sample in batch)
                    padded_tensors = []
                    for sample in batch:
                        tensor = torch.tensor(sample[key])
                        pad_size = max_len - tensor.shape[0]
                        if pad_size > 0:
                            pad = (0, 0) * (tensor.dim() - 1) + (0, pad_size)
                            tensor = F.pad(tensor, pad, "constant", 0)
                        padded_tensors.append(tensor)
                    collated[key] = torch.stack(padded_tensors)
            return collated

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=int(self.cfg["batch_size"]),
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=custom_collate_fn
        )

        # 5) Stats
        self.global_step = 0
        self.discount = self.cfg["discount"]
        self.lambda_ = self.cfg["lambda_"]
        self.reward_rms = RunningMeanStd(epsilon=1e-5, shape=())
        self.return_rms = RunningMeanStd(epsilon=1e-5, shape=())

        self.last_action = None
        self.inference_state = None

    def reset_inference_state(self, batch_size=1):
        # Falls du z. B. das RNN oder RSSM manuell zurücksetzen willst.
        pass

    def reset(self):
        """
        Wird z. B. vom Trainer aufgerufen, um den Agenten am Episode-Beginn zurückzusetzen.
        """
        self.reset_inference_state()
        self.last_action = None
        self.inference_state = None

    def act(self, obs, sample=True):
        """
        Aktion basierend auf explorativem oder greedy Verhalten wählen.
        Nutzt das World Model und pflegt einen latenten Zustand (self.inference_state).
        """
        # 1) Falls kein interner Zustand vorhanden ist, initialisieren
        if self.inference_state is None:
            # self.inference_state kann z. B. von self.world_model.dynamics.initial(...) kommen
            batch_size = 1
            self.inference_state = self.world_model.dynamics.initial_state(batch_size)

            self.last_action = torch.zeros((batch_size, self.act_dim), device=self.device)

        # 2) Observation in passendes Format
        obs_t = torch.tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        # Erzeuge ein dict so wie im Training
        # "is_first": [0], "is_terminal": [0], "reward": optional ...
        data_dict = {
            "obs": obs_t,  # shape (1, obs_dim)
            "is_first": torch.zeros((1,), device=self.device),
            "is_terminal": torch.zeros((1,), device=self.device),
            # Falls du 'action' fürs pipeline brauchst => evtl. self.last_action
            # "action": self.last_action,
        }

        # 3) Preprocess
        data_dict = self.world_model.preprocess(data_dict)  # normalisiert Keys usw.
        data_dict["is_first"] = (data_dict["is_first"] > 0.5)

        # 4) Encoding
        embed = self.world_model.encoder(data_dict["obs"])  # shape (1, embed_dim)

        # 5) RSSM obs_step
        # Du musst hier den latenten Zustand 'self.inference_state' und die letzte Aktion übergeben,
        # falls du das in Dreamer so verwendest (z. B. "prev_state, prev_action, embed, is_first").
        # Minimal:
        prev_state = self.inference_state  # vorher war prev_latent, aber hier verwenden wir prev_state
        prev_action = (
            torch.zeros((1, self.act_dim), device=self.device)
            if self.last_action is None
            else self.last_action
        )

        prior_state = self.world_model.dynamics.prior_step(prev_state, prev_action, sample=False)
        latent = self.world_model.dynamics.post_step(prior_state, embed, sample=False)
        feat = self.world_model.dynamics.get_feat(latent)

        # 7) Actor (exploration oder greedy)
        actor = self._expl_behavior.actor(feat)
        action = actor.sample() if sample else actor.mode()

        # 8) Speichere latenten Zustand & Aktion
        self.inference_state = latent
        self.last_action = action

        # 9) Gib Aktion zurück
        if not self.is_discrete:
            return action.squeeze(0).detach().cpu().numpy()

        else:
            return action.argmax(dim=-1).item()

    def store_transition(self, transition):
        """
        Speichert eine einzelne Transition in den Replay Buffer.
        """
        # Optionale Priorität
        priority = 1.0
        self.buffer.add(transition, worker=0)
        # Update Reward RMS
        self.reward_rms.update(np.array([transition["reward"]], dtype=np.float32))

    def train(self, num_updates=1):
        """
        Führt num_updates Trainings-Iterationen aus (WorldModel + ImagBehavior + Exploration).
        """
        logs = []
        loader_iter = iter(self.dataloader)
        for _ in range(num_updates):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.dataloader)
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break
            if batch is None:
                break

            # (B,T,...) => world_model._train() erfordert: "obs", "action", "reward", "is_first", "is_terminal" etc.
            data_dict = {
                "obs": batch["obs"].to(self.device),
                "action": batch["action"].to(self.device),
                "reward": batch["reward"].to(self.device),
                "is_terminal": batch["done"].to(self.device),
                "is_first": torch.zeros_like(batch["done"], device=self.device),
            }
            # 1) World Model Update
            post, wm_metrics = self.world_model._train(data_dict)
            print( wm_metrics["dyn_loss"])
            # 2) ImagBehavior
            def objective(feat, state, act):
                return self.world_model.heads["reward"](feat).mean

            post_feat, post_state, post_action, weights, ib_metrics = self.imag_behavior._train(
                post, objective
            )


            # 3) Exploration
            if self.cfg["expl_behavior"] != "greedy":
                context = {"feat": self.world_model.dynamics.get_feat(post)}  # z. B. Feats
                ret_expl = self._expl_behavior.train(post, context, data_dict)
                if ret_expl is not None:
                    _, mets_expl = ret_expl
                    logs.append({f"expl_{k}": v for k, v in mets_expl.items()})

            # Logging
            merged = {}
            for k, v in wm_metrics.items():
                merged[f"wm_{k}"] = v
            for k, v in ib_metrics.items():
                # Nur float-fähige Metriken
                if isinstance(v, (int, float, np.number, torch.Tensor)):
                    merged[f"ib_{k}"] = float(v)
            merged["global_step"] = self.global_step
            logs.append(merged)
            print(logs)

            self.global_step += 1

        return logs

    def state(self):
        """
        Gibt Checkpoint-Daten zurück (Modellgewichte, global_step, etc.).
        """
        return {
            "world_model": self.world_model.state_dict(),
            "imag_behavior": self.imag_behavior.state_dict(),
            "_expl_behavior": (
                self._expl_behavior.state_dict()
                if hasattr(self._expl_behavior, "state_dict")
                else None
            ),
            "global_step": self.global_step
        }

    def restore_state(self, ckpt):
        """
        Stellt Modellgewichte etc. aus ckpt wieder her.
        """
        self.world_model.load_state_dict(ckpt["world_model"])
        self.imag_behavior.load_state_dict(ckpt["imag_behavior"])
        if ckpt.get("_expl_behavior") and hasattr(self._expl_behavior, "load_state_dict"):
            self._expl_behavior.load_state_dict(ckpt["_expl_behavior"])
        self.global_step = ckpt["global_step"]
