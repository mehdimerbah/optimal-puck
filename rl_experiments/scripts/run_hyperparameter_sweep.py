import yaml
import os
from pathlib import Path
from models.ddpg.DDPGTrainer import DDPGTrainer
import argparse
import wandb
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRAINER_MAP = {
    "DDPG": DDPGTrainer,
}
EXPERIMENT_ID = "Pendulum-v1_DDPG_20250112_200510"
EXPERIMENT_PATH = Path("/Users/mehdi/Documents/Tübingen/WiSe24/ReinforcementLearning/Project/optimal-puck/rl_experiments/experiments/Pendulum-v1_DDPG_20250112_200510")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an agent for a specified experiment")
    parser.add_argument("--experiment-id", type=str, required=True, help="Experiment ID")
    return parser.parse_args()

def get_trainer(name):
    try:
        return TRAINER_MAP[name]
    except KeyError:
        raise ValueError(f"Unknown model name '{name}'. Available trainers: {list(TRAINER_MAP.keys())}")
    
def get_hyperparameters(config, model_name):
    hyperparameter_map = {}
    if model_name == "DDPG":
        hyperparameter_map = {
            'learning_rate_actor': config.learning_rate_actor,
            'learning_rate_critic': config.learning_rate_critic,
            'batch_size': config.batch_size,
            'discount': config.discount,
            'eps': config.eps,
            'buffer_size': config.buffer_size
        }

    return hyperparameter_map

def train_agent(config=None):
    with wandb.init(config=config):
        config = wandb.config

        with open(f"{EXPERIMENT_PATH}/configs/training_config.yaml", "r") as f:
            training_config = yaml.safe_load(f)
        
        trainer_cls = get_trainer(training_config["model"]["name"])
        hyperparams = get_hyperparameters(config, training_config["model"]["name"])

        trainer = trainer_cls(
            env_name=training_config["environment"]["name"],
            training_config=training_config["training"],
            model_config=hyperparams,
            experiment_path=EXPERIMENT_PATH,
            wandb_run=wandb
        )
        trainer.train()

def main():
    args = parse_args()
    experiment_path = Path(f"rl_experiments/experiments/{args.experiment_id}")

    try:
        with open(f"{experiment_path}/configs/training_config.yaml", "r") as f:
            training_config = yaml.safe_load(f)
            wandb_config = training_config["model"]["wandb_sweep_config"]
            
    except Exception as e:
        logger.error(f"Failed to load sweep configuration: {e}")
        return

    sweep_id = wandb.sweep(sweep=wandb_config, project="optimal-tuning")
    wandb.agent(sweep_id, function=train_agent)

if __name__ == "__main__":
    main()
