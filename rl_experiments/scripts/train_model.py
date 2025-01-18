import yaml
import argparse
import json
from pathlib import Path
from datetime import datetime
<<<<<<< Updated upstream:rl_experiments/scripts/train_model.py
from models.dqn import DQNTrainer
from models.ddpg import DDPGTrainer
=======
import logging
#from models.dqn.DQNTrainer import DQNTrainer
from models.ddpg.DDPGTrainer import DDPGTrainer
from models.td3.TD3Trainer import TD3Trainer
>>>>>>> Stashed changes:scripts/train_model.py
# from models.ppo import PPOTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Train online RL models')
    parser.add_argument('--experiment-id', type=str, required=True,
                        help='Experiment ID (e.g., lunarlander_online_20241212_1600)')
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['dqn', 'ppo', 'sac', 'ddpg'],
                        help='Type of model to train')
    return parser.parse_args()

def load_config(config_path):
    """Load experiment configurations from YAML files."""
    with open(config_path / 'config.yaml', 'r') as f:
        return yaml.safe_load(f)

def save_training_results(metadata_path, model_type, metrics):
    """Update and save the training results in the experiment metadata."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if 'training_results' not in metadata:
        metadata['training_results'] = {}

    metadata['training_results'][model_type] = {
        'training_completed': datetime.now().isoformat(),
        'final_metrics': metrics
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

<<<<<<< Updated upstream:rl_experiments/scripts/train_model.py
def main():
    args = parse_args()
    
    # Setup paths
    experiment_path = Path('experiments') / args.experiment_id
    config_path = experiment_path / 'config.yaml'
    metadata_path = experiment_path / f'meta_data_{args.experiment_id}.json'
    
    # Load configurations
    config = load_config(Path(experiment_path))
    env_config = config['environment']
    training_config = config['models'][args.model_type]['training']
    eval_config = config['evaluation']
    resources = config.get('resources', {})
    
    # Select the appropriate trainer
    trainer_map = {
        'dqn': DQNTrainer,
        #'ppo': PPOTrainer
    }
    
    trainer_class = trainer_map[args.model_type]
    trainer = trainer_class(
        env_name=env_config['name'],
        training_config=training_config,
        eval_config=eval_config,
        resources=resources,
=======
def setup_paths(experiment_id):
    """Setup and return experiment and metadata paths."""
    experiment_path = Path('experiments') / experiment_id
    metadata_path = experiment_path / f'meta_data_{experiment_id}.json'
    return experiment_path, metadata_path

def initialize_trainer(args, config, experiment_path):
    """Initialize and return the trainer based on the model type."""
    trainer_map = {
        # 'dqn': DQNTrainer,
        'ddpg': DDPGTrainer,
        'td3': TD3Trainer,
        # 'ppo': PPOTrainer  # Uncomment when PPO is implemented
    }
    
    if args.model_type not in trainer_map:
        raise ValueError(f"Model type '{args.model_type}' not implemented. Available models: {list(trainer_map.keys())}")
    
    model_config = config['model']
    if model_config['type'] != args.model_type:
        raise ValueError(f"Config file is for model type '{model_config['type']}' but requested model type is '{args.model_type}'")
    
    trainer_class = trainer_map[args.model_type]
    return trainer_class(
        env_name=config['environment']['name'],
        training_config=config['training'],
        model_config=model_config,
>>>>>>> Stashed changes:scripts/train_model.py
        experiment_path=experiment_path
    )

def execute_training(trainer, model_type):
    """Execute the training process and return the metrics."""
    logging.info(f"Starting training for {model_type}...")
    metrics = trainer.train()
    logging.info(f"\nTraining completed for {model_type}!")
    logging.info(f"Final metrics: {metrics}")
    return metrics

def main():
    args = parse_args()
    experiment_path, metadata_path = setup_paths(args.experiment_id)
    config = load_config(experiment_path)
    trainer = initialize_trainer(args, config, experiment_path)
    metrics = execute_training(trainer, args.model_type)
    save_training_results(metadata_path, args.model_type, metrics)

if __name__ == "__main__":
    main()
