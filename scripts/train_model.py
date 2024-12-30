import yaml
import argparse
import json
from pathlib import Path
from datetime import datetime
#from models.dqn.DQNTrainer import DQNTrainer
#from models.ddpg.DDPGTrainer import DDPGTrainer
from models.td3.TD3Trainer import TD3Trainer
# from models.ppo import PPOTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train online RL models')
    parser.add_argument('--experiment-id', type=str, required=True,
                        help='Experiment ID (e.g., lunarlander_online_20241212_1600)')
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['dqn', 'ddpg', 'td3', 'ppo'],
                        help='Type of model to train')
    return parser.parse_args()

def load_config(config_path):
    """Load experiment configurations from YAML files."""
    config_file = config_path / 'configs' / 'training_config.yaml'
    with open(config_file, 'r') as f:
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

def main():
    args = parse_args()
    
    # Setup paths
    experiment_path = Path('experiments') / args.experiment_id
    metadata_path = experiment_path / f'meta_data_{args.experiment_id}.json'
    
    # Load configurations
    config = load_config(experiment_path)
    env_config = config['environment']
    model_config = config['model']
    training_config = config['training']
    
    # Update the trainer mapping to include all imported models
    trainer_map = {
        #'dqn': DQNTrainer,
        #'ddpg': DDPGTrainer,
        'td3': TD3Trainer,
        # 'ppo': PPOTrainer  # Uncomment when PPO is implemented
    }
    
    # Validate model type
    if args.model_type not in trainer_map:
        raise ValueError(f"Model type '{args.model_type}' not implemented. Available models: {list(trainer_map.keys())}")
    
    # Validate that the config matches the requested model type
    if model_config['type'] != args.model_type:
        raise ValueError(f"Config file is for model type '{model_config['type']}' but requested model type is '{args.model_type}'")
    
    trainer_class = trainer_map[args.model_type]
    trainer = trainer_class(
        env_name=env_config['name'],
        training_config=training_config,
        model_config=model_config,
        experiment_path=experiment_path
    )
    
    # Train the model
    print(f"Starting training for {args.model_type}...")
    metrics = trainer.train()
    
    # Update and save metadata
    save_training_results(metadata_path, args.model_type, metrics)
    
    print(f"\nTraining completed for {args.model_type}!")
    print(f"Final metrics: {metrics}")

if __name__ == "__main__":
    main()
