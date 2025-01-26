import yaml
from pathlib import Path
from models.ddpg.DDPGTrainer import DDPGTrainer
from models.dreamer.DREAMTrainer import DreamerV3Trainer
import argparse


def main():
    parser = argparse.ArgumentParser(description='Train an agent for a specified experiment')

    
    parser.add_argument('--experiment-id',
                        type=str,
                        help='experiment id',
                        dest='experiment_id')
    
    args = parser.parse_args()
    experiment_id = args.experiment_id
    experiment_path = Path(f"rl_experiments/experiments/{experiment_id}")
    
    with open(f"{experiment_path}/configs/training_config.yaml", "r") as f:
        training_config = yaml.safe_load(f)

    env_name = training_config['environment']['name']
    training_parameters = training_config['training']
    model_config = training_config['model']['config']
    

    trainer_map = {
        #"DDPG": DDPGTrainer,
        # "PPO": PPOTrainer,
        # "DQN": DQNTrainer,
        "Dreamer":DreamerV3Trainer
    }

    trainer = trainer_map[training_config['model']['name']](
        env_name=env_name,
        training_config=training_parameters,
        model_config=model_config,
        experiment_path=experiment_path
    )
    
    final_results = trainer.train()
    print("Training complete. Final metrics:", final_results)

if __name__ == "__main__":
    main()
