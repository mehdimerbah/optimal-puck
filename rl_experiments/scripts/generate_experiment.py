import os
import yaml
import json
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentGenerator:
    def __init__(self, master_config_path: str):
        """Initialize experiment generator with master config file."""
        with open(master_config_path, 'r') as f:
            self.master_config = yaml.safe_load(f)

    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID based on environment, agent type, and timestamp."""
        env_name = self.master_config['environment']['name'].lower()
        agent_type = 'DQN'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        return f"{env_name}_{agent_type}_{timestamp}"

    def _create_directory_structure(self, experiment_path: Path):
        """Create experiment directory structure."""
        directories = [
            'configs',
            'logs',
            'results/metrics',
            'results/plots',
            'results/gifs',
        ]
        for model in self.master_config:
            directories.append(model)
            directories.append(f'{model}/evaluation')
            directories.append(f'{model}/evaluation/logs')

        for dir_path in directories:
            (experiment_path / dir_path).mkdir(parents=True, exist_ok=True)

    def _generate_training_config(self) -> dict:
        """Generate training configuration."""
        return {
            'environment': self.master_config['environment'],
            'model': self.master_config['models']['dqn'],
            'training': self.master_config['training']
        }

    def _generate_evaluation_config(self) -> dict:
        """Generate evaluation configuration."""
        return {
            'n_episodes': self.master_config['evaluation']['n_episodes'],
            'max_steps_per_episode': self.master_config['evaluation']['max_steps_per_episode'],
            'metrics': self.master_config['evaluation']['metrics']
        }

    def generate_experiment(self) -> str:
        """Generate complete experiment structure and configurations."""
        experiment_id = self._generate_experiment_id()
        experiment_path = Path('experiments') / experiment_id

        logger.info(f"Generating experiment: {experiment_id}")

        self._create_directory_structure(experiment_path)

        # Generate and save configurations
        configs = {
            'training_config.yaml': self._generate_training_config(),
            'evaluation_config.yaml': self._generate_evaluation_config()
        }
        config_path = experiment_path / 'configs'
        for filename, config in configs.items():
            with open(config_path / filename, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

        # Create metadata file
        metadata = {
            'experiment_id': experiment_id,
            'created_at': datetime.now().isoformat(),
            'environment': self.master_config['environment']['name'],
            #'agent_type': self.master_config['agent']['type'],
            'model_type': 'DQN',
            'status': 'initialized'
        }
        with open(experiment_path / f'meta_data_{experiment_id}.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Experiment structure created at {experiment_path}")
        return experiment_id

def main():
    """Main function to run experiment generation."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate experiment structure')
    parser.add_argument('--config', type=str, default='configs/master_config.yaml',
                        help='Path to master configuration file')
    args = parser.parse_args()

    try:
        generator = ExperimentGenerator(args.config)
        experiment_id = generator.generate_experiment()
        logger.info(f"Experiment generated successfully: {experiment_id}")
    except Exception as e:
        logger.error(f"Error generating experiment: {str(e)}")
        raise

if __name__ == '__main__':
    main()
