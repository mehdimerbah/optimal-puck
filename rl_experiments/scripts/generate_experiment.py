#!/usr/bin/env python3

import os
import yaml
import json
import logging
from datetime import datetime
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentGenerator:
    def __init__(self, master_config_path: str):
        """Initialize experiment generator with master config file."""
        with open(master_config_path, 'r') as f:
            self.master_config = yaml.safe_load(f)

        # Create top-level structure if it doesn't exist
        self.top_level_dir = Path("rl_experiments")
        self.configs_dir = self.top_level_dir / "configs"
        self.experiments_dir = self.top_level_dir / "experiments"
        self.scripts_dir = self.top_level_dir / "scripts"

        self._ensure_top_level_structure()

    def _ensure_top_level_structure(self):
        """Ensure the top-level directories (configs, experiments, scripts) exist."""
        for directory in [self.top_level_dir, self.configs_dir, self.experiments_dir, self.scripts_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Optionally create an empty README and requirements.txt if they don't exist
        readme_path = self.top_level_dir / "README.md"
        if not readme_path.exists():
            readme_path.write_text("# RL Experiments\n\nDocumentation goes here.\n")

        req_path = self.top_level_dir / "requirements.txt"
        if not req_path.exists():
            req_path.write_text("# Add your Python dependencies here.\n")

    def _generate_experiment_id(self) -> str:
        """
        Generate unique experiment ID based on environment name, agent type, and timestamp.
        By default, we’ll assume a single agent type is DQN (change as needed).
        """
        env_name = self.master_config['environment']['name'].replace(" ", "_")
    
        agent_type = self.master_config.get('model', {}).get('name')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{env_name}_{agent_type}_{timestamp}"

    def _create_experiment_structure(self, experiment_path: Path):
        """
        Create the detailed structure within a single experiment directory:
        - configs/
          - training_config.yaml
          - evaluation_config.yaml
        - results/
          - training/(saved_models, stats, plots, logs)
          - evaluation/(stats, plots, logs, gifs)
        - meta_data_{experiment_timestamp}.json
        """
        # Create subdirectories
        subdirs = [
            "configs",
            "results/training/saved_models",
            "results/training/stats",
            "results/training/plots",
            "results/training/logs",
            "results/evaluation/stats",
            "results/evaluation/plots",
            "results/evaluation/logs",
            "results/evaluation/gifs",
        ]
        for sdir in subdirs:
            (experiment_path / sdir).mkdir(parents=True, exist_ok=True)

    def _generate_training_config(self) -> dict:
        """
        Generate training configuration for *all* models specified in the master config.
        Each model in self.master_config['models'] will have its 'training' dict included.
        """
        # Construct the overall training config
        return {
            'environment': self.master_config['environment'],
            'training': self.master_config['training'],
            'model': self.master_config['model'],
        }


    def _generate_evaluation_config(self) -> dict:
        """Generate evaluation configuration."""
        return {
            'evaluation': self.master_config['evaluation'],
        }


    def generate_experiment(self) -> str:
        """
        Generate the entire experiment structure:
         1. Create an experiment folder under rl_experiments/experiments
         2. Create subdirectories for configs, results/training, results/evaluation
         3. Save training_config.yaml and evaluation_config.yaml
         4. Save meta_data_{timestamp}.json
         5. Optionally, save a copy of the config at top-level configs/
        """
        experiment_id = self._generate_experiment_id()
        experiment_path = self.experiments_dir / experiment_id

        logger.info(f"Generating experiment: {experiment_id}")
        self._create_experiment_structure(experiment_path)

        # Generate config files for training and evaluation
        training_config = self._generate_training_config()
        evaluation_config = self._generate_evaluation_config()

        # Save them in {experiment_path}/configs/
        with open(experiment_path / 'configs' / 'training_config.yaml', 'w') as f:
            yaml.dump(training_config, f, default_flow_style=False)
        with open(experiment_path / 'configs' / 'evaluation_config.yaml', 'w') as f:
            yaml.dump(evaluation_config, f, default_flow_style=False)

        # Create metadata file
        metadata = {
            'experiment_id': experiment_id,
            'created_at': datetime.now().isoformat(),
            'environment': self.master_config['environment']['name'],
            'agent_type': 'dqn',  # or read from config if multiple
            'status': 'initialized'
        }
        meta_file = experiment_path / f"meta_data_{experiment_id}.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=4)

        # save a top-level config for reference
        # self._save_agent_config_at_top_level('dqn')

        logger.info(f"Experiment structure created at {experiment_path}")
        return experiment_id


def main():
    parser = argparse.ArgumentParser(description='Generate experiment structure')
    parser.add_argument('--config',
                        type=str,
                        default='configs/master_config.yaml',
                        help='Path to master configuration file')
    args = parser.parse_args()

    try:
        generator = ExperimentGenerator(args.config)
        exp_id = generator.generate_experiment()
        logger.info(f"Experiment generated successfully: {exp_id}")
    except Exception as e:
        logger.error(f"Error generating experiment: {e}")
        raise

if __name__ == '__main__':
    main()
