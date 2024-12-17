We show below the directory structure of our experimental setup.


```
rl_experiments/
├── configs/
│   ├── {env_name}_{agent_type}_config.yaml
│   ├── {env_name}_{agent_type}_config.yaml
│   └── {env_name}_{agent_type}_config.yaml
├── hyperparameter_registry/
│   ├── dqn/
│   │   └── hash_registry.json
│   ├── ppo/
│   │   └── hash_registry.json
│   └── sac/
│       └── hash_registry.json
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── dqn.py
│   │   ├── ppo.py
│   │   └── sac.py
│   ├── environments/
│   │   ├── __init__.py
│   │   └── lunar_lander.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── logging.py
├── experiments/
│   ├── {env_name}_{agent_type}_{timestamp}/
│   │   ├── configs/
│   │   │   ├── training_config.yaml
│   │   │   ├── evaluation_config.yaml
│   │   ├── models/
│   │   │   ├── {model_name}_{hyperparams_hash}_{timestamp}/
│   │   │   └── evaluation/
│   │   │       ├── results.json
│   │   │       ├── trajectories.json
│   │   │       └── logs/
│   │   ├── results/
│   │   │   ├── metrics/
│   │   │   │   └── {metric_type}_{agent_type}_{timestamp}.json
│   │   │   ├── plots/
│   │   │   │   └── {metric_type}_{agent_type}_{timestamp}.png
│   │   │   └── videos/
│   │   ├── logs/
│   │   └── meta_data_{experiment_timestamp}.json
│   └── ...
├── scripts/
│   ├── generate_experiment.py
│   ├── train_agents.py
│   ├── evaluate_agents.py
│   ├── hyperparameter_sweep.py
│   └── aggregate_results.py
├── requirements.txt
└── README.md

```
