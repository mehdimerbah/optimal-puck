We show below the directory structure of our experimental setup.


```

rl_experiments/
├── configs/
│   ├── {env_name}_{agent_type}_config.yaml
│   ├── {env_name}_{agent_type}_config.yaml
│   └── {env_name}_{agent_type}_config.yaml
├── experiments/
│   ├── {env_name}_{agent_type}_{timestamp}/
│   │   ├── configs/
│   │   │   ├── training_config.yaml
│   │   │   └── evaluation_config.yaml
│   │   ├── results/
|   |   |   ├── training/
|   |   |   |   ├── saved_models/
|   |   |   |   ├── stats/
|   |   |   |   ├── plots/
|   |   |   |   └── logs/
│   │   │   └── evaluation/
|   |   |       ├── stats/
|   |   |       ├── plots/
|   |   |       ├── logs/
|   |   |       └── gifs/    
│   │   │   
│   │   └── meta_data_{experiment_timestamp}.json
│   └── ...
├── scripts/
│   ├── generate_experiment.py
│   ├── train_agent.py
│   ├── evaluate_agent.py
│   └── hyperparameter_sweep.py
│   
├── requirements.txt
└── README.md

```
