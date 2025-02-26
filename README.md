# optimal-puck

**optimal-puck** is a research repository for running reinforcement learning experiments on the LaserHockey environment for the Reinforcement Learning course by Dr. [Georg martius](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/distributed-intelligence/team/prof-dr-georg-martius/). It implements and evaluates the DDPG and TD3  algorithms supports both standard and self-play training setups. The repo also provides evaluation tools, and a framework for hyperparameter sweeps and curriculum learning.

## Features

- **Multi-Algorithm Support:** Implementations for DDPG, TD3, DQN, and Dreamer.
- **Experimental Setup:** Predefined configurations and scripts for generating and running reproducible RL experiments.
- **Evaluation & Logging:** Scripts to evaluate agent performance against various opponents with detailed logging and plotting.
- **Hyperparameter Sweeps:** Tools for running systematic sweeps via Weights & Biases.
- **Curriculum & Self-Play:** Integrated support for opponent pooling and curriculum adjustments during training for each algorithm.

## Repository Structure

```
mehdimerbah-optimal-puck/
├── README.md                  # Overview and instructions.
├── ExperimentalSetup.md       # Detailed explanation of the experimental directory layout.
├── LICENSE                    # MIT License.
├── requirements.txt           # Python dependencies for the main repo.
├── container/                 # Container-specific files:
│   ├── OptimalPuck_16-01-2025_TD3_LunarLander.sbatch  # Example SLURM job script.
│   ├── requirements.txt       # Container dependencies.
│   └── singularity.def        # Singularity definition file.
├── evaluation/                # Evaluation scripts and checkpoints for DDPG and TD3.
│   ├── DDPG/
│   └── TD3/
├── models/                    # Source code for RL models and baselines.
│   ├── baseline/              # Helper modules (MLP, memory, prioritized memory).
│   ├── ddpg/                  # DDPG agent and trainer.
│   ├── dqn/                   # DQN agent and trainer.
│   ├── dreamer/               # Dreamer agent and trainer.
│   └── td3/                   # TD3 agent and trainer.
└── rl_experiments/            # Configurations and scripts for running experiments.
    ├── configs/               # YAML config files for different agents/experiments.
    └── scripts/               # Python scripts for generating experiments, training, and hyperparameter sweeps.
```

For more detailed instructions on the experimental setup and directory organization, please refer to [ExperimentalSetup.md](./ExperimentalSetup.md).

## Experimental Setup

The `rl_experiments` folder contains:
- **configs:** YAML files for setting up different environments, training regimes, and evaluation protocols.
- **scripts:** Utilities to generate experiments, initiate training, and conduct hyperparameter sweeps.

### Generating and Experiment
  To generate an experiment, run:
    ```bash
  python rl_experiments/scripts/generate_experiment.py --config rl_experiments/configs/<CONFIG_FILE>
  ```
### Running Training

- **Standard Training:**
  
  Use the scripts in `rl_experiments/scripts/` to start training. For example, to train an agent:
  
  ```bash
  python rl_experiments/scripts/train_agent.py --experiment-id <EXPERIMENT_ID>
  ```

- **Hyperparameter Sweeps:**

  To run a hyperparameter sweep using Weights & Biases, use:

  ```bash
  python rl_experiments/scripts/run_hyperparameter_sweep.py --experiment-id <EXPERIMENT_ID>
  ```

  Make sure to update the configuration files in `rl_experiments/configs/` as needed.

### Evaluation

Evaluation scripts are provided in the `evaluation/` directory. They allow you to:
- Evaluate checkpoints for the experiments
- Generate plots and statistics to compare agent performance against various opponent types.


## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

## Acknowledgments

The repository includes code and ideas inspired by research papers on RL (e.g., DDPG, TD3, Dreamer) as well as contributions from Mehdi Merbah, Thomas Vogel, and Michael Maier.
