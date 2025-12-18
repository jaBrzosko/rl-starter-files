# RL Starter Files with Knowledge Graph Injection

This repository is a fork of the original [`rl-starter-files`](https://github.com/lcswillems/rl-starter-files), which provides a framework to train, visualize, and evaluate reinforcement learning (RL) agents with minimal setup. It extends the original functionality by introducing **Knowledge Graph (KG) Injection** to enhance RL agent training and decision-making.

<p align="center">
    <img width="300" src="README-rsrc/visualize-keycorridor.gif">
</p>

## Key Enhancements

### Knowledge Graph Integration
- **Knowledge Graph Injection**: The RL agents can now leverage structured knowledge from external sources to improve learning efficiency and decision-making.
- **KG Utilities**: Scripts and modules for managing, uploading, and querying knowledge graphs are included in the [`kg`](kg/) directory. Key files:
  - [`masker.py`](kg/masker.py): Implements masking strategies for KG data.
  - [`oxigraph.py`](kg/recommendations/oxigraphKGRecommender.py): Best KG injection strategy using PyOxiGraph library.
  - [`data`](kg/data/): Contains sample KG datasets and utilities for processing them.

### New Scripts and Logic
- **Training Enhancements**: The [`train.py`](scripts/train.py) script now supports injecting KG-based features during training.
- **Evaluation Updates**: The [`evaluate.py`](scripts/evaluate.py) script includes logic to assess the impact of KG on agent performance.
- **Visualization Improvements**: The [`visualize.py`](scripts/visualize.py) script can now display KG-influenced decision paths.

### Tests and Debugging
- **Test Coverage**: Unit tests for KG-related functionality are included in the [`test`](test/) directory.
- **Debugging Tools**: Files like [`debug_dump.nq`](kg+old/debug_dump.nq) and [`test.ipynb`](kg+old/test.ipynb) provide utilities for debugging KG data and logic.

### Compatibility
- Fully compatible with [`torch-ac`](https://github.com/lcswillems/torch-ac) for RL algorithms like A2C and PPO.
- Supports environments from [`minigrid`](https://github.com/Farama-Foundation/Minigrid).

## Features Retained from the Original Repository
- **Training Scripts**: Log training progress in TXT, CSV, and Tensorboard formats. Save and reload models seamlessly.
- **Visualization Tools**: Visualize agent behavior with options to save outputs as GIFs.
- **Algorithm Support**: Use A2C or PPO algorithms for training.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rl-starter-files-kg.git
   cd rl-starter-files-kg
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run experiments with KG injection:
   ```bash
   python -m scripts.experiments --experiment <experiment_name>
   ```

This will execute the specified experiment with knowledge graph enhancements. List of available experiments can be found in the `scripts/data/experiment_store.py` file. Some examples include:
  - `lava_gap_kg_complex`
  - `empty_kg_base`
  - `lava_gap_kg_base_episode_cutoff_ec60`
  - `door_key_kg_complex`
and much more

  ## Repository Structure

  - **`kg`**: Knowledge graph utilities and data processing scripts.
  - **`scripts`**: Training, evaluation, and visualization scripts.
  - **`test`**: Unit tests for RL and KG functionalities.
  - **`torch-ac`**: Submodule for RL algorithms.

  ## Acknowledgments

  This project builds upon the work in the original [`rl-starter-files`](https://github.com/lcswillems/rl-starter-files) and its submodule [`torch-ac`](https://github.com/lcswillems/torch-ac). Thanks to the original authors for their contributions this KG-enhanced version was more easily developed.
