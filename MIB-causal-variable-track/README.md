# MIB Causal Variable Localization Track

This repository contains the implementation of the Causal Variable Localization Track of the **M**echanistic **I**nterpretability **B**enchmark (**MIB**). This track benchmarks featurization methods—i.e., methods for transforming model activations into a space where it's easier to isolate and intervene on specific causal variables.

<p align="center">
    <a href="https://huggingface.co/collections/mib-bench/mib-datasets-67f55273612ec3067a42a56b"><img height="20px" src="https://img.shields.io/badge/data-purple.svg?logo=huggingface"></a>
    <a href="https://huggingface.co/spaces/mib-bench/leaderboard"><img height="20px" src="https://img.shields.io/badge/leaderboard-red.svg?logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48IS0tIFVwbG9hZGVkIHRvOiBTVkcgUmVwbywgd3d3LnN2Z3JlcG8uY29tLCBHZW5lcmF0b3I6IFNWRyBSZXBvIE1peGVyIFRvb2xzIC0tPgo8c3ZnIHdpZHRoPSI4MDBweCIgaGVpZ2h0PSI4MDBweCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNMTUgMTlIOVYxMi41VjguNkM5IDguMjY4NjMgOS4yNjg2MyA4IDkuNiA4SDE0LjRDMTQuNzMxNCA4IDE1IDguMjY4NjMgMTUgOC42VjE0LjVWMTlaIiBzdHJva2U9IiNmNWY1ZjUiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPHBhdGggZD0iTTE1IDVIOSIgc3Ryb2tlPSIjZjVmNWY1IiBzdHJva2Utd2lkdGg9IjEuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+CjxwYXRoIGQ9Ik0yMC40IDE5SDE1VjE1LjFDMTUgMTQuNzY4NiAxNS4yNjg2IDE0LjUgMTUuNiAxNC41SDIwLjRDMjAuNzMxNCAxNC41IDIxIDE0Ljc2ODYgMjEgMTUuMVYxOC40QzIxIDE4LjczMTQgMjAuNzMxNCAxOSAyMC40IDE5WiIgc3Ryb2tlPSIjZjVmNWY1IiBzdHJva2Utd2lkdGg9IjEuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+CjxwYXRoIGQ9Ik05IDE5VjEzLjFDOSAxMi43Njg2IDguNzMxMzcgMTIuNSA4LjQgMTIuNUgzLjZDMy4yNjg2MyAxMi41IDMgMTIuNzY4NiAzIDEzLjFWMTguNEMzIDE4LjczMTQgMy4yNjg2MyAxOSAzLjYgMTlIOVoiIHN0cm9rZT0iI2Y1ZjVmNSIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4="></a>
    <a href="https://arxiv.org/abs/2504.13151"><img height="20px" src="https://img.shields.io/badge/paper-brown.svg?logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBVcGxvYWRlZCB0bzogU1ZHIFJlcG8sIHd3dy5zdmdyZXBvLmNvbSwgR2VuZXJhdG9yOiBTVkcgUmVwbyBNaXhlciBUb29scyAtLT4KPCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KPHN2ZyBmaWxsPSIjZmZmZmZmIiB2ZXJzaW9uPSIxLjEiIGlkPSJFYmVuZV8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiAKCSB3aWR0aD0iODAwcHgiIGhlaWdodD0iODAwcHgiIHZpZXdCb3g9IjAgMCA2NCA2NCIgZW5hYmxlLWJhY2tncm91bmQ9Im5ldyAwIDAgNjQgNjQiIHhtbDpzcGFjZT0icHJlc2VydmUiPgo8Zz4KCTxwYXRoIGQ9Ik01Miw2NGMyLjIwNiwwLDQtMS43OTQsNC00VjRjMC0yLjIwNi0xLjc5NC00LTQtNEgyNmMtMS42MzgsMC0zLjY2OCwwLjg0MS00LjgyOCwyTDEwLDEzLjE3MUM4Ljg0MSwxNC4zMyw4LDE2LjM2LDgsMTh2NDIKCQljMCwyLjIwNiwxLjc5NCw0LDQsNEg1MnogTTUyLDRsMCw1NmgwSDEyVjE4aDEwYzIuMjA2LDAsNC0xLjc5NCw0LTRWNEg1MnogTTIyLDYuODI4VjE0aC03LjE3MkwyMiw2LjgyOHoiLz4KCTxwYXRoIGQ9Ik0zMiwxNmgxMmMxLjEwNCwwLDItMC44OTYsMi0ycy0wLjg5Ni0yLTItMkgzMmMtMS4xMDQsMC0yLDAuODk2LTIsMlMzMC44OTYsMTYsMzIsMTZ6Ii8+Cgk8cGF0aCBkPSJNNDQsMjJIMjBjLTEuMTA0LDAtMiwwLjg5Ni0yLDJzMC44OTYsMiwyLDJoMjRjMS4xMDQsMCwyLTAuODk2LDItMlM0NS4xMDQsMjIsNDQsMjJ6Ii8+Cgk8cGF0aCBkPSJNNDQsMzJIMjBjLTEuMTA0LDAtMiwwLjg5Ni0yLDJzMC44OTYsMiwyLDJoMjRjMS4xMDQsMCwyLTAuODk2LDItMlM0NS4xMDQsMzIsNDQsMzJ6Ii8+Cgk8cGF0aCBkPSJNNDQsNDJIMjBjLTEuMTA0LDAtMiwwLjg5Ni0yLDJzMC44OTYsMiwyLDJoMjRjMS4xMDQsMCwyLTAuODk2LDItMlM0NS4xMDQsNDIsNDQsNDJ6Ii8+Cgk8cGF0aCBkPSJNNDQsNTJIMjBjLTEuMTA0LDAtMiwwLjg5Ni0yLDJzMC44OTYsMiwyLDJoMjRjMS4xMDQsMCwyLTAuODk2LDItMlM0NS4xMDQsNTIsNDQsNTJ6Ii8+CjwvZz4KPC9zdmc+"></a>
    <a href="https://mib-bench.github.io"><img height="20px" src="https://img.shields.io/badge/website-black.svg" alt="website"></a>
</p>

## Leaderboard Submission Overview

The Causal Variable Localization Track evaluates methods that can identify and manipulate specific causal variables within language model (LM) representations. A **high-level causal model** serves as a hypothesis about how the **low-level LM** solves the task. A **featurizer** is an invertible function that maps from the activation space of a hidden vector to a new space and we refer to the dimensions of this new space as **features**. A submission to this track will align a variable from the causal model with features of a hidden vector in the LM. As such a submission will consist of a **token position** that identify residual stream or attention head hidden vectors in an LM and a **featurizer** that maps from that hidden vectors activation space into a new feature space. **The submission can be for any number of layers, but only submissions that provide a token position and featurizer for every layer will be added to the leaderboard for average performance, only best performance.**

See the Jupyter notebook [ioi_example_submission.ipynb](https://github.com/aaronmueller/MIB/blob/main/MIB-causal-variable-track/ioi_example_submission.ipynb) for an example of how to get the trained featurizer and token indices files for the ioi task and the Jupyter notebook [ioi_example_submission.ipynb](https://github.com/aaronmueller/MIB/blob/main/MIB-causal-variable-track/example_submission.ipynb) for an example of how to get the trained featurizer and token indices files for the rest of the tasks. These notebooks will save models in mock_submission folders. Currently the [mock_submission](https://github.com/aaronmueller/MIB/blob/main/MIB-causal-variable-track/) contains a mock featurizer file and mock token position file. **If not feauturizer or token position is provided in the submission, then the featurizers and token positions from the original baselines can still be used.**

The resulting submission must be formatted in a folder structure shown below and then submitted to the leaderboard at [this link](https://huggingface.co/spaces/mib-bench/leaderboard). Run the verification script [verify_submission.py](https://github.com/aaronmueller/MIB/blob/main/MIB-causal-variable-track/verify_submission.py) to ensure that custom token positions and featurizers are loaded properly. When we receive your submission, we will run the evaluation scripts [evalute_submission.py](https://github.com/aaronmueller/MIB/blob/main/MIB-causal-variable-track/evaluate_submission.py) and [ioi_evaluate_submission.py](https://github.com/aaronmueller/MIB/blob/main/MIB-causal-variable-track/ioi_evaluation_submission.py) on the private test sets and report the results on the leaderboard.

## Repository Structure

```
MIB-causal-variable-track/
├── tasks/                      # Task definitions and datasets
│   ├── IOI_task/              # Indirect Object Identification task
│   ├── simple_MCQA/           # Simple Multiple Choice QA task
│   ├── ARC/                   # ARC (AI2 Reasoning Challenge) task
│   ├── two_digit_addition_task/  # Arithmetic task
│   └── RAVEL/                 # RAVEL knowledge editing task
├── CausalAbstraction/         # Core functionality submodule
│   ├── causal/                # Causal models and datasets
│   ├── neural/                # Neural network units and featurizers
│   └── experiments/           # Intervention experiments
├── baselines/                 # Baseline method implementations
├── example_submission.ipynb   # Example submission for standard tasks
├── ioi_example_submission.ipynb  # Example submission for IOI task
├── evaluate_submission.py     # Evaluation script for standard tasks
├── ioi_evaluate_submission.py # Evaluation script for IOI task
├── aggregate_results.py       # Results aggregation script
├── process_all_submissions.py # Batch submission processing
└── verify_submission.py       # Submission format verification
```

## Getting Started

### Installation

1. Clone this repository:
```bash
git clone https://github.com/your-repo/MIB-causal-variable-track.git
cd MIB-causal-variable-track
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the CausalAbstraction submodule:
```bash
git submodule update --init --recursive
```

**sae_lens requires python 3.12**

### Running Example Submissions

We provide two example notebooks that demonstrate how to create submissions:

1. **Standard Tasks Example** (`example_submission.ipynb`): Shows how to train and submit featurizers for tasks like Simple MCQA, ARC, Arithmetic, and RAVEL
2. **IOI Task Example** (`ioi_example_submission.ipynb`): Demonstrates the specific workflow for the IOI (Indirect Object Identification) task which involves attention heads

## Available Tasks

The benchmark includes five tasks:

- **IOI (Indirect Object Identification)**: Tests the model's ability to track entity relationships and predict the correct indirect object
- **Simple MCQA**: Multiple choice question answering with object-color associations
- **ARC**: AI2 Reasoning Challenge questions testing scientific reasoning
- **Arithmetic**: Two-digit addition problems testing numerical computation
- **RAVEL**: Knowledge editing task for factual associations about countries

Each task defines specific causal variables that methods must learn to identify and manipulate.

## Supported Models

The benchmark evaluates methods on four language models:
- GPT-2 Small
- Qwen-2.5 (0.5B)
- Gemma-2 (2B)
- Llama-3.1 (8B)

## Creating a Submission

### Submission Structure

Your submission should follow this directory structure:

```
submission_folder/
├── featurizer.py              # (Optional) Custom featurizer implementation
├── token_position.py          # (Optional) Custom token position logic
└── {TASK}_{MODEL}_{VARIABLE}/
        ├── {ModelUnit}_featurizer
        ├── {ModelUnit}_inverse_featurizer
        └── {ModelUnit}_indices
```

For IOI tasks, you'll also need linear parameters for the high-level causal model:
```
submission_folder/
└── ioi_linear_params.json    # Required for IOI submissions
```
These linear parameters can be computed from training data, e.g.,  
```
baselines/ioi_baselines/ioi_learn_linear_params.py --model qwen --heads_list "(4,5)" "(12,0)"
```
This code estimates the weight for the position and token variables in the IOI task for the attention head 5 in layer 4 and attention head 0 in layer 12 of the qwen model. 


### Submission Components

1. **Featurizer Files**: PyTorch checkpoint files containing:
   - Trained featurizer weights
   - Inverse featurizer weights  
   - Feature indices

2. **Optional Custom Code**:
   - `featurizer.py`: Custom featurizer class inheriting from `Featurizer`
   - `token_position.py`: Function returning `TokenPosition` objects

3. **Task-Specific Requirements**:
   - Standard tasks: Organize by task/model/variable naming convention
   - IOI task: Include linear parameters JSON file

### Verifying Your Submission

Before submitting, verify your submission format:

```bash
python verify_submission.py /path/to/your/submission
```

## Evaluation

Submissions are evaluated by:

1. Loading your trained featurizers
2. Performing interchange interventions on test datasets
3. Measuring accuracy of causal variable predictions
4. Computing metrics across layers and datasets

Key metrics include:
- **Average Score**: Mean accuracy across examples
- **Average Accuracy Across Layers**: Mean of best accuracies per layer
- **Highest Accuracy Across Layers**: Maximum accuracy achieved

## Running Baseline Methods Used in the Paper

To replicate baseline results:

```bash
# Run baselines for a specific task
python baselines/simple_MCQA_baselines.py
python baselines/ARC_baselines.py
python baselines/arithmetic_baselines.py
python baselines/ravel_baselines.py
python baselines/ioi_baselines/ioi_baselines.py
```

## Citation

If you use this benchmark, please cite:

```bibtex
@article{mib-2025,
    title = {{MIB}: A Mechanistic Interpretability Benchmark},
    author = {Aaron Mueller and Atticus Geiger and Sarah Wiegreffe and Dana Arad and Iv{\'a}n Arcuschin and Adam Belfki and Yik Siu Chan and Jaden Fiotto-Kaufman and Tal Haklay and Michael Hanna and Jing Huang and Rohan Gupta and Yaniv Nikankin and Hadas Orgad and Nikhil Prakash and Anja Reusch and Aruna Sankaranarayanan and Shun Shao and Alessandro Stolfo and Martin Tutek and Amir Zur and David Bau and Yonatan Belinkov},
    year = {2025},
    journal = {CoRR},
    volume = {arXiv:2504.13151},
    url = {https://arxiv.org/abs/2504.13151v1}
}
```

## Resources

- [MIB Paper](https://arxiv.org/abs/2504.13151)
- [MIB Website](https://mib-bench.github.io)
- [Leaderboard](https://huggingface.co/spaces/mib-bench/leaderboard)
- [Datasets](https://huggingface.co/collections/mib-bench/mib-datasets-67f55273612ec3067a42a56b)
- [Main MIB Repository](https://github.com/aaronmueller/MIB)
