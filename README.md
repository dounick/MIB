<p align="center">
    <a>
    <img src="assets/mib_logo.png" alt="MIB Splash"/>
    </a>
</p>

<p align="center">
    <!-- <h3 align="center">A <b>M</b>echanistic <b>I</b>nterpretability <b>B</b>enchmark</h3> -->
        A benchmark for systematic comparison of <em>featurization</em> and <em>localization</em> methods.
        <br/>
        <sub>
            <strong>circuits · causal variables · localization · featurization · faithfulness · interchange interventions · SAEs · counterfactuals</strong>
        </sub>
</p>

<p align="center" style="color: #cccccc">
    <a href="LICENSE.md"><img height="20px" src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="Apache 2.0"></a>
    <a href="https://huggingface.co/collections/mib-bench/mib-datasets-67f55273612ec3067a42a56b"><img height="20px" src="https://img.shields.io/badge/data-purple.svg?logo=huggingface"></a>
    <a href="https://huggingface.co/spaces/mib-bench/leaderboard"><img height="20px" src="https://img.shields.io/badge/leaderboard-red.svg?logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48IS0tIFVwbG9hZGVkIHRvOiBTVkcgUmVwbywgd3d3LnN2Z3JlcG8uY29tLCBHZW5lcmF0b3I6IFNWRyBSZXBvIE1peGVyIFRvb2xzIC0tPgo8c3ZnIHdpZHRoPSI4MDBweCIgaGVpZ2h0PSI4MDBweCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNMTUgMTlIOVYxMi41VjguNkM5IDguMjY4NjMgOS4yNjg2MyA4IDkuNiA4SDE0LjRDMTQuNzMxNCA4IDE1IDguMjY4NjMgMTUgOC42VjE0LjVWMTlaIiBzdHJva2U9IiNmNWY1ZjUiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPHBhdGggZD0iTTE1IDVIOSIgc3Ryb2tlPSIjZjVmNWY1IiBzdHJva2Utd2lkdGg9IjEuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+CjxwYXRoIGQ9Ik0yMC40IDE5SDE1VjE1LjFDMTUgMTQuNzY4NiAxNS4yNjg2IDE0LjUgMTUuNiAxNC41SDIwLjRDMjAuNzMxNCAxNC41IDIxIDE0Ljc2ODYgMjEgMTUuMVYxOC40QzIxIDE4LjczMTQgMjAuNzMxNCAxOSAyMC40IDE5WiIgc3Ryb2tlPSIjZjVmNWY1IiBzdHJva2Utd2lkdGg9IjEuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+CjxwYXRoIGQ9Ik05IDE5VjEzLjFDOSAxMi43Njg2IDguNzMxMzcgMTIuNSA4LjQgMTIuNUgzLjZDMy4yNjg2MyAxMi41IDMgMTIuNzY4NiAzIDEzLjFWMTguNEMzIDE4LjczMTQgMy4yNjg2MyAxOSAzLjYgMTlIOVoiIHN0cm9rZT0iI2Y1ZjVmNSIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4="></a>
    <a href="https://arxiv.org/abs/2504.13151"><img height="20px" src="https://img.shields.io/badge/paper-brown.svg?logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBVcGxvYWRlZCB0bzogU1ZHIFJlcG8sIHd3dy5zdmdyZXBvLmNvbSwgR2VuZXJhdG9yOiBTVkcgUmVwbyBNaXhlciBUb29scyAtLT4KPCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KPHN2ZyBmaWxsPSIjZmZmZmZmIiB2ZXJzaW9uPSIxLjEiIGlkPSJFYmVuZV8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiAKCSB3aWR0aD0iODAwcHgiIGhlaWdodD0iODAwcHgiIHZpZXdCb3g9IjAgMCA2NCA2NCIgZW5hYmxlLWJhY2tncm91bmQ9Im5ldyAwIDAgNjQgNjQiIHhtbDpzcGFjZT0icHJlc2VydmUiPgo8Zz4KCTxwYXRoIGQ9Ik01Miw2NGMyLjIwNiwwLDQtMS43OTQsNC00VjRjMC0yLjIwNi0xLjc5NC00LTQtNEgyNmMtMS42MzgsMC0zLjY2OCwwLjg0MS00LjgyOCwyTDEwLDEzLjE3MUM4Ljg0MSwxNC4zMyw4LDE2LjM2LDgsMTh2NDIKCQljMCwyLjIwNiwxLjc5NCw0LDQsNEg1MnogTTUyLDRsMCw1NmgwSDEyVjE4aDEwYzIuMjA2LDAsNC0xLjc5NCw0LTRWNEg1MnogTTIyLDYuODI4VjE0aC03LjE3MkwyMiw2LjgyOHoiLz4KCTxwYXRoIGQ9Ik0zMiwxNmgxMmMxLjEwNCwwLDItMC44OTYsMi0ycy0wLjg5Ni0yLTItMkgzMmMtMS4xMDQsMC0yLDAuODk2LTIsMlMzMC44OTYsMTYsMzIsMTZ6Ii8+Cgk8cGF0aCBkPSJNNDQsMjJIMjBjLTEuMTA0LDAtMiwwLjg5Ni0yLDJzMC44OTYsMiwyLDJoMjRjMS4xMDQsMCwyLTAuODk2LDItMlM0NS4xMDQsMjIsNDQsMjJ6Ii8+Cgk8cGF0aCBkPSJNNDQsMzJIMjBjLTEuMTA0LDAtMiwwLjg5Ni0yLDJzMC44OTYsMiwyLDJoMjRjMS4xMDQsMCwyLTAuODk2LDItMlM0NS4xMDQsMzIsNDQsMzJ6Ii8+Cgk8cGF0aCBkPSJNNDQsNDJIMjBjLTEuMTA0LDAtMiwwLjg5Ni0yLDJzMC44OTYsMiwyLDJoMjRjMS4xMDQsMCwyLTAuODk2LDItMlM0NS4xMDQsNDIsNDQsNDJ6Ii8+Cgk8cGF0aCBkPSJNNDQsNTJIMjBjLTEuMTA0LDAtMiwwLjg5Ni0yLDJzMC44OTYsMiwyLDJoMjRjMS4xMDQsMCwyLTAuODk2LDItMlM0NS4xMDQsNTIsNDQsNTJ6Ii8+CjwvZz4KPC9zdmc+"></a>
    <a href="mailto:mueller@technion.ac.il"><img height="20px" src="https://img.shields.io/badge/contact-gray.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGcgaWQ9IlNWR1JlcG9fYmdDYXJyaWVyIiBzdHJva2Utd2lkdGg9IjAiPjwvZz48ZyBpZD0iU1ZHUmVwb190cmFjZXJDYXJyaWVyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjwvZz48ZyBpZD0iU1ZHUmVwb19pY29uQ2FycmllciI+IDxwYXRoIGQ9Ik00IDcuMDAwMDVMMTAuMiAxMS42NUMxMS4yNjY3IDEyLjQ1IDEyLjczMzMgMTIuNDUgMTMuOCAxMS42NUwyMCA3IiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48L3BhdGg+IDxyZWN0IHg9IjMiIHk9IjUiIHdpZHRoPSIxOCIgaGVpZ2h0PSIxNCIgcng9IjIiIHN0cm9rZT0iI2ZmZmZmZiIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiPjwvcmVjdD4gPC9nPjwvc3ZnPg==" alt="Email"></a>
    <a href="https://mib-bench.github.io"><img height="20px" src="https://img.shields.io/badge/website-black.svg" alt="website"></a>
</p>

- [About MIB](#overview)
- [Tracks](#tracks)
    - [Circuit Localization](#circuit-localization)
    - [Causal Variable Localization](#causal-variable-localization)
- [Data and Models](#data-and-models)
- [Leaderboard](#leaderboard)
- [Submission](#submission)
- [Citation](#citation)

# Overview

This repository documents the [**M**echanistic **I**nterpretability **B**enchmark (**MIB**)](https://arxiv.org/abs/2504.13151). Here, you can find links to the code for both tracks, information about and links to our datasets, information about the **MIB** leaderboard, instructions for submission, and a citation that we ask you use if you use any of the resources provided here or at the links provided below.

# Tracks

**MIB** contains two tracks. The [*circuit localization track*](#circuit-localization) benchmarks methods that aim to locate graphs of causal dependencies in neural networks. The [*causal variable localization track*](#causal-variable-localization) benchmarks methods that aim to locate specific human-interpretable causal variables in neural networks.

## Circuit Localization
<p align="center">
    <a>
    <img src="assets/circuit_track.png" alt="Overview of the circuit localization track."/>
    </a>
</p>

This track benchmarks circuit discovery methods—i.e., methods for locating graphs of causal dependencies in neural networks. Most circuit discovery pipelines look something like this:

1. Compute importance scores for each component or each edge between components.
2. Ablate all components from the network except those that surpass some importance threshold, or those in the top k%.
3. Evaluate how well the circuit (model with only the most important components not ablated) performs, or replicates the full model's behavior.

In the [circuit localization track's repository](https://github.com/hannamw/MIB-circuit-track), we provide code for discovering and evaluating circuits.

Notice that step (3) mentioned *two distinct* evaluation criteria: how well the circuit performs, and how well the circuit replicates the model's behavior. Past work often implicitly conflates these two, whether by discovering a circuit using one criterion and then evaluating another, or by not clarifying the precise goals. We believe these are complementary but separate concepts, so we split them into two separate evaluation metrics. See [circuit localization repo](https://github.com/hannamw/MIB-circuit-track) or [the paper](https://arxiv.org/abs/2504.13151) for more details.

## Causal Variable Localization
<p align="center">
    <a>
    <img src="assets/causal_variable_track.png" alt="Overview of the causal variable localization track."/>
    </a>
</p>

This track benchmarks featurization methods—i.e., methods for transforming model activations into a space where it's easier to isolate a given causal variable. Most pipelines under this paradigm look like this:

1. Curate a dataset of contrastive pairs, where each pair differs only with respect to the targeted causal variable.
2. If using a supervised method, train the featurization method using the contrastive pairs.
3. To evaluate: feed the model an input from a pair, use the featurizer to transform an activation vector, intervene in the transformed space, transform back out, and see whether the model's new behavior aligns with what is expected under the intervention.

In the [causal variable localization track's repository](https://github.com/atticusg/CausalAbstraction), we provide code for training and evaluating featurizers.

We provide the results from our baseline experiments in [this Google Drive folder](https://drive.google.com/drive/folders/1StruQr0Cv3ay4dnFfcqZuwTDN4gn8vGS?usp=sharing). These results were generated using the scripts provided in this repo.

# Data and Models

Our benchmark consists of five datasets: IOI, MCQA, Arithmetic, ARC, and RAVEL. These were chosen to represent (1) a mixture of commonly studied and unstudied tasks, (2) tasks of varying formats, and (3) tasks of varying difficulty levels. Each dataset comes with a train, validation and public test set. We also hold out a private test set, which can only be evaluated on by submitting to [the leaderboard](https://huggingface.co/spaces/mib-bench/leaderboard).

Mechanistic interpretability comparisons are only valid for a given task-and-model pair. Thus, we choose four realisitic language models of varying sizes and capability levels to standardize comparisons: GPT-2 Small, Qwen-2.5 (0.5B), Gemma-2 (2B), Llama-3.1 (8B). We also include an InterpBench model, which we train to encode a ground-truth circuit for the IOI task.

## InterpBench: A Ground-truth Circuit

Language models, when trained according to a next-token prediction objective, encode unpredictable mechanisms and concepts. Thus, it is difficult to define metrics that capture any notion of a ground-truth. For the circuit localization track, we therefore also include an InterpBench model, which we train to encode a known mechanism that we specify. Because we know what the ground-truth nodes and edges are in the circuit, we can compute more meaningful metrics, like the AUROC!

For more detail on how InterpBench models are trained, please see the [InterpBench paper](https://arxiv.org/abs/2407.14494).

# Leaderboard

To encourage participation, we have created a [leaderboard, hosted on HuggingFace](https://huggingface.co/spaces/mib-bench/leaderboard). This leaderboard shows scores on the *private* test set. We have set a strict rate limit of 2 submission per user per week to discourage hill-climbing on the private test set.

Our hope is that the public validation and test sets will enable fast iteration on mechanistic interpretability methods, while the private test will remain a more stable and meaningful measure of the state of the art.

# Submission

Submit to the **MIB** leaderboard at [this link](https://huggingface.co/spaces/mib-bench/leaderboard). You will need the following:
- Circuit localization: 9 circuits per model/task combination. These should be of varying sizes, and satisfy the criteria described in the circuit localization repo. See [here](https://huggingface.co/mib-bench/mib-circuits-example/tree/main/importances/pt) for an example submission with one .pt file per model/task, and [here](https://huggingface.co/mib-bench/mib-circuits-example/tree/main/multiple_circuits/pt) for an example submission with separate circuit files for each circuit size threshold per model/task. See the [circuit localization track repository](https://github.com/hannamw/MIB-circuit-track) for more details on the format of these files.
- Causal variable localization: a featurization function that follows the API format specified in the causal variable localization repo, a token position function specifying where the featurizer should be applied, and a folder containing trained (inverse) featurizers and token indices. See [here](https://huggingface.co/mib-bench/mib-causalvariable-example/tree/main) for an example submission. See the [causal variable localization track repository](https://github.com/atticusg/CausalAbstraction) for more details on the format of these files, see the Jupyter notebook [main/MIB-causal-variable-track/ioi_example_submission.ipynb]](https://github.com/aaronmueller/MIB/blob/main/MIB-causal-variable-track/ioi_example_submission.ipynb) for an example of how to get the trained featurizer and token indices files for the ioi task and the Jupyter notebook [MIB-causal-variable-track/ioi_example_submission.ipynb]](https://github.com/aaronmueller/MIB/blob/main/MIB-causal-variable-track/example_submission.ipynb) for an example of how to get the trained featurizer and token indices files for the rest of the tasks. The README [MIB-causal-variable-track/README.md](https://github.com/aaronmueller/MIB/blob/main/MIB-causal-variable-track/READ_ME.md) explains how to structure the folder for a submission to this track. 

The leaderboard submission portal will verify that your submission is in the correct format. For the circuit localization track, this is done on our backend.

For the causal variable localization track, please ensure that your submission is valid using our [automated submission checker script](https://github.com/aaronmueller/MIB/blob/main/causal_variable_verify_submission.py). Once you've verified, please provide the requested HF repository linking to your files. This should be a model repository, not a dataset.

# Citation
If you use any of the **MIB** datasets or code, please cite [our paper](https://arxiv.org/abs/2504.13151):
```
@article{mib-2025,
	title = {{MIB}: A Mechanistic Interpretability Benchmark},
	author = {Aaron Mueller and Atticus Geiger and Sarah Wiegreffe and Dana Arad and Iv{\'a}n Arcuschin and Adam Belfki and Yik Siu Chan and Jaden Fiotto-Kaufman and Tal Haklay and Michael Hanna and Jing Huang and Rohan Gupta and Yaniv Nikankin and Hadas Orgad and Nikhil Prakash and Anja Reusch and Aruna Sankaranarayanan and Shun Shao and Alessandro Stolfo and Martin Tutek and Amir Zur and David Bau and Yonatan Belinkov},
	year = {2025},
	journal = {CoRR},
	volume = {arXiv:2504.13151},
	url = {https://arxiv.org/abs/2504.13151v1}
}
```

# License
We release the content in this repository and all sub-repositories under an Apache 2.0 license.
