import json
import csv
from tqdm import tqdm

def convert_two_digit_addition_to_das_format(filtered_datasets, tokenizer, output_prefix="two_digit_addition"):
    """
    Converts filtered_datasets from the two-digit arithmetic pipeline into
    the same CSV/JSON format used by the price-tagging DAS task.
    """

    def compute_sum(example_dict):
        """Compute the numeric sum based on digit fields."""
        op1 = 10 * example_dict["op1_tens"] + example_dict["op1_ones"]
        op2 = 10 * example_dict["op2_tens"] + example_dict["op2_ones"]
        return op1 + op2

    def carry_flag(example_dict):
        return 1 if (example_dict["op1_ones"] + example_dict["op2_ones"]) >= 10 else 0

    # def format_prompt(example_dict):
    #     """Use raw_input if available, otherwise rebuild the prompt."""
    #     if "raw_input" in example_dict:
    #         return example_dict["raw_input"]
    #     else:
    #         op1 = 10 * example_dict["op1_tens"] + example_dict["op1_ones"]
    #         op2 = 10 * example_dict["op2_tens"] + example_dict["op2_ones"]
    #         return f"{op1} + {op2} = "
    def format_prompt(example_dict):
        """The default input prompts are mixed in format, maybe its better to standardize? This might 
        be how MIB does it too but im not sure..."""
        op1 = 10 * example_dict["op1_tens"] + example_dict["op1_ones"]
        op2 = 10 * example_dict["op2_tens"] + example_dict["op2_ones"]
        return f"Q: What is {op1} + {op2} = "
    
    def tokenize_and_make_labels(prompt_text, label_text):
        enc = tokenizer(prompt_text, add_special_tokens=True)
        # take final token id for label (works with Llama number-tokenization behavior)
        label_token_id = tokenizer(label_text, add_special_tokens=False)["input_ids"][-1]
        labels = [-100] * (len(enc["input_ids"]) - 1) + [label_token_id]
        return enc["input_ids"], labels

    for name, dataset in filtered_datasets.items():
        rows = []
        print(f"Processing {name} ({len(dataset)} examples)...")

        for ex in tqdm(dataset):
            base = ex["input"]
            cf = ex["counterfactual_inputs"][0]

            base_prompt = format_prompt(base)
            cf_prompt = format_prompt(cf)

            # compute base sum and carry flags
            base_sum = compute_sum(base)
            base_c = carry_flag(base)
            cf_c = carry_flag(cf)

            # compute post-intervention sum = base_sum +/- 10 depending on carry flip
            post_sum = base_sum + (cf_c - base_c) * 10
            label_text = str(post_sum)

            base_ids, labels = tokenize_and_make_labels(base_prompt, label_text)
            cf_ids, _ = tokenize_and_make_labels(cf_prompt, label_text)  # source tokens (label_text only used for labels shaping)

            rows.append({
                "input_ids": json.dumps(base_ids),
                "source_input_ids": json.dumps(cf_ids),
                "labels": json.dumps(labels),
                "intervention_ids": 0,
            })

        csv_path = f"{output_prefix}_{name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["input_ids", "source_input_ids", "labels", "intervention_ids"])
            writer.writeheader()
            writer.writerows(rows)

        json_path = f"{output_prefix}_{name}.json"
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)

        print(f"✅ Saved {csv_path} and {json_path}")

from transformers import AutoTokenizer

import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent.parent))

#from tasks.simple_MCQA.simple_MCQA import get_token_positions, get_counterfactual_datasets, get_causal_model
from tasks.two_digit_addition_task.arithmetic import get_token_positions, get_counterfactual_datasets, get_causal_model
from CausalAbstraction.experiments.aggregate_experiments import residual_stream_baselines
from CausalAbstraction.neural.pipeline import LMPipeline
from CausalAbstraction.experiments.filter_experiment import FilterExperiment
import gc
import torch
import os
from tqdm import tqdm
from CausalAbstraction.causal.counterfactual_dataset import CounterfactualDataset
import copy

gc.collect()
torch.cuda.empty_cache()
def checker(output_text, expected):
    # Clean the output by extracting just the numbers
    import re
    #print("checking")
    numbers_in_output = re.findall(r'\d+', output_text)
    if not numbers_in_output:
        return False
    
    # Get the first number found
    first_number = numbers_in_output[0]
    
    # Handle the case where expected has leading zero and output doesn't
    if expected[0] == "0":
        expected_no_leading_zero = expected[1:]
        return first_number == expected_no_leading_zero or first_number == expected
    return first_number == expected

model_name = "meta-llama/Llama-3.1-8B"
device = "cuda:0"
counterfactual_datasets = get_counterfactual_datasets(hf=True, size=None, load_private_data=False)

def standardize_prompt(example):
    # Update input raw_input
    op1 = example['input']['op1_tens'] * 10 + example['input']['op1_ones']
    op2 = example['input']['op2_tens'] * 10 + example['input']['op2_ones']
    example['input']['raw_input'] = f'{op1} + {op2} = '
    
    # Update counterfactual_inputs raw_input
    cf_op1 = example['counterfactual_inputs'][0]['op1_tens'] * 10 + example['counterfactual_inputs'][0]['op1_ones']
    cf_op2 = example['counterfactual_inputs'][0]['op2_tens'] * 10 + example['counterfactual_inputs'][0]['op2_ones']
    example['counterfactual_inputs'][0]['raw_input'] = f'{cf_op1} + {cf_op2} = '
    
    return example

# Modify the underlying HuggingFace dataset for each CounterfactualDataset
for dataset_name in ['random_train', 'random_test', 'ones_carry_train', 'ones_carry_test']:
    counterfactual_datasets[dataset_name].dataset = counterfactual_datasets[dataset_name].dataset.map(standardize_prompt)

# Now check
print(counterfactual_datasets['random_train'][0])
print(counterfactual_datasets['random_test'][100])
print(counterfactual_datasets['ones_carry_train'][1000])
print(counterfactual_datasets['ones_carry_test'][-1])


#if want to standardize prompt need to change the filtering part too
#TODO
causal_model = get_causal_model()
pipeline = LMPipeline(model_name, max_new_tokens=1, device=device, dtype=torch.float16)
pipeline.tokenizer.padding_side = "left"

exp = FilterExperiment(pipeline, causal_model, checker)

tokenizer = AutoTokenizer.from_pretrained(model_name)
filtered_datasets = exp.filter(counterfactual_datasets, verbose=True, batch_size=1024)
for k, v in filtered_datasets.items():
    print(k, len(v))
    break
sample = filtered_datasets["random_train"][0]
import pprint
pprint.pprint(sample)
convert_two_digit_addition_to_das_format(filtered_datasets, tokenizer)
