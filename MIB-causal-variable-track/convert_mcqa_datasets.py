"""
MCQA Dataset Converter for DAS Pipeline
Converts MCQA datasets to a format compatible with the DAS training pipeline,
with model-specific filtering to remove incorrectly answered examples.
"""

import sys
from pathlib import Path
import argparse
import torch
import gc
from tqdm import tqdm
import pickle
import json
from typing import Dict, List, Tuple, Any

# Add parent directories to path for imports
sys.path.append(str(Path().resolve().parent.parent))

from tasks.simple_MCQA.simple_MCQA import get_token_positions, get_counterfactual_datasets, get_causal_model
from CausalAbstraction.neural.pipeline import LMPipeline
from CausalAbstraction.experiments.filter_experiment import FilterExperiment
from transformers import AutoTokenizer


def clear_memory():
    """Clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def checker(output_text: str, expected: str) -> bool:
    """Check if model output matches expected answer."""
    return expected in output_text


def format_mcqa_prompt(example: Dict) -> str:
    """
    Format an MCQA example into a prompt string.
    
    Args:
        example: Dictionary with keys like 'question', 'choice0', 'choice1', etc.
    
    Returns:
        str: Formatted prompt
    """
    # Extract question - handle both list and string formats
    if isinstance(example['question'], list):
        question_text = example['question'][1].replace('question: ', '').strip()
        question_text = question_text.capitalize()
    else:
        question_text = example['question']
    
    # Build the prompt
    prompt = f"Question: {question_text}\n"
    prompt += f"A. {example['choice0']}\n"
    prompt += f"B. {example['choice1']}\n"
    prompt += f"C. {example['choice2']}\n"
    prompt += f"D. {example['choice3']}\n"
    prompt += "Answer:"
    
    return prompt


def find_answer_token_position(tokenizer, prompt: str, answer: str) -> int:
    """
    Find the position where the answer token should appear.
    
    Args:
        tokenizer: HuggingFace tokenizer
        prompt: The full prompt string
        answer: The answer letter (A, B, C, or D)
    
    Returns:
        int: Position index for the answer token
    """
    # Tokenize the full prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    return len(tokens) - 1  # Answer goes after the last token (after "Answer:")


def get_answer_token_id(tokenizer, answer: str) -> int:
    """
    Get the token ID for an answer letter, handling different tokenization schemes.
    
    Args:
        tokenizer: HuggingFace tokenizer
        answer: The answer letter (A, B, C, or D)
    
    Returns:
        int: Token ID for the answer
    """
    # Try different tokenization approaches
    candidates = [
        f" {answer}",  # With leading space
        answer,        # Without space
        f"{answer}.",  # With period
        f" {answer}.", # With space and period
    ]
    
    # Use the first candidate that produces a single token
    for candidate in candidates:
        tokens = tokenizer.encode(candidate, add_special_tokens=False)
        if len(tokens) == 1:
            return tokens[0]
    
    # Fallback: return the last token of " {answer}"
    return tokenizer.encode(f" {answer}", add_special_tokens=False)[-1]


def get_correct_answer(example_dict: Dict) -> str:
    """
    Extract the correct answer letter from an MCQA example.
    
    Args:
        example_dict: MCQA example dictionary
    
    Returns:
        Answer letter (A, B, C, or D)
    """
    # The question field contains [answer, question_text]
    question = example_dict.get('question', [])
    if isinstance(question, list) and len(question) > 0:
        correct_answer_text = question[0]  # e.g., 'brown'
        
        # Find which choice matches
        for i in range(4):
            choice_key = f'choice{i}'
            symbol_key = f'symbol{i}'
            if example_dict.get(choice_key) == correct_answer_text:
                return example_dict.get(symbol_key, chr(65 + i))  # Return A, B, C, or D
    
    # Fallback: return A
    return 'A'


def convert_example_to_das_format(
    base_example: Dict,
    cf_example: Dict,
    tokenizer,
    answer_token_position: int
) -> Tuple[List[int], List[int], List[int], int]:
    """
    Convert a single MCQA example pair to DAS format.
    
    Args:
        base_example: Base example dictionary (from 'input' key)
        cf_example: Counterfactual example dictionary (from 'counterfactual_inputs' list)
        tokenizer: HuggingFace tokenizer
        answer_token_position: Position where answer should be predicted
    
    Returns:
        Tuple of (input_ids, source_input_ids, labels, intervention_id)
    """
    # Format prompts - use raw_input if available, otherwise format from components
    if 'raw_input' in base_example:
        base_prompt = base_example['raw_input']
    else:
        base_prompt = format_mcqa_prompt(base_example)
    
    if 'raw_input' in cf_example:
        cf_prompt = cf_example['raw_input']
    else:
        cf_prompt = format_mcqa_prompt(cf_example)
    
    # Tokenize prompts
    base_tokens = tokenizer.encode(base_prompt, add_special_tokens=True)
    cf_tokens = tokenizer.encode(cf_prompt, add_special_tokens=True)
    
    # Get answer token ID
    base_answer = get_correct_answer(base_example)
    answer_token_id = get_answer_token_id(tokenizer, base_answer)
    
    # Create labels (all -100 except at answer position)
    labels = [-100] * len(base_tokens)
    # Add the answer token at the end
    labels.append(answer_token_id)
    
    return base_tokens, cf_tokens, labels, 0


def convert_filtered_datasets(
    filtered_datasets: Dict[str, List],
    tokenizer,
    causal_model,
    token_positions: List
) -> Dict[str, Dict[str, List]]:
    """
    Convert all filtered datasets to DAS format.
    
    Args:
        filtered_datasets: Dictionary of filtered MCQA datasets
        tokenizer: HuggingFace tokenizer
        causal_model: Causal model for getting answers
        token_positions: Token position information
    
    Returns:
        Dictionary with converted datasets in DAS format
    """
    converted = {}
    
    # Get answer token position from a sample
    if token_positions and len(filtered_datasets) > 0:
        sample_dataset = next(iter(filtered_datasets.values()))
        if len(sample_dataset) > 0:
            sample_example = sample_dataset[0]
            # Use the base input from the 'input' key
            sample_base = sample_example['input']
            if 'raw_input' in sample_base:
                sample_prompt = sample_base['raw_input']
            else:
                sample_prompt = format_mcqa_prompt(sample_base)
            
            # Get the correct answer
            sample_answer = get_correct_answer(sample_base)
            answer_token_position = find_answer_token_position(
                tokenizer, 
                sample_prompt, 
                sample_answer
            )
        else:
            answer_token_position = -1
    else:
        answer_token_position = -1
    
    print(f"\nAnswer token position: {answer_token_position}")
    
    for dataset_name, dataset in filtered_datasets.items():
        print(f"\nConverting dataset: {dataset_name} ({len(dataset)} examples)...")
        
        input_ids_list = []
        source_input_ids_list = []
        labels_list = []
        intervention_ids_list = []
        
        for example in tqdm(dataset, desc=f"Converting {dataset_name}"):
            # Structure: example has 'input' (base) and 'counterfactual_inputs' (list of CFs)
            base_input = example['input']
            cf_input = example['counterfactual_inputs'][0]  # Take first counterfactual
            
            # Convert to DAS format
            input_ids, source_input_ids, labels, intervention_id = convert_example_to_das_format(
                base_input,
                cf_input,
                tokenizer,
                answer_token_position
            )
            
            input_ids_list.append(input_ids)
            source_input_ids_list.append(source_input_ids)
            labels_list.append(labels)
            intervention_ids_list.append(intervention_id)
        
        converted[dataset_name] = {
            'input_ids': input_ids_list,
            'source_input_ids': source_input_ids_list,
            'labels': labels_list,
            'intervention_ids': intervention_ids_list,
            'metadata': {
                'num_examples': len(dataset),
                'answer_token_position': answer_token_position,
                'dataset_name': dataset_name
            }
        }
        
        print(f"Converted {len(input_ids_list)} examples for {dataset_name}")
    
    return converted


def save_converted_datasets(
    converted_datasets: Dict,
    output_dir: str,
    model_name: str,
    tokenizer_info: Dict
):
    """
    Save converted datasets to disk.
    
    Args:
        converted_datasets: Dictionary of converted datasets
        output_dir: Directory to save to
        model_name: Name of the model used
        tokenizer_info: Information about tokenizer settings
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle for easy loading
    pickle_path = output_path / "mcqa_datasets.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(converted_datasets, f)
    print(f"\nSaved pickle to: {pickle_path}")
    
    # Save metadata as JSON
    metadata = {
        'model_name': model_name,
        'tokenizer_info': tokenizer_info,
        'datasets': {
            name: data['metadata'] 
            for name, data in converted_datasets.items()
        },
        'total_examples': sum(
            data['metadata']['num_examples'] 
            for data in converted_datasets.values()
        )
    }
    
    json_path = output_path / "mcqa_metadata.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {json_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Total datasets: {len(converted_datasets)}")
    print(f"Total examples: {metadata['total_examples']}")
    print("\nDataset breakdown:")
    for name, data in converted_datasets.items():
        print(f"  {name}: {data['metadata']['num_examples']} examples")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MCQA datasets to DAS pipeline format with model-specific filtering"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b",
        help="Model to use for filtering"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/mcqa",
        help="Directory to save processed datasets"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for filtering"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for model"
    )
    parser.add_argument(
        "--load_private_data",
        action="store_true",
        help="Load private MCQA data"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Limit dataset size (None for full dataset)"
    )
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    print("="*60)
    print("MCQA DATASET CONVERTER")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    
    # Load datasets
    print("\nLoading MCQA datasets...")
    counterfactual_datasets = get_counterfactual_datasets(
        hf=True,
        size=args.size,
        load_private_data=args.load_private_data
    )
    print(f"Loaded {len(counterfactual_datasets)} datasets")
    print(f"Available datasets: {list(counterfactual_datasets.keys())}")
    
    # Get causal model
    causal_model = get_causal_model()
    
    # Load model pipeline
    print(f"\nLoading model pipeline: {args.model}")
    pipeline = LMPipeline(
        args.model,
        max_new_tokens=1,
        device=args.device,
        dtype=dtype
    )
    pipeline.tokenizer.padding_side = "left"
    print(f"Model loaded on device: {pipeline.model.device}")
    
    # Test model on a sample
    print("\nTesting model on sample input...")
    sampled_example = next(iter(counterfactual_datasets.values()))[0]
    print(f"Input: {sampled_example['input']['raw_input'][:100]}...")
    expected = causal_model.run_forward(sampled_example["input"])["raw_output"]
    prediction = pipeline.dump(pipeline.generate(sampled_example["input"]))
    print(f"Expected: {expected}")
    print(f"Model prediction: {prediction}")
    print(f"Match: {expected in prediction}")
    
    # Filter datasets based on model performance
    print("\n" + "="*60)
    print("FILTERING DATASETS")
    print("="*60)
    exp = FilterExperiment(pipeline, causal_model, checker)
    filtered_datasets = exp.filter(
        counterfactual_datasets,
        verbose=True,
        batch_size=args.batch_size
    )
    
    # Get token positions
    token_positions = get_token_positions(pipeline, causal_model)
    
    # Display token highlighting for a sample
    print("\n" + "="*60)
    print("TOKEN POSITIONS")
    print("="*60)
    for dataset in filtered_datasets.values():
        if len(dataset) > 0:
            for token_position in token_positions:
                example = dataset[0]
                highlighted = token_position.highlight_selected_token(
                    example["counterfactual_inputs"][0]
                )
                print(f"Sample with highlighted answer position:\n{highlighted}")
                break
            break
    
    # Clear memory before conversion
    clear_memory()
    
    # Convert datasets to DAS format
    print("\n" + "="*60)
    print("CONVERTING TO DAS FORMAT")
    print("="*60)
    converted_datasets = convert_filtered_datasets(
        filtered_datasets,
        pipeline.tokenizer,
        causal_model,
        token_positions
    )
    
    # Save converted datasets
    tokenizer_info = {
        'padding_side': pipeline.tokenizer.padding_side,
        'vocab_size': pipeline.tokenizer.vocab_size,
        'model_max_length': pipeline.tokenizer.model_max_length
    }
    
    save_converted_datasets(
        converted_datasets,
        args.output_dir,
        args.model,
        tokenizer_info
    )
    
    print("\n✓ Conversion complete!")


if __name__ == "__main__":
    main()