import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tasks.two_digit_addition_task.arithmetic import get_token_positions, get_counterfactual_datasets, get_causal_model
from experiments.aggregate_experiments import residual_stream_baselines
from neural.pipeline import LMPipeline
from experiments.filter_experiment import FilterExperiment
import torch
import gc
import os

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run arithmetic experiments with optional flags.")
    parser.add_argument("--skip_gemma", action="store_true", help="Skip running experiments for Gemma model.")
    parser.add_argument("--skip_llama", action="store_true", help="Skip running experiments for Llama model.")
    parser.add_argument("--use_gpu1", action="store_true", help="Use GPU1 instead of GPU0 if available.")
    parser.add_argument("--methods", nargs="+", 
                        default=["full_vector", "DAS", "DBM+SVD", "DBM+PCA", "DBM", "DBM+SAE"], 
                        help="List of methods to run")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=1024, help="Batch size for evaluation")
    parser.add_argument("--results_dir", type=str, default="arithmetic_results", help="Directory to save results")
    parser.add_argument("--model_dir", type=str, default="arithmetic_models", help="Directory to save trained models")
    parser.add_argument("--quick_test", action="store_true", help="Run quick test with reduced dataset size and layers")
    args = parser.parse_args()

    # Clear memory before starting
    gc.collect()
    torch.cuda.empty_cache()

    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.use_gpu1 and torch.cuda.is_available():
        device = "cuda:1"
    
    # Check function for evaluating model outputs
    def checker(output_text, expected):
        # Clean the output by extracting just the numbers
        import re
        numbers_in_output = re.findall(r'\d+', output_text)
        if not numbers_in_output:
            return False
        
        # Get the first number found
        first_number = numbers_in_output[0]
        
        return first_number == expected
    
    # Function to clear memory between experiments
    def clear_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Get counterfactual datasets and causal model
    dataset_size = 10 if args.quick_test else 10000
    counterfactual_datasets = get_counterfactual_datasets(hf=True, size=dataset_size)
    causal_model = get_causal_model()

    # Print available datasets
    print("Available datasets:", counterfactual_datasets.keys())
    
    # Set up models to test
    models = []
    if not args.skip_gemma:
        models.append("google/gemma-2-2b")
    if not args.skip_llama:
        models.append("meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    for model_name in models:
        print(f"\n===== Testing model: {model_name} =====")
        
        # Set up LM Pipeline with appropriate max_new_tokens for each model
        if "llama" in model_name.lower():
            max_new_tokens = 1
        elif "gemma" in model_name.lower():
            max_new_tokens = 3
        else:
            max_new_tokens = 3
            
        pipeline = LMPipeline(model_name, max_new_tokens=max_new_tokens, device=device, dtype=torch.float16)
        pipeline.tokenizer.padding_side = "left"
        print("DEVICE:", pipeline.model.device)
        
        # Get a sample input and check model's prediction
        sampled_example = next(iter(counterfactual_datasets.values()))[0]
        print("INPUT:", sampled_example["input"])
        print("EXPECTED OUTPUT:", causal_model.run_forward(sampled_example["input"])["raw_output"])
        print("MODEL PREDICTION:", pipeline.dump(pipeline.generate(sampled_example["input"])))
        
        # Filter the datasets based on model performance
        print("\nFiltering datasets based on model performance...")
        exp = FilterExperiment(pipeline, causal_model, checker)
        filtered_datasets = exp.filter(counterfactual_datasets, verbose=True, batch_size=args.eval_batch_size)
        
        # Get token positions for intervention
        token_positions = get_token_positions(pipeline, causal_model)
        
        # Display token highlighting for a sample
        print("\nToken positions highlighted in samples:")
        for dataset in filtered_datasets.values():
            for token_position in token_positions:
                example = dataset[0]
                print(token_position.highlight_selected_token(example["input"]))
                break
            break
        
        # Clear memory before running experiments
        clear_memory()
        
        # Setup experiment configuration
        start = 0 
        end = 1 if args.quick_test else pipeline.get_num_layers()
        
        config = {
            "batch_size": args.batch_size, 
            "evaluation_batch_size": args.eval_batch_size, 
            "training_epoch": 1, 
            "n_features": 16, 
            "regularization_coefficient": 0.0, 
            "output_scores": False
        }
        
        # Adjust batch size for Llama
        if "llama" in model_name.lower():
            config["batch_size"] = 256 
            config["evaluation_batch_size"] = 1024
        
        # Prepare dataset names - based on the HF dataset structure
        names = ["random", "ones_carry"]
        
        # Make sure results and model directories exist
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        
        # Run experiments for ones_carry variable only
        print(f"\nRunning experiments for target variable: ones_carry")
        
        # Prepare train and test data dictionaries
        train_data = {}
        test_data = {}
        
        for name in names:
            if name + "_train" in filtered_datasets:
                train_data[name + "_train"] = filtered_datasets[name + "_train"]
            if name + "_test" in filtered_datasets:
                test_data[name + "_test"] = filtered_datasets[name + "_test"]
            if name + "_testprivate" in filtered_datasets:
                test_data[name + "_testprivate"] = filtered_datasets[name + "_testprivate"]
        
        residual_stream_baselines(
            pipeline=pipeline,
            task=causal_model,
            token_positions=token_positions,
            train_data=train_data,
            test_data=test_data,
            config=config,
            target_variables=["ones_carry"],
            checker=checker,
            start=start,
            end=end,
            verbose=True,
            model_dir=os.path.join(args.model_dir, "ones_carry"),
            results_dir=args.results_dir,
            methods=args.methods
        )
        clear_memory()
        
        # Clean up pipeline to free memory before starting next model
        del pipeline
        clear_memory()
    
    print("\nAll experiments completed.")