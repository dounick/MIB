#!/usr/bin/env python3
"""
Comprehensive test script for the submission evaluation pipeline.

This script performs the following tests:
1. Run the notebook ioi_example_submission.ipynb and example_submission.ipynb
2. Assert that JSON files with the same name in results and results_loaded folders are identical
3. Run aggregate_results.py on all four results folders
4. Run evaluate_submission.py and ioi_evaluate_submission.py with private data flag
5. Run aggregate_results.py on submission folders after evaluation
6. Move submissions to all_submissions/ and run process_all_submission.py
7. Compare all aggregated results

Expected file structure:
- Notebooks create:
  - mock_submission/ (contains trained models)
  - mock_submission_results/ (evaluation results from notebook)
  - mock_submission_results_loaded/ (results from loading saved models)
  - ioi_submission/ (contains trained models)
  - ioi_submission_results/ (evaluation results from notebook)
  - ioi_submission_results_loaded/ (results from loading saved models)

- Evaluation scripts save results inside submission folders:
  - mock_submission/<task_folder>/*__results.json
  - ioi_submission/<task_folder>/*__results.json
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import traceback


def run_command(cmd, description, cwd=None):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        if result.returncode != 0:
            print(f"ERROR: Command failed with return code {result.returncode}")
            return False
            
        return True
    except Exception as e:
        print(f"ERROR running command: {e}")
        traceback.print_exc()
        return False


def run_notebook(notebook_path):
    """Execute a Jupyter notebook."""
    print(f"\nExecuting notebook: {notebook_path}")
    
    # Use nbconvert to execute the notebook
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=600",
        notebook_path
    ]
    
    return run_command(cmd, f"Executing {os.path.basename(notebook_path)}")


def compare_json_files(file1, file2):
    """Compare two JSON files and return True if identical."""
    print(f"\nComparing JSON files:")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")
    
    try:
        with open(file1, 'r') as f1:
            data1 = json.load(f1)
        with open(file2, 'r') as f2:
            data2 = json.load(f2)
        
        # Deep comparison
        if data1 == data2:
            print("  ✓ Files are identical")
            return True
        else:
            print("  ✗ Files differ")
            # Print first few differences for debugging
            print("  Sample differences:")
            for key in list(data1.keys())[:3]:
                if key not in data2:
                    print(f"    Key '{key}' missing in second file")
                elif data1[key] != data2[key]:
                    print(f"    Key '{key}' has different values")
            return False
            
    except Exception as e:
        print(f"  ERROR comparing files: {e}")
        traceback.print_exc()
        return False


def compare_folders(folder1, folder2):
    """Compare all JSON files in two folders."""
    print(f"\nComparing folders:")
    print(f"  Folder 1: {folder1}")
    print(f"  Folder 2: {folder2}")
    
    if not os.path.exists(folder1):
        print(f"  ERROR: Folder 1 does not exist")
        return False
    if not os.path.exists(folder2):
        print(f"  ERROR: Folder 2 does not exist")
        return False
    
    # Get JSON files from both folders
    files1 = {f for f in os.listdir(folder1) if f.endswith('.json')}
    files2 = {f for f in os.listdir(folder2) if f.endswith('.json')}
    
    if files1 != files2:
        print(f"  WARNING: Different files in folders")
        print(f"    Only in folder 1: {files1 - files2}")
        print(f"    Only in folder 2: {files2 - files1}")
    
    # Compare common files
    common_files = files1 & files2
    all_match = True
    
    for filename in sorted(common_files):
        file1_path = os.path.join(folder1, filename)
        file2_path = os.path.join(folder2, filename)
        
        if not compare_json_files(file1_path, file2_path):
            all_match = False
    
    return all_match and (files1 == files2)


def compare_aggregated_results(file1, file2, name1="Result 1", name2="Result 2"):
    """Compare two aggregated results files with detailed output."""
    print(f"\nComparing aggregated results:")
    print(f"  {name1}: {file1}")
    print(f"  {name2}: {file2}")
    
    try:
        with open(file1, 'r') as f:
            data1 = json.load(f)
        with open(file2, 'r') as f:
            data2 = json.load(f)
        
        # Check if keys match
        keys1 = set(data1.keys())
        keys2 = set(data2.keys())
        
        if keys1 != keys2:
            print(f"  WARNING: Different keys in results")
            print(f"    Only in {name1}: {keys1 - keys2}")
            print(f"    Only in {name2}: {keys2 - keys1}")
            return False
        
        # Compare values for common keys
        differences = []
        for key in sorted(keys1):
            if data1[key] != data2[key]:
                differences.append(key)
        
        if differences:
            print(f"  ✗ Found {len(differences)} differences")
            for key in differences[:5]:  # Show first 5 differences
                print(f"    Key '{key}':")
                print(f"      {name1}: {data1[key]}")
                print(f"      {name2}: {data2[key]}")
            if len(differences) > 5:
                print(f"    ... and {len(differences) - 5} more differences")
            return False
        else:
            print("  ✓ Results are identical")
            return True
            
    except Exception as e:
        print(f"  ERROR comparing results: {e}")
        traceback.print_exc()
        return False


def cleanup_directory(path):
    """Safely remove a directory if it exists."""
    if os.path.exists(path):
        print(f"Cleaning up: {path}")
        shutil.rmtree(path)


def main():
    """Run the comprehensive test pipeline."""
    print("Starting comprehensive submission pipeline test")
    print("=" * 80)
    
    # Track test results
    test_results = {
        "notebooks_executed": False,
        "submission_verification": False,
        "json_comparison": False,
        "initial_aggregation": False,
        "evaluation": False,
        "post_eval_aggregation": False,
        "process_all_submissions": False,
        "final_comparison": False
    }
    
    # Clean up any previous test artifacts
    test_artifacts = [
        "test_all_submissions",
        "test_all_results",
        "test_aggregated_mock.json",
        "test_aggregated_mock_loaded.json",
        "test_aggregated_ioi.json",
        "test_aggregated_ioi_loaded.json",
        "test_aggregated_mock_eval.json",
        "test_aggregated_ioi_eval.json"
    ]
    
    for artifact in test_artifacts:
        if os.path.exists(artifact):
            if os.path.isdir(artifact):
                cleanup_directory(artifact)
            else:
                os.remove(artifact)
    
    try:
        # Step 1: Run notebooks
        print("\n" + "="*80)
        print("STEP 1: Running notebooks")
        print("="*80)
        
        notebooks_success = True
        
        if os.path.exists("example_submission.ipynb"):
            if not run_notebook("example_submission.ipynb"):
                print("ERROR: Failed to run example_submission.ipynb")
                notebooks_success = False
        else:
            print("WARNING: example_submission.ipynb not found")
        
        if os.path.exists("ioi_example_submission.ipynb"):
            if not run_notebook("ioi_example_submission.ipynb"):
                print("ERROR: Failed to run ioi_example_submission.ipynb")
                notebooks_success = False
        else:
            print("WARNING: ioi_example_submission.ipynb not found")
        
        test_results["notebooks_executed"] = notebooks_success
        
        # Step 1.5: Verify submissions
        print("\n" + "="*80)
        print("STEP 1.5: Verifying submissions")
        print("="*80)
        
        verification_success = True
        
        # Verify mock_submission
        if os.path.exists("mock_submission"):
            cmd = [sys.executable, "verify_submission.py", "mock_submission"]
            if not run_command(cmd, "Verifying mock_submission"):
                print("ERROR: Failed to verify mock_submission")
                verification_success = False
        else:
            print("WARNING: mock_submission folder not found")
        
        # Verify ioi_submission  
        if os.path.exists("ioi_submission"):
            cmd = [sys.executable, "verify_submission.py", "ioi_submission"]
            if not run_command(cmd, "Verifying ioi_submission"):
                print("ERROR: Failed to verify ioi_submission")
                verification_success = False
        else:
            print("WARNING: ioi_submission folder not found")
        
        test_results["submission_verification"] = verification_success
        
        # Step 2: Compare JSON files in results folders
        print("\n" + "="*80)
        print("STEP 2: Comparing JSON files in results folders")
        print("="*80)
        
        comparison_success = True
        
        # Check if the notebooks created these folders
        results_folders_exist = (
            os.path.exists("mock_submission_results") and 
            os.path.exists("mock_submission_results_loaded") and
            os.path.exists("ioi_submission_results") and 
            os.path.exists("ioi_submission_results_loaded")
        )
        
        if not results_folders_exist:
            print("INFO: Some results folders don't exist yet - they may be created by notebooks")
            print("  Checking for existing folders:")
            for folder in ["mock_submission_results", "mock_submission_results_loaded",
                          "ioi_submission_results", "ioi_submission_results_loaded"]:
                if os.path.exists(folder):
                    print(f"    ✓ {folder} exists")
                else:
                    print(f"    ✗ {folder} not found")
        
        # Compare mock submission results if they exist
        if os.path.exists("mock_submission_results") and os.path.exists("mock_submission_results_loaded"):
            if not compare_folders("mock_submission_results", "mock_submission_results_loaded"):
                print("ERROR: mock_submission results folders differ")
                comparison_success = False
        
        # Compare IOI submission results if they exist
        if os.path.exists("ioi_submission_results") and os.path.exists("ioi_submission_results_loaded"):
            if not compare_folders("ioi_submission_results", "ioi_submission_results_loaded"):
                print("ERROR: ioi_submission results folders differ")
                comparison_success = False
        
        # Don't fail if folders don't exist, as they may not have been created yet
        test_results["json_comparison"] = comparison_success if results_folders_exist else True
        
        # Step 3: Run aggregate_results.py on all four results folders
        print("\n" + "="*80)
        print("STEP 3: Running aggregate_results.py on all results folders")
        print("="*80)
        
        aggregation_success = True
        
        # Only aggregate folders that exist
        if os.path.exists("mock_submission_results"):
            cmd = [sys.executable, "aggregate_results.py", 
                   "--folder_path", "mock_submission_results",
                   "--output", "test_aggregated_mock.json"]
            if not run_command(cmd, "Aggregating mock_submission_results"):
                aggregation_success = False
        else:
            print("INFO: Skipping mock_submission_results - folder not found")
        
        if os.path.exists("mock_submission_results_loaded"):
            cmd = [sys.executable, "aggregate_results.py",
                   "--folder_path", "mock_submission_results_loaded",
                   "--output", "test_aggregated_mock_loaded.json"]
            if not run_command(cmd, "Aggregating mock_submission_results_loaded"):
                aggregation_success = False
        else:
            print("INFO: Skipping mock_submission_results_loaded - folder not found")
        
        if os.path.exists("ioi_submission_results"):
            cmd = [sys.executable, "aggregate_results.py",
                   "--folder_path", "ioi_submission_results",
                   "--output", "test_aggregated_ioi.json"]
            if not run_command(cmd, "Aggregating ioi_submission_results"):
                aggregation_success = False
        else:
            print("INFO: Skipping ioi_submission_results - folder not found")
        
        if os.path.exists("ioi_submission_results_loaded"):
            cmd = [sys.executable, "aggregate_results.py",
                   "--folder_path", "ioi_submission_results_loaded",
                   "--output", "test_aggregated_ioi_loaded.json"]
            if not run_command(cmd, "Aggregating ioi_submission_results_loaded"):
                aggregation_success = False
        else:
            print("INFO: Skipping ioi_submission_results_loaded - folder not found")
        
        test_results["initial_aggregation"] = aggregation_success
        
        # Compare aggregated results
        print("\nComparing aggregated results...")
        
        # Compare mock submission aggregations if both exist
        if (os.path.exists("test_aggregated_mock.json") and 
            os.path.exists("test_aggregated_mock_loaded.json")):
            if not compare_aggregated_results("test_aggregated_mock.json", 
                                            "test_aggregated_mock_loaded.json",
                                            "Mock Original", "Mock Loaded"):
                print("WARNING: Mock submission aggregated results differ")
        else:
            print("INFO: Skipping mock aggregation comparison - not all files exist")
        
        # Compare IOI submission aggregations if both exist
        if (os.path.exists("test_aggregated_ioi.json") and
            os.path.exists("test_aggregated_ioi_loaded.json")):
            if not compare_aggregated_results("test_aggregated_ioi.json",
                                            "test_aggregated_ioi_loaded.json", 
                                            "IOI Original", "IOI Loaded"):
                print("WARNING: IOI submission aggregated results differ")
        else:
            print("INFO: Skipping IOI aggregation comparison - not all files exist")
        
        # Step 4: Run evaluation scripts with private data flag
        print("\n" + "="*80)
        print("STEP 4: Running evaluation scripts with private data")
        print("="*80)
        
        evaluation_success = True
        
        # Evaluate mock submission
        if os.path.exists("mock_submission"):
            cmd = [sys.executable, "evaluate_submission.py",
                   "--submission_folder", "mock_submission",
                   "--private_data"]
            if not run_command(cmd, "Evaluating mock_submission"):
                evaluation_success = False
        else:
            print("WARNING: mock_submission folder not found")
            evaluation_success = False
        
        # Evaluate IOI submission
        if os.path.exists("ioi_submission"):
            cmd = [sys.executable, "ioi_evaluate_submission.py",
                   "--submission_folder", "ioi_submission",
                   "--private_data"]
            if not run_command(cmd, "Evaluating ioi_submission"):
                evaluation_success = False
        else:
            print("WARNING: ioi_submission folder not found")
            evaluation_success = False
        
        test_results["evaluation"] = evaluation_success
        
        # Step 5: Run aggregate_results.py on submission folders after evaluation
        print("\n" + "="*80)
        print("STEP 5: Aggregating results after evaluation")
        print("="*80)
        
        post_eval_success = True
        
        # Find JSON result files created by evaluation in submission folders
        # evaluate_submission.py and ioi_evaluate_submission.py save results 
        # inside the task subfolders, not in separate result directories
        mock_eval_files = []
        ioi_eval_files = []
        
        if os.path.exists("mock_submission"):
            print("Searching for evaluation results in mock_submission...")
            for root, dirs, files in os.walk("mock_submission"):
                for file in files:
                    if file.endswith('__results.json'):  # Standard result file pattern
                        full_path = os.path.join(root, file)
                        mock_eval_files.append(full_path)
                        print(f"  Found: {os.path.relpath(full_path, 'mock_submission')}")
        
        if os.path.exists("ioi_submission"):
            print("Searching for evaluation results in ioi_submission...")
            for root, dirs, files in os.walk("ioi_submission"):
                for file in files:
                    if file.endswith('__results.json') and 'linear_params' not in file:
                        full_path = os.path.join(root, file)
                        ioi_eval_files.append(full_path)
                        print(f"  Found: {os.path.relpath(full_path, 'ioi_submission')}")
        
        # Create temporary directories for aggregation
        if mock_eval_files:
            print(f"\nAggregating {len(mock_eval_files)} mock submission evaluation results")
            os.makedirs("temp_mock_eval", exist_ok=True)
            for file in mock_eval_files:
                dest_name = os.path.basename(file)
                # If multiple files have same name, prepend parent folder name
                if os.path.exists(os.path.join("temp_mock_eval", dest_name)):
                    parent_name = os.path.basename(os.path.dirname(file))
                    dest_name = f"{parent_name}_{dest_name}"
                shutil.copy2(file, os.path.join("temp_mock_eval", dest_name))
            
            cmd = [sys.executable, "aggregate_results.py",
                   "--folder_path", "temp_mock_eval",
                   "--output", "test_aggregated_mock_eval.json",
                   "--private"]
            if not run_command(cmd, "Aggregating mock evaluation results"):
                post_eval_success = False
            
            cleanup_directory("temp_mock_eval")
        else:
            print("WARNING: No evaluation result files found in mock_submission")
        
        if ioi_eval_files:
            print(f"\nAggregating {len(ioi_eval_files)} IOI submission evaluation results")
            os.makedirs("temp_ioi_eval", exist_ok=True)
            for file in ioi_eval_files:
                dest_name = os.path.basename(file)
                # If multiple files have same name, prepend parent folder name
                if os.path.exists(os.path.join("temp_ioi_eval", dest_name)):
                    parent_name = os.path.basename(os.path.dirname(file))
                    dest_name = f"{parent_name}_{dest_name}"
                shutil.copy2(file, os.path.join("temp_ioi_eval", dest_name))
            
            cmd = [sys.executable, "aggregate_results.py",
                   "--folder_path", "temp_ioi_eval",
                   "--output", "test_aggregated_ioi_eval.json",
                   "--private"]
            if not run_command(cmd, "Aggregating IOI evaluation results"):
                post_eval_success = False
            
            cleanup_directory("temp_ioi_eval")
        else:
            print("WARNING: No evaluation result files found in ioi_submission")
        
        test_results["post_eval_aggregation"] = post_eval_success
        
        # Step 6: Move submissions and run process_all_submission.py
        print("\n" + "="*80)
        print("STEP 6: Testing process_all_submission.py")
        print("="*80)
        
        process_all_success = True
        
        # Create all_submissions directory
        os.makedirs("test_all_submissions", exist_ok=True)
        
        # Copy submissions to test directory
        if os.path.exists("mock_submission"):
            shutil.copytree("mock_submission", "test_all_submissions/mock_submission")
        if os.path.exists("ioi_submission"):
            shutil.copytree("ioi_submission", "test_all_submissions/ioi_submission")
        
        # Run process_all_submission.py
        cmd = [sys.executable, "process_all_submissions.py",
               "--parent_dir", "test_all_submissions",
               "--output_dir", "test_all_results",
               "--private_data"]
        if not run_command(cmd, "Processing all submissions"):
            process_all_success = False
        
        test_results["process_all_submissions"] = process_all_success
        
        # Step 7: Compare all aggregated results
        print("\n" + "="*80)
        print("STEP 7: Comparing all aggregated results")
        print("="*80)
        
        final_comparison_success = True
        
        # List all aggregated results files
        aggregated_files = []
        
        # From individual runs
        for f in ["test_aggregated_mock.json", "test_aggregated_mock_eval.json",
                  "test_aggregated_ioi.json", "test_aggregated_ioi_eval.json"]:
            if os.path.exists(f):
                aggregated_files.append(f)
        
        # From process_all_submissions
        if os.path.exists("test_all_results"):
            for root, dirs, files in os.walk("test_all_results"):
                for file in files:
                    if file == "aggregated_results.json":
                        aggregated_files.append(os.path.join(root, file))
        
        print(f"\nFound {len(aggregated_files)} aggregated results files:")
        for f in aggregated_files:
            print(f"  - {f}")
        
        # Compare relevant pairs
        if os.path.exists("test_aggregated_mock_eval.json") and \
           os.path.exists("test_all_results/mock_submission/aggregated_results.json"):
            if not compare_aggregated_results(
                "test_aggregated_mock_eval.json",
                "test_all_results/mock_submission/aggregated_results.json",
                "Direct mock evaluation", "Process all mock"
            ):
                final_comparison_success = False
        
        if os.path.exists("test_aggregated_ioi_eval.json") and \
           os.path.exists("test_all_results/ioi_submission/aggregated_results.json"):
            if not compare_aggregated_results(
                "test_aggregated_ioi_eval.json",
                "test_all_results/ioi_submission/aggregated_results.json",
                "Direct IOI evaluation", "Process all IOI"
            ):
                final_comparison_success = False
        
        test_results["final_comparison"] = final_comparison_success
        
    except Exception as e:
        print(f"\nERROR: Unexpected error during testing: {e}")
        traceback.print_exc()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in test_results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("="*80)
    print("\nWhat this test validates:")
    print("1. Notebooks create submissions and run evaluations correctly")
    print("2. Submissions pass verification checks (structure, featurizer, token positions)")
    print("3. JSON result files are consistent between training and loading phases")
    print("4. Aggregation script works on various result folder structures")
    print("5. Evaluation scripts correctly process submission folders")
    print("6. Process_all_submissions script handles multiple submissions")
    print("7. All aggregated results are consistent across different evaluation paths")
    print("="*80)
    
    if all_passed:
        print("✓ ALL TESTS PASSED - Pipeline is working correctly")
        return 0
    else:
        print("✗ SOME TESTS FAILED - Check errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())