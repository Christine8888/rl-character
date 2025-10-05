#!/usr/bin/env python3
"""Script to run private tests on generation results and categorize outcomes."""

import json
import asyncio
import argparse
import random
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from threading import Lock
import logging
from tqdm.asyncio import tqdm

# Add the parent directory to path for imports
sys.path.append("/workspace/rl-character")
from code_generation.formats import GenerationResult, CodeProblem, TestCase
from code_generation.grader import TestExecutionGrader
from code_generation.utils import extract_code, load_generation_results

logging.basicConfig(level=logging.INFO)

def create_output_folder(output_folder: str) -> Dict[str, str]:
    """Create output folder and return file paths."""
    folder_path = Path(output_folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    
    return {
        "pass": folder_path / "pass.jsonl",
        "fail": folder_path / "fail.jsonl", 
        "timeout": folder_path / "timeout.jsonl"
    }

def save_result_to_file(result: Dict[str, Any], file_path: Path, file_lock: Lock) -> None:
    """Save result to JSONL file with thread safety."""
    with file_lock:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

async def test_generation_result(
    generation_result: GenerationResult,
    n_private_tests: int,
    grader: TestExecutionGrader,
    paths: Dict[str, Path],
    file_lock: Lock
) -> Dict[str, Any]:
    """Test a single generation result against private tests."""
    problem = generation_result.problem
    final_code = extract_code(generation_result.final_code)
    
    # Get private test cases (exclude public tests)
    public_test_cases = problem.public_test_cases
    private_tests = [tc for tc in problem.test_cases if tc not in public_test_cases]
    
    if len(private_tests) < n_private_tests:
        logging.warning(f"Insufficient private tests for problem {problem.problem_id}: found {len(private_tests)}, need {n_private_tests}")
        return {
            "problem_id": problem.problem_id,
            "status": "insufficient_private_tests",
            "final_code": final_code,
            "available_private_tests": len(private_tests),
            "required_private_tests": n_private_tests
        }
    
    # Sample n_private_tests from available private tests
    selected_private_tests = random.sample(private_tests, min(n_private_tests, len(private_tests)))
    
    # Run private tests
    try:
        grading_result = await grader.grade_solution(
            problem=problem,
            solution=final_code,
            test_cases=selected_private_tests
        )
        
        # Determine outcome
        if grading_result.success:
            status = "pass"
            file_path = paths["pass"]
        else:
            # Check if it only failed due to timeouts
            timeout_only = all(
                "timed out" in error["error"].lower() 
                for error in grading_result.errors
            )
            
            if timeout_only:
                status = "timeout"
                file_path = paths["timeout"]
            else:
                status = "fail"
                file_path = paths["fail"]
        
        result = {
            "problem_id": problem.problem_id,
            "status": status,
            "final_code": final_code,
            "grading_result": {
                "success": grading_result.success,
                "errors": grading_result.errors,
                "num_private_tests": len(selected_private_tests)
            }
        }
        
        # Save result incrementally
        save_result_to_file(result, file_path, file_lock)
        
        return result
        
    except Exception as e:
        logging.error(f"Error testing problem {problem.problem_id}: {e}")
        result = {
            "problem_id": problem.problem_id,
            "status": "error",
            "final_code": final_code,
            "error": str(e)
        }
        save_result_to_file(result, paths["fail"], file_lock)
        return result

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run private tests on generation results")
    parser.add_argument("input_file", help="Path to JSONL file with generation results")
    parser.add_argument("--output-folder", "-o", required=True, help="Output folder for results")
    parser.add_argument("--n-private-tests", type=int, default=10, help="Number of private tests to sample")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Maximum concurrent test executions")
    parser.add_argument("--timeout", type=float, default=20.0, help="Timeout for code execution in seconds")
    
    args = parser.parse_args()
    
    # Load generation results
    logging.info(f"Loading generation results from: {args.input_file}")
    generations = load_generation_results(args.input_file)
    logging.info(f"Loaded {len(generations)} generation results")
    
    if not generations:
        logging.error("No generation results to process!")
        return
    
    # Create output folder and get file paths
    paths = create_output_folder(args.output_folder)
    file_lock = Lock()
    
    # Initialize grader
    grader = TestExecutionGrader(use_hackable_executor=False, timeout=args.timeout)
    
    # Create semaphore for concurrent execution
    semaphore = asyncio.Semaphore(args.max_concurrent)
    
    async def test_with_semaphore(generation_result):
        async with semaphore:
            return await test_generation_result(
                generation_result=generation_result,
                n_private_tests=args.n_private_tests,
                grader=grader,
                paths=paths,
                file_lock=file_lock
            )
    
    # Process all generation results
    logging.info(f"Testing {len(generations)} generation results with up to {args.max_concurrent} concurrent executions")
    tasks = [test_with_semaphore(gen) for gen in generations]
    
    # Run with progress bar
    results = []
    with tqdm(total=len(tasks), desc="Testing generation results", unit="result") as pbar:
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
            except Exception as e:
                results.append(e)
            pbar.update(1)
    
    # Count results
    status_counts = {"pass": 0, "fail": 0, "timeout": 0, "error": 0, "no_private_tests": 0, "insufficient_private_tests": 0}
    for result in results:
        if isinstance(result, dict) and "status" in result:
            status = result["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
    
    # Print summary
    logging.info("Results summary:")
    for status, count in status_counts.items():
        logging.info(f"  {status}: {count}")
    
    logging.info(f"Results saved to {args.output_folder}")

if __name__ == "__main__":
    asyncio.run(main())