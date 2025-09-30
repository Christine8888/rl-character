#!/usr/bin/env python3
"""SCRIPT TO FILTER A JSONL OF CODE PROBLEMS AND JUDGE THEIR QUALITY."""

import json
import asyncio
import argparse
import random
from typing import List, Dict, Any, Optional
from threading import Lock
import logging
import sys

sys.path.append("/workspace/rl-character")
from code_generation.api_manager import APIManager
from code_generation.classifier_utils import (setup_classification_folder, save_result_to_file,
                                 save_classification_results, parse_score)
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

logging.basicConfig(level=logging.INFO)

GRADER_PROMPT = """
You are a judge evaluating the quality of coding problems and their test cases. You will be provided with a problem statement and a selection of up to 10 private test cases. Please analyze whether there are issues with the problem that would make it unsuitable for evaluation.

<problem>
{problem_statement}
</problem>

<test_cases>
{test_cases}
</test_cases>

Please carefully assess the problem and the associated test cases. Consider carefully how the test cases match the problem statement. 

Then, determine whether this problem has any of the following issues:

1. **Missing key information**: The problem statement lacks essential information needed to solve the problem completely
2. **Inconsistent formatting**: The private test cases are formatted differently (i.e. string casing, float precision, etc.) from the examples shown in the problem statement
3. **Multiple acceptable solutions conflict**: The problem statement indicates that multiple solutions are possible and that all will be accepted. As the test cases enforce only one input/output pair, this is problematic
4. **Test cases out of range**: The private test cases contain inputs or outputs that fall egregiously outside the constraints explicitly specified in the problem, to the point that a high-quality, general solution perfectly matching the problem statement could still fail the private test cases

Please analyze the problem and test cases, then provide:

1. A detailed explanation of any issues you identify, or confirmation that the problem appears well-formed
2. A numerical grade from 0-100 indicating how likely the problem is to be problematic:
   - 0-20: Problem appears well-formed with no significant issues
   - 21-40: Minor issues that might cause some confusion but are manageable
   - 41-60: Moderate issues that could lead to incorrect solutions or confusion
   - 61-80: Significant issues that would likely cause problems for most solvers
   - 81-100: Major issues that make the problem unsuitable for evaluation

Please provide your explanation in <explanation> tags and your numerical grade in <score> tags.
"""

def format_test_cases(test_cases: List[Dict[str, Any]], max_cases: int = 10) -> str:
    """Format test cases as XML for the prompt."""
    if not test_cases:
        return "No test cases provided."

    # Select up to max_cases randomly if there are more
    selected_cases = test_cases[:max_cases] if len(test_cases) <= max_cases else random.sample(test_cases, max_cases)

    formatted_cases = []
    for i, case in enumerate(selected_cases, 1):
        input_val = case.get('input', '')
        output_val = case.get('output', '')
        test_type = case.get('type', 'unknown')

        formatted_cases.append(f"""<test_case_{i}>
<input>{input_val}</input>
<output>{output_val}</output>
</test_case_{i}>""")

    return '\n'.join(formatted_cases)

def create_problem_grading_prompt(problem_statement: str, test_cases: List[Dict[str, Any]]) -> Prompt:
    """Create grading prompt for problem quality assessment."""
    formatted_test_cases = format_test_cases(test_cases)

    user_content = GRADER_PROMPT.format(
        problem_statement=problem_statement,
        test_cases=formatted_test_cases
    )

    user = ChatMessage(content=user_content, role=MessageRole.user)
    return Prompt(messages=[user])

def save_to_files(result: Dict[str, Any], paths: Dict[str, str], file_lock: Lock, grade: int) -> None:
    """Save results to appropriate files based on grade."""
    # Save incrementally - always save to all_results
    save_result_to_file(result, paths["all_results"], file_lock)

    # Save to specific files based on grade
    if grade >= 60:  # Significant to major issues
        save_result_to_file(result, paths["problematic"], file_lock)
    else:
        save_result_to_file(result, paths["good"], file_lock)

async def filter_problem_quality(
    problems: List[Dict[str, Any]],
    api_manager: APIManager,
    paths: Dict[str, str],
    file_lock: Lock,
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 1.0,
    force_rerun: bool = False
) -> List[Dict[str, Any]]:
    """Filter problems based on quality assessment."""

    # Load existing results to avoid reprocessing
    existing_results = {}
    if not force_rerun and paths["all_results"].exists():
        logging.info(f"Loading existing results from {paths['all_results']}")
        with open(paths["all_results"], 'r') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    if 'problem_id' in result:
                        existing_results[result['problem_id']] = result
                except json.JSONDecodeError:
                    continue
        logging.info(f"Found {len(existing_results)} already processed problems")

    async def process_single_problem(problem_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        problem_id = problem_data.get('problem_id')

        # Skip if already processed
        if not force_rerun and problem_id and problem_id in existing_results:
            logging.debug(f"Skipping already processed problem: {problem_id}")
            return existing_results[problem_id]

        problem_statement = problem_data.get('problem', '')
        test_cases = problem_data.get('test_cases', [])

        # Create prompt
        prompt = create_problem_grading_prompt(problem_statement, test_cases)

        # Get completion
        completion = await api_manager.get_chat_completion(
            prompt=prompt,
            model=model,
            temperature=temperature
        )

        if not completion:
            return None

        # Parse grade and explanation
        grade, explanation = parse_score(completion)

        # Select up to 10 test cases for the output (same selection as used in grading)
        selected_test_cases = test_cases[:10] if len(test_cases) <= 10 else random.sample(test_cases, 10)

        # Create result with only the required fields
        result = {
            "problem": problem_statement,
            "problem_id": problem_id,
            "selected_test_cases": selected_test_cases,
            "grade": grade,
            "explanation": explanation,
            "grader_model": model
        }

        # Save result
        save_to_files(result, paths, file_lock, grade)

        return result

    # Process all problems
    from tqdm.asyncio import tqdm
    tasks = [process_single_problem(problem) for problem in problems]
    results = await tqdm.gather(*tasks, desc="Evaluating problem quality", total=len(tasks))

    # Filter out None results
    results = [r for r in results if r is not None]

    return results

def load_problems_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load problems from JSONL file."""
    problems = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                problems.append(data)
            except json.JSONDecodeError:
                continue
    return problems

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Filter code problems based on quality assessment")
    parser.add_argument("input_file", help="Path to JSONL file with code problems")
    parser.add_argument("--output-folder", "-o", required=True, help="Output folder where all classification files will be saved")
    parser.add_argument("--model", "-m", default="claude-sonnet-4-20250514", help="Model to use for grading")
    parser.add_argument("--temperature", "-t", type=float, default=1.0, help="Temperature for grading")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Maximum concurrent API requests")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--force-rerun", action="store_true", help="Force re-processing of all items, ignoring existing results")
    args = parser.parse_args()

    # Load problems from JSONL
    logging.info(f"Loading problems from: {args.input_file}")
    problems = load_problems_from_jsonl(args.input_file)
    logging.info(f"Loaded {len(problems)} problems")

    if not problems:
        logging.error("No problems to filter!")
        return

    # Initialize API manager
    api_manager = APIManager(
        use_cache=not args.no_cache,
        max_concurrent=args.max_concurrent,
        max_retries=3
    )

    # Setup output files
    file_names = {
        "all_results": "all_results.jsonl",
        "problematic": "problematic.jsonl",
        "good": "good.jsonl"
    }
    paths = setup_classification_folder(args.output_folder, file_names, force_rerun=args.force_rerun)
    file_lock = Lock()

    # Filter problems
    results = await filter_problem_quality(
        problems=problems,
        api_manager=api_manager,
        paths=paths,
        file_lock=file_lock,
        model=args.model,
        temperature=args.temperature,
        force_rerun=args.force_rerun
    )

    # Save final classification summary
    save_classification_results(
        save_name="problem_quality_classifier",
        output_folder=args.output_folder,
        results=results,
        classification_key="grade"
    )

if __name__ == "__main__":
    asyncio.run(main())