import re
import json
from random import randint, seed
from typing import List, Tuple, Dict, Optional
import itertools
import torch
import logging

logger = logging.getLogger(__name__)

EVAL_MARGIN = 1e-5


# ---------------------- Dataset Generator ----------------------
def combine_nums(a: int, b: int) -> List[Tuple[int, str]]:
    """
    Given two integers, return all valid results from applying
    arithmetic operations (addition, subtraction, multiplication, division)
    along with their string representations.

    Only performs integer division when it divides evenly.
    """
    results = [(a + b, f"{a} + {b} = {a + b}"), (a * b, f"{a} * {b} = {a * b}")]

    if a <= b:
        results.append((b - a, f"{b} - {a} = {b - a}"))
        if a != 0 and b % a == 0:
            results.append((b // a, f"{b} / {a} = {b // a}"))
    else:
        results.append((a - b, f"{a} - {b} = {a - b}"))
        if b != 0 and a % b == 0:
            results.append((a // b, f"{a} / {b} = {a // b}"))

    return results


def search(
    target: int, nums: List[int], operations: List[str] = []
) -> Optional[List[str]]:
    """
    Recursively searches for a sequence of arithmetic operations on `nums`
    that results in the target value.

    Returns a list of operation strings if a valid solution is found,
    otherwise None.
    """
    if len(nums) == 1:
        return operations if nums[0] == target else None

    for i, j in itertools.combinations(range(len(nums)), 2):
        num1, num2 = nums[i], nums[j]
        remaining_nums = [nums[k] for k in range(len(nums)) if k != i and k != j]

        for result, operation in combine_nums(num1, num2):
            new_nums = remaining_nums + [result]
            new_operations = operations + [operation]
            solution = search(target, new_nums, new_operations)
            if solution:
                return solution

    return None


def gen_dataset(
    num_samples: int,
    num_operands: int = 6,
    max_target: int = 1000,
    min_number: int = 1,
    max_number: int = 100,
    seed_value: int = 42,
    save_path: Optional[str] = None,
) -> List[Dict]:

    seed(seed_value)

    samples = []

    while len(samples) < num_samples:
        # Generate target & numbers
        target = randint(1, max_target)
        numbers = [randint(min_number, max_number) for _ in range(num_operands)]

        # Attempt to find a solution
        solution = search(target, numbers)

        # If we find one, store it and we keep trying till we get our num_samples
        if solution:
            samples.append({"target": target, "numbers": numbers, "solution": solution})

        if len(samples) % 100 == 0:
            logger.info(f"Generated {len(samples)} of {num_samples} samples so far...")

    if save_path:
        with open(save_path, "w") as f:
            json.dump(samples, f, indent=2)

    return samples


# ---------------------- Evaluation Utils ----------------------
def extract_solution(solution_str: str) -> Optional[str]:
    """
    Extract the final arithmetic equation from the model output.
    """
    eq_pattern = r"([0-9+\-*/\(\)\s]+=[0-9+\-*/\(\)\s]+)"

    # Find all possible equations
    matches = re.findall(eq_pattern, solution_str)
    if not matches:
        return None

    # Take the last matched equation (in case there's more than one)
    equation = matches[-1].strip()

    equation = re.sub(r"\\boxed\{(.*?)\}", r"\1", equation)  # remove \boxed{ }
    equation = re.sub(r"[\[\]]", "", equation)  # remove [ and ]

    # strip newlines or extra spaces
    equation = " ".join(equation.split())

    return equation if equation else None


def validate_equation(equation_str: str, available_numbers: List[int]) -> bool:
    """
    Ensures that the left side of the equation (before '=') uses exactly
    the numbers in `available_numbers`
    """
    try:
        # Split at the '=' to isolate the left side
        if "=" not in equation_str:
            return False

        lhs, _ = equation_str.split("=", 1)  # split just once

        # Extract the numbers from the left side
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", lhs)]

        # Compare sorted lists
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except:
        return False


def evaluate_equation(equation_str: str) -> Optional[float]:
    try:
        lhs, _ = equation_str.split("=", 1)  # split just once
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, lhs):
            raise ValueError("Invalid characters in equation.")
        return eval(lhs, {"__builtins__": None}, {})
    except:
        return None


def compute_metrics(
    output: str, query: Dict, format_score: float = 0.1, full_score: float = 1.0
) -> Dict[str, float]:
    """
    Compute two metrics for the countdown solution:
    1) 'reward_score': partial or full points
    2) 'accuracy': 0.0 or 1.0 indicating correctness

    Returns:
        A dict with {'reward_score': float, 'accuracy': float}.
    """

    # Default values
    reward_score = 0.0
    accuracy = 0.0

    target = query["target"]
    numbers = query["numbers"]

    equation = extract_solution(output)

    if equation is not None:
        if validate_equation(equation, numbers):
            result = evaluate_equation(equation)
            if result is not None:
                if abs(result - target) < EVAL_MARGIN:
                    # Correct numeric result
                    reward_score = full_score
                    accuracy = 1.0
                else:
                    # Numeric result but not correct
                    reward_score = format_score
            else:
                # Could not evaluate
                reward_score = format_score
        else:
            # Equation doesn't match the numbers exactly
            reward_score = format_score

    # Finally, return the updated dictionary
    return {"reward_score": reward_score, "accuracy": accuracy}


def batch_compute_metrics(
    outputs: List[List[str]],
    queries: List[Dict],
    format_score: float = 0.1,
    full_score: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch process the outputs and queries to compute rewards and accuracies.

    Args:
        outputs: List of lists of model outputs, should be of shape (batch_size, G).
        queries: List of query dictionaries, should be of length batch_size.
        format_score: Score for partial (format) correctness.
        full_score: Score for full correctness.

    Returns:
        A tensor for rewards and accuracies, each should be of shape (batch_size, G).
    """
    rewards = []
    accuracies = []

    for i, output_group in enumerate(outputs):
        group_rewards = []
        group_accuracies = []

        for output in output_group:
            metrics = compute_metrics(output, queries[i], format_score, full_score)
            group_rewards.append(metrics["reward_score"])
            group_accuracies.append(metrics["accuracy"])

        rewards.append(group_rewards)
        accuracies.append(group_accuracies)
    # Convert to tensors
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    accuracies_tensor = torch.tensor(accuracies, dtype=torch.float32)
    return rewards_tensor, accuracies_tensor
