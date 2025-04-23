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
    Extract the last expression in the solution between <answer> and </answer> tags.
    """
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str)
    if not matches:
        return None
    # If there are multiple sets of answer tags, take the last matched expression
    expression = matches[-1].strip()
    return expression if expression else None


def validate_equation_pattern(equation_str: str) -> bool:
    """
    Validates the equation string to ensure it contains only valid characters.
    """
    allowed_pattern = r"^[\d+\-*/().\s]+$"
    return bool(re.match(allowed_pattern, equation_str))


def validate_equation_contains_numbers(
    equation_str: str, available_numbers: List[int]
) -> bool:
    """
    Ensures that the equation uses each of the numbers in `available_numbers` once
    """
    available_numbers = [
        num for num in available_numbers if num > 0
    ]  # Filter out zero padding from collate function
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

        # Compare sorted lists
        match = sorted(numbers_in_eq) == sorted(available_numbers)
        return match
    except:
        return False


def validate_equation_correct(
    equation_str: str, numbers: list[int], target: int
) -> bool:
    """
    Validates that the equation contains the right numbers and evaluates to the target.
    """
    if not validate_equation_contains_numbers(equation_str, numbers):
        return False
    # If we want to add a validation score for the right numbers we can break up this function

    result = evaluate_equation(equation_str)

    if result is None:
        return False
    return abs(result - target) <= EVAL_MARGIN


def evaluate_equation(equation_str: str) -> Optional[float]:
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        return eval(equation_str, {"__builtins__": None}, {})
    except:
        return None


def compute_metrics(
    output: str, query: Dict, format_score: float = 0.5, correctness_score: float = 3.0
) -> Dict[str, float]:
    """
    Compute four metrics for the countdown solution:
    1) 'format_reward': partial or full points
    2) 'correctness_reward': partial or full points
    3) 'total_reward': format_reward + correctness_reward
    4) 'accuracy': 0.0 or 1.0 indicating correctness

    Returns:
        A dict with {'format_reward': float, 'correctness_reward': float, 'total_reward': float, 'accuracy': float}.
    """

    # Default values
    format_reward = 0.0
    correctness_reward = 0.0
    total_reward = 0.0
    accuracy = 0.0

    target = query["target"]
    numbers = query["numbers"]

    # equation = extract_solution(output)
    equation = extract_solution(output)

    logger.info(
        f"Extracted Equation: {equation} | Numbers: {numbers} | Target: {target}"
    )

    if equation is not None:
        if "=" in equation:
            # Split the equation into left and right parts
            lhs, rhs = equation.split("=", 1)
            lhs_valid = validate_equation_pattern(lhs)
            rhs_valid = validate_equation_pattern(rhs)
            # Evaluate the left and right sides
            if lhs_valid:
                lhs_correct = validate_equation_correct(lhs, numbers, target)
                if lhs_correct:
                    format_reward = format_score
                    correctness_reward = correctness_score
                    accuracy = 1.0
                else:
                    format_reward = format_score
                    accuracy = 0.0
            # We only check the right side if the left side wasn't correct
            if rhs_valid and not (lhs_valid and lhs_correct):
                rhs_correct = validate_equation_correct(rhs, numbers, target)
                if rhs_correct:
                    format_reward = format_score
                    correctness_reward = correctness_score
                    accuracy = 1.0
                else:
                    format_reward = format_score
                    accuracy = 0.0
        else:
            valid_pattern = validate_equation_pattern(equation)
            if valid_pattern:
                is_correct = validate_equation_correct(equation, numbers, target)
                if is_correct:
                    format_reward = format_score
                    correctness_reward = correctness_score
                    accuracy = 1.0
                else:
                    format_reward = format_score
                    accuracy = 0.0
            else:
                total_reward = 0.0
    total_reward = format_reward + correctness_reward
    logger.info(f"Format Score: {format_reward} | Correctness Score: {correctness_reward} | Total Reward: {total_reward} | Accuracy: {accuracy}")
    return {"format_score": format_reward, "correctness_score": correctness_reward, "total_score": total_reward, "accuracy": accuracy}


def batch_compute_metrics(
    outputs: List[List[str]],
    queries: Dict,
    format_score: float = 0.5,
    correctness_score: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch process the outputs and queries to compute all rewards and accuracies.

    Args:
        outputs: List of lists of model outputs, should be of shape (batch_size, G).
        queries: List of query dictionaries, should be of length batch_size.
        format_score: Score for partial (format) correctness.
        correctness_score: Score for full correctness.

    Returns:
        A tensor for format_rewards, correctness_rewards, total_rewards and accuracies, each should be of shape (batch_size, G).
    """
    format_rewards = []
    correctness_rewards = []
    total_rewards = []
    accuracies = []
    # Numbers is a list of tensors each of shape (batchsize), combine them into a single tensor
    numbers_tensor = queries["numbers"]

    for i, output_group in enumerate(outputs):
        group_rewards = []
        group_accuracies = []

        query = {
            "numbers": numbers_tensor[i].tolist(),
            "target": queries["target"][i],
        }
        # TODO: Could revisit for a more efficient implementation
        for output in output_group:
            metrics = compute_metrics(output, query, format_score, correctness_score)
            format_rewards.append(metrics["format_score"])
            correctness_rewards.append(metrics["correctness_score"])
            total_rewards.append(metrics["total_score"])
            accuracies.append(metrics["accuracy"])

    # Convert to tensors
    format_rewards_tensor = torch.tensor(format_rewards, dtype=torch.bfloat16)
    correctness_rewards_tensor = torch.tensor(correctness_rewards, dtype=torch.bfloat16)
    total_rewards_tensor = torch.tensor(total_rewards, dtype=torch.bfloat16)
    accuracies_tensor = torch.tensor(accuracies, dtype=torch.bfloat16)
    return format_rewards_tensor, correctness_rewards_tensor, total_rewards_tensor, accuracies_tensor
