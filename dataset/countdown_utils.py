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
    This code closely follows the implementation in TinyZero.
    https://github.com/Jiayi-Pan/TinyZero/blob/0349609d618d477ed3a9834d56952c7647a50c2d/verl/utils/reward_score/countdown.py#L7
    """
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str: str, available_numbers: List[int]) -> bool:
    """
    Ensures that the left side of the equation (before '=') uses exactly
    the numbers in `available_numbers`
    """
    available_numbers = [
        num for num in available_numbers if num > 0
    ]  # Filter out zero padding from collate function
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
    output: str, query: Dict, format_score: float = 0.5, equation_score: float = 1.0
) -> Dict[str, float]:
    """
    Compute four metrics for the countdown solution:
    1) 'format_reward': reward for correct format
    2) 'equation_reward': reward for correct equation
    3) 'total_reward': sum of format_reward and equation_reward
    4) 'accuracy': 0.0 or 1.0 indicating correctness

    Args:
        output: The model's generated response string.
        query: Dictionary containing:
            - 'target': The target number to reach
            - 'numbers': List of available numbers to use
        format_score: Reward given for correct format but incorrect equation (default: 0.5)
        equation_score: Reward given for correct equation (default: 1.0)

    Returns:
        Dict[str, float]: Dictionary containing:
            - 'format_reward': float (0.0 or format_score)
            - 'equation_reward': float (0.0 or equation_score)
            - 'total_reward': float (sum of format_reward and equation_reward)
            - 'accuracy': float (0.0 or 1.0)
    """

    # Default values
    total_reward = 0.0
    format_reward = 0.0
    equation_reward = 0.0
    accuracy = 0.0

    target = query["target"]
    numbers = query["numbers"]

    equation = extract_solution(output)

    logger.info(f"Extracted Equation: {equation} | Numbers: {numbers} | Target: {target}")

    if equation is not None:
        format_reward = format_score
        if validate_equation(equation, numbers):
            result = evaluate_equation(equation)
            if result is not None and abs(result - target) < EVAL_MARGIN:
                equation_reward = equation_score
                accuracy = 1.0

    total_reward = format_reward + equation_reward
    # Finally, return the updated dictionary
    return {"format_reward": format_reward, "equation_reward": equation_reward, "total_reward": total_reward, "accuracy": accuracy}


def batch_compute_metrics(
    outputs: List[List[str]],
    queries: Dict,
    format_score: float = 0.5,
    equation_score: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch process the outputs and queries to compute rewards and accuracies.

    Args:
        outputs: List of lists of model outputs, shape (batch_size, G)
            where G is the number of samples per query
        queries: Dictionary containing:
            - 'target': Tensor of target numbers, shape (batch_size,)
            - 'numbers': List of tensors of available numbers, each shape (batch_size,)
        format_score: Reward for correct format but incorrect equation (default: 0.5)
        equation_score: Reward for correct equation (default: 1.0)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Four tensors of shape (batch_size, G):
            - format_rewards: Rewards for correct format
            - equation_rewards: Rewards for correct equations
            - total_rewards: Sum of format and equation rewards
            - accuracies: Binary indicators of correctness
    """
    format_rewards = []
    equation_rewards = []
    total_rewards = []
    accuracies = []
    # Numbers is a list of tensors each of shape (batchsize), combine them into a single tensor
    numbers_tensor = queries["numbers"]

    for i, output_group in enumerate(outputs):
        group_format_rewards = []
        group_equation_rewards = []
        group_total_rewards = []
        group_accuracies = []

        query = {
            "numbers": numbers_tensor[i].tolist(),
            "target": queries["target"][i],
        }
        # TODO: Could revisit for a more efficient implementation
        for output in output_group:
            metrics = compute_metrics(output, query, format_score, equation_score)
            group_format_rewards.append(metrics["format_reward"])
            group_equation_rewards.append(metrics["equation_reward"])
            group_total_rewards.append(metrics["reward_score"])
            group_accuracies.append(metrics["accuracy"])

        format_rewards.append(group_format_rewards)
        equation_rewards.append(group_equation_rewards)
        total_rewards.append(group_total_rewards)
        accuracies.append(group_accuracies)
    # Convert to tensors
    format_rewards_tensor = torch.tensor(format_rewards, dtype=torch.float32)
    equation_rewards_tensor = torch.tensor(equation_rewards, dtype=torch.float32)
    total_rewards_tensor = torch.tensor(total_rewards, dtype=torch.float32)
    accuracies_tensor = torch.tensor(accuracies, dtype=torch.float32)
    return format_rewards_tensor, equation_rewards_tensor, total_rewards_tensor, accuracies_tensor
