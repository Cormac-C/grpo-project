import os
import sys
import argparse
import logging

# Setup module path for local imports
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from dataset.countdown_utils import gen_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Create a dataset.")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/data/countdown.json",
        help="The directory to save the trained model.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="The number of outputs to generate.",
    )
    parser.add_argument(
        "--num-operands",
        type=int,
        default=6,
        help="The number of operands to use for each sample.",
    )
    parser.add_argument(
        "--max-target", type=int, default=1000, help="The maximum target value."
    )
    parser.add_argument(
        "--min-number", type=int, default=1, help="The minimum number to use."
    )
    parser.add_argument(
        "--max-number", type=int, default=100, help="The maximum number to use."
    )
    parser.add_argument(
        "--seed-value",
        type=int,
        default=42,
        help="The seed value for random number generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Creating dataset...")

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "countdown.json")
    logger.info(f"Saving dataset to {save_path}")

    # Generate the dataset
    dataset = gen_dataset(
        num_samples=args.num_samples,
        num_operands=args.num_operands,
        max_target=args.max_target,
        min_number=args.min_number,
        max_number=args.max_number,
        seed_value=args.seed_value,
        save_path=save_path,
    )

    logger.info(f"Dataset saved to {save_path}")
