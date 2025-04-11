import wandb
import os
import sys
import argparse
import logging
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# Setup module path for local imports
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

# Importing the necessary modules
from grpo import grpo_iteration
from dataset.countdown_utils import batch_compute_metrics


# Read arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="The base model to use for training.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/countdown.json",
        help="The path to the dataset to use for training.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="The directory to save the trained model.",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=1, help="The number of epochs to train for."
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="The batch size to use for training."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="The learning rate to use for training.",
    )
    parser.add_argument(
        "--num-ouputs", type=int, default=5, help="The number of outputs to generate."
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="Epsilon value for the training."
    )
    parser.add_argument(
        "--beta", type=float, default=0.05, help="Beta value for the training."
    )
    parser.add_argument(
        "--mu", type=float, default=1, help="Mu value for the training."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting training with base model: %s", args.base_model)

    # Initialize wandb
    os.environ["WANDB_API_KEY"] = "your_wandb_api_key"
    os.environ["WANDB_PROJECT"] = "your_project_name"
    wandb.init(project="your_project_name", entity="your_entity_name")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Load the dataset
    # Note: need to create the dataset elsewhere
    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        logger.error("Dataset path does not exist: %s", dataset_path)
        return
    logger.info("Loading dataset from: %s", dataset_path)
    # Load your dataset here
    dataset = load_dataset(dataset_path)
    if dataset is None:
        logger.error("Failed to load dataset from: %s", dataset_path)
        return
    logger.info("Dataset loaded successfully.")
    batch_size = args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logger.info("DataLoader created with batch size: %d", batch_size)

    # Load the model and send to GPUs
    model_name = args.base_model
    logger.info("Loading policy model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.to(device)
    logger.info("Policy Model loaded successfully.")
    logger.info("Loading reference model: %s", model_name)
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    reference_model.to(device)
    logger.info("Reference Model loaded successfully.")

    # Set up the optimizer
    learning_rate = args.learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    logger.info("Optimizer set up with learning rate: %f", learning_rate)

    # Load needed arguments
    G = args.num_outputs
    eps = args.epsilon
    beta = args.beta
    mu = args.mu
    logger.info("Loaded arguments: G=%d, eps=%f, beta=%f, mu=%f", G, eps, beta, mu)

    # Training loop
    logger.info("Starting training...")
    for epoch in range(args.num_epochs):
        logger.info("Epoch %d/%d", epoch + 1, args.num_epochs)
        model.train()
        for batch in tqdm(dataloader, desc="Training"):
            # Transform batch numbers and target into list of dictionaries
            # This is slightly hacky, might look at instead reworking the reward model to deal with tensors
            batch_numbers = list(map(list, zip(*batch["numbers"])))
            batch_target = batch["target"]
            prompt_batch = batch["prompt"]
            raw_values_batch = [
                {"numbers": numbers, "target": target}
                for numbers, target in zip(batch_numbers, batch_target)
            ]
            model = grpo_iteration(
                query_batch_prompts=prompt_batch,
                query_batch_raw=raw_values_batch,
                policy_model=model,
                reference_model=model,
                reward_model=batch_compute_metrics,
                tokenizer=tokenizer,
                optimizer=optimizer,
                G=G,
                eps=eps,
                beta=beta,
                mu=mu,
            )
        logger.info("Epoch %d completed.", epoch + 1)
