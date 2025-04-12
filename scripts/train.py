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
from grpo import grpo_iteration, evaluate_policy
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
        default="/data/countdown.json",
        help="The path to the dataset to use for training.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/output",
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
    # Split dataset into train and test sets
    train_test = dataset.train_test_split(test_size=0.1)
    dataset = train_test["train"]
    test_dataset = train_test["test"]
    logger.info("Dataset split into train and test sets.")

    batch_size = args.batch_size
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logger.info("DataLoader created with batch size: %d", batch_size)

    test_batch_size = 16
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False
    )

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
    reference_model.eval()
    reference_model.to("cpu")
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

    # TODO: Revisit is it worth wrapping this in a trainer class: https://huggingface.co/docs/transformers/en/main_classes/trainer

    # Training loop
    logger.info("Starting training...")
    for epoch in range(args.num_epochs):
        logger.info("Epoch %d/%d", epoch + 1, args.num_epochs)
        model.train()
        batch_iter = 0
        for batch in tqdm(train_dataloader, desc="Training"):
            batch_iter += 1
            # Transform batch numbers and target into list of dictionaries
            # This is slightly hacky, might look at instead reworking the reward model to deal with tensors
            prompt_batch, raw_values_batch = process_batch(batch)
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
            if batch_iter % 10 == 0:
                logger.info("Batch %d/%d completed.", batch_iter, len(train_dataloader))
                logger.info("Evaluating model...")
                full_rewards, full_accuracies = [], []
                for test_batch in tqdm(test_dataloader, desc="Evaluating"):
                    test_batch_prompts, test_batch_raw_values = process_batch(
                        test_batch
                    )
                    rewards, accuracies = evaluate_policy(
                        policy_model=model,
                        tokenizer=tokenizer,
                        reward_model=batch_compute_metrics,
                        test_batch=test_batch_prompts,
                        query_batch_raw=test_batch_raw_values,
                    )
                    full_rewards.append(rewards)
                    full_accuracies.append(accuracies)
                full_rewards = torch.cat(full_rewards)
                full_accuracies = torch.cat(full_accuracies)
                logger.info("Evaluation completed.")
                logger.info("Mean reward: %f", full_rewards.mean().item())
                logger.info("Mean accuracy: %f", full_accuracies.mean().item())
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "batch": batch_iter,
                        "mean_reward": full_rewards.mean().item(),
                        "mean_accuracy": full_accuracies.mean().item(),
                    }
                )

        logger.info("Epoch %d completed.", epoch + 1)

    # Save the model
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    logger.info("Model saved to: %s", output_dir)
    wandb.finish()
    logger.info("Training completed successfully.")


def process_batch(batch):
    batch_numbers = list(map(list, zip(*batch["numbers"])))
    batch_target = batch["target"]
    prompt_batch = batch["prompt"]
    raw_values_batch = [
        {"numbers": numbers, "target": target}
        for numbers, target in zip(batch_numbers, batch_target)
    ]
    return prompt_batch, raw_values_batch


if __name__ == "__main__":
    main()
