import wandb
import os
import sys
import argparse
import logging
import torch
from dotenv import load_dotenv

from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# Setup module path for local imports
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

# Importing the necessary modules
from grpo import grpo_iteration, evaluate_policy, GRAD_CLIPPING_NORM
from dataset.countdown_utils import batch_compute_metrics
from dataset.countdown_dataloader import *

POLICY_MODEL_PRECISION = torch.bfloat16
REF_MODEL_PRECISION = torch.bfloat16
RANDOM_SEED = 42
EVALUATION_FREQUENCY = 20


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
        "--dataset-type",
        type=str,
        default="JSON",
        help="Using JSON or HuggingFace dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/data/countdown.json",
        help="The path to the dataset to use for training if using a local JSON.",
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
        "--batch-size", type=int, default=8, help="Logical batch size for training (prompts)."
    )
    parser.add_argument(
        "--micro-batch-size", type=int, default=2, help="Prompts per forward/backward pass."
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=4, help="Micro batches to accumulate before optimiser step."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="The learning rate to use for training.",
    )
    parser.add_argument(
        "--num-outputs", type=int, default=4, help="The number of outputs to generate."
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="Epsilon value for the training."
    )
    parser.add_argument(
        "--beta", type=float, default=0.001, help="Beta value for the training."
    )
    parser.add_argument("--mu", type=int, default=1, help="Mu value for the training.")
    return parser.parse_args()

# ---------- helper to slice a dict‑batch ---------------------------------- #
def slice_batch(batch: Dict, start: int, end: int) -> Dict:
    """Return a shallow copy of *batch* containing rows start:end."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, list):
            out[k] = v[start:end]
        else:  # torch tensor
            out[k] = v[start:end]
    return out


def main():
    # Load environment variables
    load_dotenv()

    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting training with base model: %s", args.base_model)

    # Initialize wandb
    wandb.login(key=os.environ["WANDB_KEY"], relogin=True, force=True)

    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ["WANDB_ENTITY"],
        config={
            "base_model": args.base_model,
            "dataset": args.dataset,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_outputs": args.num_outputs,
            "epsilon": args.epsilon,
            "beta": args.beta,
            "mu": args.mu,
        },
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Determine the model type
    model_name = args.base_model
    model_type = "instruct" if "instruct" in model_name.lower() else "base"

    # Load the dataset
    if args.dataset_type == "JSON":
        dataset_path = args.dataset
        if not os.path.exists(dataset_path):
            logger.error("Dataset path does not exist: %s", dataset_path)
            return
        logger.info("Loading dataset from: %s", dataset_path)
        dataset = Countdown(dataset_path, model_type)  # Ours
    else:
        logger.info("Loading dataset from Hugging Face...")
        dataset = Countdown_HF(model_type)  # HuggingFace
    if dataset is None:
        logger.error("Failed to load dataset from: %s", dataset_path)
        return
    logger.info("Dataset loaded successfully.")
    # Split dataset into train and test sets
    dataset_size = len(dataset)
    test_size = int(0.1 * dataset_size)
    train_size = dataset_size - test_size
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=generator
    )
    logger.info("Dataset split into train and test sets.")

    batch_size = args.batch_size
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    logger.info("DataLoader created with batch size: %d", batch_size)

    test_batch_size = 16
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Load the model and send to GPUs
    logger.info("Loading policy model: %s", model_name)
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=POLICY_MODEL_PRECISION,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    policy_model.to(device)
    logger.info("Policy Model loaded successfully.")
    logger.info("Loading reference model: %s", model_name)
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=REF_MODEL_PRECISION,
    )
    reference_model.eval()
    reference_model.to("cpu")
    logger.info("Reference Model loaded successfully.")

    # Set up the optimizer
    learning_rate = args.learning_rate
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
    logger.info("Optimizer set up with learning rate: %f", learning_rate)

    # Load needed arguments
    G = args.num_outputs
    eps = args.epsilon
    beta = args.beta
    mu = args.mu
    micro_batch_size = args.micro_batch_size
    grad_accum_steps = args.grad_accum_steps
    step_count = 0
    logger.info("Loaded arguments: G=%d, eps=%f, beta=%f, mu=%d, micro_batch_size=%d, grad_accum_steps=%d", G, eps, beta, mu, micro_batch_size, grad_accum_steps)

    # TODO: Revisit is it worth wrapping this in a trainer class: https://huggingface.co/docs/transformers/en/main_classes/trainer

    # Training loop
    logger.info("Starting training...")
    for epoch in range(args.num_epochs):
        logger.info("Epoch %d/%d", epoch + 1, args.num_epochs)
        policy_model.train()
        for big_batch in tqdm(train_dataloader, desc="Training"):
            optimizer.zero_grad()
            num_prompts = len(big_batch["prompt"])
            total_loss = 0
            for start in range(0, num_prompts, micro_batch_size):
                end = min(start + micro_batch_size, num_prompts)
                mini_batch = slice_batch(big_batch, start, end)
                loss = grpo_iteration(
                    query_batch=mini_batch,
                    policy_model=policy_model,
                    reference_model=reference_model,
                    reward_model=batch_compute_metrics,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    G=G,
                    eps=eps,
                    beta=beta,
                    mu=mu,
                    accumulate=True,
                )
                (loss / grad_accum_steps).backward()
                total_loss += loss.item()

                if ((end // micro_batch_size) % grad_accum_steps == 0) or (end == num_prompts):
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=GRAD_CLIPPING_NORM)
                    optimizer.step()
                    optimizer.zero_grad()
                    step_count += 1
                    logger.info("Step %d completed.", step_count)
                    wandb.log({"train_loss": total_loss / grad_accum_steps, "step": step_count})
                    total_loss = 0
                
                if step_count % EVALUATION_FREQUENCY == 0:
                    logger.info("Batch %d/%d completed.", step_count, len(train_dataloader))
                    logger.info("Evaluating model...")
                    policy_model.eval()
                    full_format_rewards, full_equation_rewards, full_total_rewards, full_accuracies = [], [], [], []
                    with torch.no_grad():
                        for test_batch in tqdm(test_dataloader, desc="Evaluating"):
                            format_rewards, equation_rewards, total_rewards, accuracies = evaluate_policy(
                                policy_model=policy_model,
                                tokenizer=tokenizer,
                                reward_model=batch_compute_metrics,
                                test_batch=test_batch,
                            )
                            full_format_rewards.append(format_rewards)
                            full_equation_rewards.append(equation_rewards)
                            full_total_rewards.append(total_rewards)
                            full_accuracies.append(accuracies)
                        full_format_rewards = torch.cat(full_format_rewards)
                        full_equation_rewards = torch.cat(full_equation_rewards)
                        full_total_rewards = torch.cat(full_total_rewards)
                        full_accuracies = torch.cat(full_accuracies)
                        logger.info("Evaluation completed.")
                        logger.info("Mean format reward: %f", full_format_rewards.mean().item())
                        logger.info("Mean equation reward: %f", full_equation_rewards.mean().item())
                        logger.info("Mean total reward: %f", full_total_rewards.mean().item())
                        logger.info("Mean accuracy: %f", full_accuracies.mean().item())
                        wandb.log(
                            {
                                "epoch": epoch + 1,
                                "batch": step_count,
                                "test_mean_format_reward": full_format_rewards.mean().item(),
                                "test_mean_equation_reward": full_equation_rewards.mean().item(),
                                "test_mean_total_reward": full_total_rewards.mean().item(),
                                "test_mean_accuracy": full_accuracies.mean().item(),
                            }
                        )
                    policy_model.train()

        logger.info("Epoch %d completed.", epoch + 1)

    # Save the model
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    policy_model.save_pretrained(output_dir)
    logger.info("Model saved to: %s", output_dir)
    wandb.finish()
    logger.info("Training completed successfully.")


def collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    targets = torch.tensor([item["target"] for item in batch])
    numbers = [torch.tensor(item["numbers"]) for item in batch]
    # Pad the numbers to the same length
    padded_numbers = pad_sequence(
        numbers, batch_first=True, padding_value=0
    )  # Pad with zeros
    return {
        "prompt": prompts,
        "target": targets,
        "numbers": padded_numbers,
    }


if __name__ == "__main__":
    main()
