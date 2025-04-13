import torch
import wandb
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import logging
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase
from typing import Callable, Dict, List
import gc

MAX_NEW_TOKENS = 1024
TEMPERATURE = 1.0
STABILITY_CONST = 1e-4
GRAD_CLIPPING_NORM = 1.0

logger = logging.getLogger(__name__)


def grpo_iteration(
    query_batch: Dict,
    policy_model: PreTrainedModel,
    reference_model: PreTrainedModel,
    reward_model: Callable,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    G: int,
    eps: float,
    beta: float,
    mu: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> PreTrainedModel:
    """
    Perform one iteration of the GRPO algorithm.

    Args:
        query_batch_prompts: Batch of queries.
        query_batch_raw: Raw batch of inputs and targets for queries.
        policy_model: The current policy model.
        reward_model: The reward model.
        tokenizer: The tokenizer for the policy model.
        optimizer: The optimizer for the policy model.
        G: The number of outputs to sample.
        eps: The clipping width in GRPO objective.
        beta: The influence of KL div term.
        mu: The number of policy updates per iteration.

    Returns:
        The updated policy.
    """
    # Sample G outputs from the policy for each query in query_batch
    outputs_ids, outputs = sample_outputs(
        policy_model, tokenizer, query_batch["prompt"], G, max_new_tokens, temperature
    )

    logger.debug(f"Outputs: {outputs}")

    clear_cache()

    # Compute rewards and accuracies for each output
    rewards, accuracies = reward_model(outputs, query_batch)
    logger.debug(f"Rewards: {rewards}")
    logger.info(f"Average Reward: {rewards.mean()}")
    logger.info(f"Average Accuracy: {accuracies.mean()}")
    wandb.log(
        {
            "train_mean_reward": rewards.mean().item(),
            "train_mean_accuracy": accuracies.mean().item(),
        }
    )

    # Compute token-level advantage for each token in each output
    advantages = calculate_grpo_advantage(rewards)
    logger.debug(f"Advantages: {advantages}")

    #  Compute log probabilities for reference model and pre-update policy, no gradients here
    with torch.no_grad():
        old_log_probs = compute_log_probs(
            policy=policy_model,
            tokenizer=tokenizer,
            query_batch=query_batch["prompt"],
            generated_ids=outputs_ids,
        )
        # Swap the policy model and reference model
        gpu_device = policy_model.device
        policy_model.to("cpu")
        reference_model.to(gpu_device)

        reference_model_log_probs = compute_log_probs(
            policy=reference_model,
            tokenizer=tokenizer,
            query_batch=query_batch["prompt"],
            generated_ids=outputs_ids,
        )
        # Swap back the models
        reference_model.to("cpu")
        policy_model.to(gpu_device)
    clear_cache()
    for i in range(mu):
        logger.info(f"Update iteration: {i+1}/{mu}")
        optimizer.zero_grad()
        # Compute log probabilities for the current policy model, this needs gradients
        model_log_probs = compute_log_probs(
            policy=policy_model,
            tokenizer=tokenizer,
            query_batch=query_batch["prompt"],
            generated_ids=outputs_ids,
        )
        # Compute GRPO objective
        objective = calculate_grpo_objective(
            model_log_probs=model_log_probs,
            old_model_log_probs=old_log_probs,
            ref_model_log_probs=reference_model_log_probs,
            advantages=advantages,
            eps=eps,
            beta=beta,
        )

        # Compute gradient of the GRPO objective
        loss = -objective
        # Take the mean loss across the batch
        loss = torch.mean(loss)
        logger.info(f"Loss: {loss.item()}")
        wandb.log({"train_loss": loss.item()})
        if not torch.isnan(loss) and torch.abs(loss) > STABILITY_CONST:
            loss.backward()

            clip_grad_norm_(policy_model.parameters(), max_norm=GRAD_CLIPPING_NORM)

            grad_norm = find_grad_norm(policy_model)
            logger.info(f"Gradient norm: {grad_norm}")
            wandb.log({"train_grad_norm": grad_norm})

            # Update the policy
            optimizer.step()
    clear_cache()
    return policy_model


def find_grad_norm(model: PreTrainedModel) -> float:
    norm = torch.norm(
        torch.stack(
            [
                torch.norm(p.grad.detach(), 2)
                for p in model.parameters()
                if p.grad is not None
            ]
        ),
        2,
    )
    return norm.item()


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


def sample_outputs(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    query_batch: List[str],
    G: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> tuple[torch.Tensor, List[str]]:
    """
    Sample G outputs from the policy for each query in query_batch. Doesn't track gradients or log probs.

    Args:
        policy: The current policy.
        tokenizer: The tokenizer for the policy model.
        query_batch: Batch of queries, a list of strings of length batch_size.
        G: The number of outputs to sample.

    Returns:
        Generated ids, a tensor of shape (batch_size, G, max_length).
        Generated text, a list of strings of shape (batch_size, G).
    """
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        num_return_sequences=G,
        temperature=temperature,
        return_dict_in_generate=True,
        output_scores=False,
    )

    tokenized_queries = tokenizer(
        query_batch, return_tensors="pt", padding=True, truncation=True
    )
    tokenized_queries = {
        key: value.to(policy.device) for key, value in tokenized_queries.items()
    }

    with torch.no_grad():
        output = policy.generate(**tokenized_queries, generation_config=gen_config)
    output_ids = output.sequences

    generated_ids = output_ids[:, tokenized_queries["input_ids"].shape[1] :]

    batch_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # Reshape the responses to have shape (batch_size, G), each item contains the generated text and log probs
    batch_size = len(query_batch)
    responses_reshaped = [
        batch_responses[i * G : (i + 1) * G] for i in range(batch_size)
    ]

    # Reshape the generated IDs to have shape (batch_size, G, max_length)
    generated_ids_reshaped = generated_ids.view(batch_size, G, -1)
    assert (
        generated_ids_reshaped.shape[0] == tokenized_queries["input_ids"].shape[0]
    ), "Generated IDs must have the same batch size as input IDs"
    assert (
        generated_ids_reshaped.shape[1] == G
    ), "Generated IDs must have the same number of outputs as G"

    return generated_ids_reshaped, responses_reshaped


def calculate_grpo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """
    Calculate advantage for each output.
    Args:
        rewards: The rewards for each output, have shape (batch_size, G).
    Returns:
        A tensor of advantages for each output, shape (batch_size, G).
    """

    advantages = rewards.clone()
    for i in range(rewards.shape[0]):
        group_mean = torch.mean(rewards[i])
        group_std = torch.std(rewards[i])
        # Normalize the advantage by group mean and std
        advantages[i] = (rewards[i] - group_mean) / (group_std + STABILITY_CONST)
    return advantages


def compute_log_probs(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    query_batch: List[str],
    generated_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate log probabilities for the generated IDs for a given policy.
    Args:
        policy: The current policy.
        tokenizer: The tokenizer for the policy model.
        query_batch: Batch of queries, should be of shape (batch_size).
        generated_ids: The generated IDs, should be of shape (batch_size, G, max_length).
    Returns:
        Log probabilities for the generated IDs, should be of shape (batch_size, G, max_length).

    """
    tokenized_queries = tokenizer(
        query_batch, return_tensors="pt", padding=True, truncation=True
    )
    query_ids = tokenized_queries["input_ids"]

    # Expand the query IDs to match the shape of generated IDs
    query_ids = query_ids.unsqueeze(1).expand(-1, generated_ids.shape[1], -1)
    assert (
        query_ids.shape[0] == generated_ids.shape[0]
    ), "Query IDs and generated IDs must have the same batch size"
    assert (
        query_ids.shape[1] == generated_ids.shape[1]
    ), "Query IDs and generated IDs must have the same number of outputs"

    device = policy.device
    query_ids = query_ids.to(device)
    generated_ids = generated_ids.to(device)
    input_ids = torch.cat((query_ids, generated_ids), dim=2)
    # Reshape the input IDs to have shape (batch_size * G, max_length)
    input_ids = input_ids.view(-1, input_ids.shape[-1])
    assert (
        input_ids.shape[0] == query_ids.shape[0] * query_ids.shape[1]
    ), "Input IDs must have the same batch size as query IDs and generated IDs"

    input_ids = input_ids.to(policy.device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to(policy.device)

    # Run forward pass to get the logits
    outputs = policy(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    generated_logits = logits[:, query_ids.shape[2] :]

    # Calculate log probabilities
    log_probs = F.log_softmax(generated_logits, dim=-1)
    generated_ids = generated_ids.view(-1, generated_ids.shape[-1])
    log_probs = log_probs.gather(2, generated_ids.unsqueeze(-1)).squeeze(-1)
    batch_size = len(query_batch)
    log_probs = log_probs.view(batch_size, -1, generated_ids.shape[-1])

    assert (
        log_probs.shape[0] * log_probs.shape[1] == generated_ids.shape[0]
    ), "Log probabilities must have the same number of queries and outputs as generated IDs"
    assert (
        log_probs.shape[2] == generated_ids.shape[1]
    ), "Log probabilities must have the same length as generated IDs"

    return log_probs


def kl_div_estimator(
    model_log_probs: torch.Tensor, ref_model_log_probs: torch.Tensor
) -> torch.Tensor:
    """
    Estimate the KL divergence between the model and reference model.
    Args:
        model_log_probs: Log probabilities from the model.
        ref_model_log_probs: Log probabilities from the reference model.
    Returns:
        A tensor of KL divergence values. Calculated with unbiased estimator from http://joschu.net/blog/kl-approx.html (cited in GRPO paper)
    """
    log_quotient = ref_model_log_probs - model_log_probs
    kl_div = torch.exp(log_quotient) - log_quotient - 1
    return kl_div


def calculate_grpo_objective(
    model_log_probs: torch.Tensor,
    old_model_log_probs: torch.Tensor,
    ref_model_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    eps: float = 0.1,
    beta: float = 0.005,
) -> torch.Tensor:
    """
    Calculate the GRPO objective.
    Args:
        model_log_probs: Log probabilities from the model.
        old_model_log_probs: Log probabilities from the old model.
        ref_model_log_probs: Log probabilities from the reference model.
        advantages: The advantages for each token in each output.
        eps: The clipping width in GRPO objective.
        beta: The influence of KL div term.
    Returns:
        The GRPO objective value, of shape (batch_size).
    """
    prob_ratios = torch.exp(model_log_probs - old_model_log_probs)
    clipped_ratios = torch.clamp(prob_ratios, 1 - eps, 1 + eps)
    assert (
        prob_ratios.shape == clipped_ratios.shape
    ), "Prob ratios and clipped ratios must have the same shape"
    # Expand the advantages to match the dimensions of prob_ratios
    advantages = advantages.unsqueeze(-1)
    device = prob_ratios.device
    advantages = advantages.to(device)
    min_product = torch.min(prob_ratios * advantages, clipped_ratios * advantages)
    assert (
        min_product.shape[0] == model_log_probs.shape[0]
        and min_product.shape[1] == model_log_probs.shape[1]
    ), "Min product must have the same batch size and number of outputs as model log probs"

    kl_div = kl_div_estimator(model_log_probs, ref_model_log_probs)

    objective = min_product - beta * kl_div
    # Take mean across all tokens and all outputs
    objective = torch.mean(objective, dim=[1, 2])
    assert (
        objective.shape[0] == model_log_probs.shape[0]
    ), "Objective must have the same batch size as model log probs"
    return objective


def evaluate_policy(
    policy_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    reward_model: Callable,
    test_batch: Dict,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
):
    """
    Evaluate the policy model on a test batch.
    Args:
        policy_model: The current policy model.
        tokenizer: The tokenizer for the policy model.
        reward_model: The reward model.
        test_batch: Batch of queries, should be of shape (batch_size).
        max_new_tokens: The maximum number of new tokens to generate.
        temperature: The temperature for sampling.
    Returns:
        Generated text, a list of strings of shape (batch_size).
    """
    policy_model.eval()
    with torch.no_grad():
        _, outputs = sample_outputs(
            policy_model,
            tokenizer,
            test_batch["prompt"],
            G=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        logger.debug(f"Evaluate Outputs: {outputs}")

    clear_cache()

    # Compute rewards and accuracies for each output
    rewards, accuracies = reward_model(outputs, test_batch)

    return rewards, accuracies
