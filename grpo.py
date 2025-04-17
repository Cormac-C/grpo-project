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
GRAD_CLIPPING_NORM = 10.0

LOWER_PRECISION = torch.bfloat16

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
    output_ids, generated_ids, outputs = sample_outputs(
        policy_model,
        tokenizer,
        query_batch["prompt"],
        G,
        max_new_tokens,
        temperature,
    )

    padding_mask = generated_ids.ne(tokenizer.pad_token_id)
    logger.info(f"Num non-zero tokens in padding mask: {padding_mask.sum()}")

    clear_cache()

    # Compute rewards and accuracies for each output
    rewards, accuracies = reward_model(outputs, query_batch)
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
    logger.info(f"Advantages: {advantages}")
    #  Compute log probabilities for reference model and pre-update policy, no gradients here
    with torch.no_grad():
        if mu > 1:
            old_log_probs = compute_log_probs(
                policy=policy_model,
                tokenizer=tokenizer,
                output_ids=output_ids,
                generated_ids=generated_ids,
                temperature=temperature,
            )
        else:
            # If mu=1, we don't need to compute old log probs, these are dummy values
            old_log_probs = torch.zeros_like(generated_ids, dtype=torch.bfloat16)
        # Swap the policy model and reference model
        gpu_device = policy_model.device
        reference_model.to(gpu_device)

        reference_model_log_probs = compute_log_probs(
            policy=reference_model,
            tokenizer=tokenizer,
            output_ids=output_ids,
            generated_ids=generated_ids,
            temperature=temperature,
        )
        # Swap back the models
        reference_model.to("cpu")

    policy_model.train()
    for i in range(mu):
        logger.info(f"Update iteration: {i+1}/{mu}")
        optimizer.zero_grad()

        # Compute log probabilities for the current policy model, this needs gradients
        model_log_probs = compute_log_probs(
            policy=policy_model,
            tokenizer=tokenizer,
            output_ids=output_ids,
            generated_ids=generated_ids,
            temperature=temperature,
        )
        # Compute GRPO objective
        objective = calculate_grpo_objective(
            model_log_probs=model_log_probs,
            old_model_log_probs=old_log_probs,
            ref_model_log_probs=reference_model_log_probs,
            padding_mask=padding_mask,
            advantages=advantages,
            mu=mu,
            eps=eps,
            beta=beta,
        )

        # Compute gradient of the GRPO objective
        loss = -objective
        # Take the mean loss across the batch
        loss = torch.mean(loss)
        logger.info(f"Loss: {loss.item()}")
        wandb.log({"train_loss": loss.item()})
        loss.backward()

        grad_norm = find_grad_norm(policy_model)
        logger.info(f"Gradient norm (no clipping): {grad_norm}")
        wandb.log({"train_grad_norm_no_clip": grad_norm})

        clip_grad_norm_(policy_model.parameters(), max_norm=GRAD_CLIPPING_NORM)

        grad_norm = find_grad_norm(policy_model)
        logger.info(f"Gradient norm (clipping): {grad_norm}")
        wandb.log({"train_grad_norm_w_clip": grad_norm})

        # Update the policy
        optimizer.step()
    clear_cache()
    return policy_model


def find_grad_norm(model: PreTrainedModel) -> float:
    """
    Find the gradient norm of the model parameters.
    Args:
        model: The model to find the gradient norm for.
    Returns:
        The gradient norm of the model parameters.
    """
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
    """
    Clear the GPU cache and collect garbage to free up GPU memory.
    """
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
        max_new_tokens: The maximum number of new tokens to generate.
        temperature: The temperature for sampling.

    Returns:
        Output ids, a tensor of shape (batch_size, G, max_length).
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

    # Reshape the generated IDs and output IDs to have shape (batch_size, G, max_length)
    output_ids_reshaped = output_ids.reshape(batch_size, G, -1)
    generated_ids_reshaped = generated_ids.view(batch_size, G, -1)

    assert (
        output_ids_reshaped.shape[0] == tokenized_queries["input_ids"].shape[0]
        and generated_ids_reshaped.shape[0] == tokenized_queries["input_ids"].shape[0]
    ), "Output IDs and Generated IDs must have the same batch size as input IDs"
    assert (
        output_ids_reshaped.shape[1] == G and generated_ids_reshaped.shape[1] == G
    ), "Output IDs and Generated IDs must have the same number of outputs as G"

    return output_ids_reshaped, generated_ids_reshaped, responses_reshaped


def calculate_grpo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """
    Calculate advantage for each output.
    Args:
        rewards: The rewards for each output, have shape (batch_size, G).
    Returns:
        A tensor of advantages for each output, shape (batch_size, G).
    """

    means = torch.mean(rewards, dim=1, keepdim=True)
    stds = torch.std(rewards, dim=1, keepdim=True)
    # Normalize the advantage by group mean and std
    advantages = (rewards - means) / (stds + STABILITY_CONST)
    return advantages


def compute_log_probs(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    temperature: float = TEMPERATURE,
) -> torch.Tensor:
    """
    Calculate log probabilities for the generated IDs for a given policy.
    Args:
        policy: The current policy.
        tokenizer: The tokenizer for the policy model.
        output_ids: The output IDs, should be of shape (batch_size, G, max_length).
        generated_ids: The generated IDs, should be of shape (batch_size, G, gen_length).
        temperature: The temperature for sampling.
    Returns:
        Log probabilities for the generated IDs, should be of shape (batch_size, G, gen_length).
    """
    device = policy.device

    batch_size, G, max_length = output_ids.shape
    generated_length = generated_ids.shape[-1]
    query_length = max_length - generated_length

    flattened_output_ids = output_ids.reshape(batch_size * G, max_length).to(device)
    attention_mask = flattened_output_ids.ne(tokenizer.pad_token_id).long().to(device)

    outputs = policy(input_ids=flattened_output_ids, attention_mask=attention_mask)

    # Scale logits by temperature
    logits = outputs.logits / temperature
    logits = logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)

    # Shift output IDs to the left to match the logits
    shifted_output_ids = flattened_output_ids[:, 1:]

    log_probs = log_probs.gather(2, shifted_output_ids.unsqueeze(-1)).squeeze(-1)

    # Remove log probs for query tokens
    generated_log_probs = log_probs[:, (query_length - 1) :]

    # Mask the log probs for padding tokens
    padding_mask = (
        shifted_output_ids[:, (query_length - 1) :].ne(tokenizer.pad_token_id).long()
    )
    generated_log_probs = generated_log_probs * padding_mask

    return generated_log_probs.reshape(batch_size, G, -1)


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
    # torch.expm1 is more stable than torch.exp - 1 ref: https://pytorch.org/docs/stable/special.html#torch.special.expm1
    kl_div = torch.expm1(log_quotient) - log_quotient
    return kl_div


def calculate_grpo_objective(
    model_log_probs: torch.Tensor,
    old_model_log_probs: torch.Tensor,
    ref_model_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    padding_mask: torch.Tensor = None,
    mu: int = 1,
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
        mu: The number of policy updates per iteration.
        eps: The clipping width in GRPO objective.
        beta: The influence of KL div term.
    Returns:
        The GRPO objective value, of shape (batch_size).
    """
    # If mu=1, just set prob_ratios to 1
    logger.info(f"Model log probs: {model_log_probs}")
    logger.info(f"Old model log probs: {old_model_log_probs}")
    if mu == 1:
        prob_ratios = torch.ones_like(model_log_probs)
    else:
        prob_ratios = torch.exp(model_log_probs - old_model_log_probs)
    logger.info(f"Prob ratios: {prob_ratios}")
    clipped_ratios = torch.clamp(prob_ratios, 1 - eps, 1 + eps)
    assert (
        prob_ratios.shape == clipped_ratios.shape
    ), "Prob ratios and clipped ratios must have the same shape"
    # Expand the advantages to match the dimensions of prob_ratios
    advantages = advantages.unsqueeze(-1)
    device = prob_ratios.device
    advantages = advantages.to(device)
    # TODO: check if the advantage is being distributed to padding?
    # Maybe add a mask to the advantages, setting the padding to 0
    min_product = torch.min(prob_ratios * advantages, clipped_ratios * advantages)

    expected_advantage = model_log_probs * min_product
    # Apply the padding mask to the expected advantage
    if padding_mask is not None:
        expected_advantage = expected_advantage * padding_mask

    assert (
        min_product.shape[0] == model_log_probs.shape[0]
        and min_product.shape[1] == model_log_probs.shape[1]
    ), "Min product must have the same batch size and number of outputs as model log probs"

    kl_div = kl_div_estimator(model_log_probs, ref_model_log_probs)

    logger.info(f"Mean KL Div: {torch.mean(kl_div).item()}")
    wandb.log({"Mean KL Div": torch.mean(kl_div).item()})
    objective = expected_advantage - beta * kl_div
    logger.info(f"Objective before mean: {objective}")
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
        _, _, outputs = sample_outputs(
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
