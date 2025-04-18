import torch
import wandb
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import logging
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase
from typing import Callable, Dict, List, Union
import gc

SampleDict = Dict[str, Union[torch.Tensor, List[List[str]]]]
MAX_NEW_TOKENS = 1024
TEMPERATURE = 1.0
STABILITY_CONST = 1e-4
GRAD_CLIPPING_NORM = 10.0
IGNORE_INDEX = -100

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
    model_outputs = sample_outputs(
        policy_model,
        tokenizer,
        query_batch["prompt"],
        G,
        max_new_tokens,
        temperature,
    )

    all_responses = model_outputs["all_responses"]

    clear_cache()

    # Compute rewards and accuracies for each output
    rewards, accuracies = reward_model(all_responses, query_batch)
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
    wandb.log(f"Mean Advantages: {advantages.mean()}")
    #  Compute log probabilities for reference model and pre-update policy, no gradients here
    with torch.no_grad():
        if mu > 1:
            old_log_probs = compute_log_probs(
                policy=policy_model,
                inputs=model_outputs,
                temperature=temperature,
            )
        else:
            # If mu=1, we don't need to compute old log probs
            old_log_probs = torch.zeros_like(model_outputs["labels"], dtype=torch.bfloat16)
        # Swap the policy model and reference model
        gpu_device = policy_model.device
        policy_model.to("cpu")
        reference_model.to(gpu_device)

        reference_model_log_probs = compute_log_probs(
            policy=reference_model,
            inputs=model_outputs,  # The dictionary from sample_outputs
            temperature=temperature,
        )
        # Swap back the models
        reference_model.to("cpu")
        policy_model.to(gpu_device)

    policy_model.train()
    for i in range(mu):
        logger.info(f"Update iteration: {i+1}/{mu}")
        optimizer.zero_grad()

        # Compute log probabilities for the current policy model, this needs gradients
        model_log_probs = compute_log_probs(
            policy=policy_model,
            inputs=model_outputs,
            temperature=temperature,
        )
        # Compute GRPO objective
        objective = calculate_grpo_objective(
            model_log_probs=model_log_probs,
            old_model_log_probs=old_log_probs,
            ref_model_log_probs=reference_model_log_probs,
            advantages=advantages,
            mu=mu,
            eps=eps,
            beta=beta,
        )

        # Compute gradient of the GRPO objective
        loss = -objective
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
) -> SampleDict:
    """
    Sample G outputs from the policy for each query in query_batch. Doesn't track gradients or log probs.

    Args:
        policy: The current policy.
        tokenizer: The tokenizer for the policy model.
        query_batch: Batch of queries, a list of strings of length batch_size.
        G: The number of outputs to sample.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Temperature for sampling.

    Returns:        
        labels: A tensor of shape (batch_size*G,  max_length) with token ids for the model responses, -100 on query/input and padding tokens
        all_responses: A list of lists of strings of shape (batch_size, G) containing the generated text.
        complete_sequences_ids: A tensor of shape (batch_size*G, max_length) containing the complete sequences (including input)
        attention_mask: Attention Mask over the complete sequence, ignoring padding (batch_size*G, max_length)
    """
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        num_return_sequences=G,
        temperature=temperature,
        return_dict_in_generate=True,
        output_scores=False,
    )

    # Tokenize queries and move to device
    tokenized_queries = tokenizer(
        query_batch, return_tensors="pt", padding=True, truncation=True
    )
    tokenized_queries = {
        key: value.to(policy.device) for key, value in tokenized_queries.items()
    }

    # Generate outputs
    with torch.no_grad():
        output = policy.generate(**tokenized_queries, generation_config=gen_config)

    # Get the full output sequences (including input)
    complete_sequences_ids = output.sequences

    query_length  = tokenized_queries["input_ids"].shape[1]
    responses_ids = complete_sequences_ids[:, query_length:]

    # Put an IGNORE_INDEX over query tokens and padding tokens
    label_ids = complete_sequences_ids.clone()
    label_ids[:, :query_length] = IGNORE_INDEX
    label_ids[label_ids == tokenizer.pad_token_id] = IGNORE_INDEX

    attention_mask = complete_sequences_ids.ne(tokenizer.pad_token_id)

    # Reshape the responses to have shape (batch_size, G), each item contains the generated text and log probs
    batch_responses = tokenizer.batch_decode(responses_ids, skip_special_tokens=True)
    batch_size = len(query_batch)
    all_responses = [
        batch_responses[i * G : (i + 1) * G] for i in range(batch_size)
    ]

    return {
        "labels":                  label_ids,
        "all_responses":           all_responses,
        "complete_sequences_ids":  complete_sequences_ids,
        "attention_mask":          attention_mask
    }


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
    advantages = (rewards - means) / (stds + STABILITY_CONST)
    return advantages


def compute_log_probs(
    policy: PreTrainedModel,
    inputs: SampleDict,
    temperature: float = TEMPERATURE,
) -> torch.Tensor:
    """
    The following code closely follows the implementation found at:
    https://github.com/McGill-NLP/nano-aha-moment/blob/0ce62ece2681eefad8041b9336fc7d2cd1a3687a/utils.py#L109


    Compute token-level log probabilities for a sequence using the policy model.

    This function processes input sequences through the policy model and calculates
    the log probabilities for each token position, taking into account:
    - Temperature scaling of logits
    - Causal language modeling requirements
    - Valid token positions (ignoring padding and query tokens)

    Args:
        policy: The language model to use for computing log probabilities
        inputs: Dictionary containing:
            - complete_sequences_ids: Token IDs tensor of shape [batch_size * G, seq_len]
            - attention_mask: Attention mask tensor of shape [batch_size * G, seq_len]
            - labels: Target labels tensor of shape [batch_size * G, seq_len] where
              positions to ignore are marked with -100
        temperature: Scaling factor for logits before softmax (default: TEMPERATURE)

    Returns:
        torch.Tensor: Log probabilities tensor of shape [batch_size * G, seq_len-1]
            - Each value represents the log probability of the token that appeared
            - Sequence length is reduced by 1 due to causal modeling requirements
            - Invalid positions (padding/query tokens) have log probability of 0
    """
    # Move inputs to correct device
    device = policy.device
    complete_sequences_ids = inputs["complete_sequences_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    labels = inputs["labels"].to(device)

    # Run forward pass
    outputs = policy(
        input_ids=complete_sequences_ids,
        attention_mask=attention_mask,
        return_dict=True,
        use_cache=False,
    )

    # Get logits and apply temperature
    logits = outputs.logits.float() / temperature

    # Shift sequences for causal modeling
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Create mask for valid labels
    label_mask = (shift_labels != IGNORE_INDEX).float()
    shift_labels[shift_labels == IGNORE_INDEX] = 0

    # Calculate log probabilities
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    log_probs = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(2))
    log_probs = log_probs.squeeze(2)
    log_probs = log_probs * label_mask

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
    # torch.expm1 is more stable than torch.exp - 1 ref: https://pytorch.org/docs/stable/special.html#torch.special.expm1
    kl_div = torch.expm1(log_quotient) - log_quotient
    return kl_div


def calculate_grpo_objective(
    model_log_probs: torch.Tensor,
    old_model_log_probs: torch.Tensor,
    ref_model_log_probs: torch.Tensor,
    advantages: torch.Tensor,
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
        prob_ratios = torch.ones_like(model_log_probs) # (batch_size * G, seq_length-1)
    else:
        prob_ratios = torch.exp(model_log_probs - old_model_log_probs) # (batch_size * G, seq_length-1)
    logger.info(f"Prob ratios: {prob_ratios}")
    # TODO: try increasing the epsilon
    clipped_ratios = torch.clamp(prob_ratios, 1 - eps, 1 + eps) # (batch_size * G, seq_length-1)
    assert (
        prob_ratios.shape == clipped_ratios.shape
    ), "Prob ratios and clipped ratios must have the same shape"
    # Expand the advantages to match the dimensions of prob_ratios
    advantages = advantages.unsqueeze(-1)
    device = prob_ratios.device
    advantages = advantages.to(device)

    """Reshape the advantages from (batch_size, G, 1) to (batch_size * G, 1) 
    before multiplying it with prob_ratios (batch_size * G, seq_length-1)"""
    advantages = advantages.reshape(prob_ratios.shape[0], -1)
    min_product = prob_ratios * advantages if mu ==1 else torch.min(prob_ratios * advantages, clipped_ratios * advantages)

    expected_advantage = model_log_probs * min_product # (batch_size * G, seq_length-1), model_log_probs zero outs padding tokens 

    assert (
        min_product.shape[0] == model_log_probs.shape[0]
        and min_product.shape[1] == model_log_probs.shape[1]
    ), "Min product must have the same batch size and number of outputs as model log probs"

    kl_div = kl_div_estimator(model_log_probs, ref_model_log_probs) # (batch_size * G, seq_length-1)

    logger.info(f"Mean KL Div: {torch.mean(kl_div).item()}") 
    wandb.log({"Mean KL Div": torch.mean(kl_div).item()})
    objective = expected_advantage - beta * kl_div 
    logger.info(f"Objective before mean: {objective}")
    # Take mean across all tokens and all outputs, and batch
    objective = torch.mean(objective)
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
        model_outputs = sample_outputs(
            policy_model,
            tokenizer,
            test_batch["prompt"],
            G=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        logger.debug(f"Evaluate Outputs: {model_outputs['all_responses']}")

    clear_cache()

    # Compute rewards and accuracies for each output
    rewards, accuracies = reward_model(model_outputs['all_responses'], test_batch)

    return rewards, accuracies
