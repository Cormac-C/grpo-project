import torch
import torch.nn.functional as F
import logging
from transformers import GenerationConfig

# Define constants
# TODO: Reduced max_new_tokens for testing, return to original value for full training
MAX_NEW_TOKENS = 64  # 1024
TEMPERATURE = 1.0
STABILITY_CONST = 1e-8

# Configure logging
logger = logging.getLogger(__name__)

# TODO: add types for all functions
# TODO: clean up logger messages


def grpo_iteration(
    query_batch_prompts,
    query_batch_raw,
    policy_model,
    reference_model,
    reward_model,
    tokenizer,
    optimizer,
    G,
    eps,
    beta,
    mu,
):
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
        policy_model, tokenizer, query_batch_prompts, G
    )

    # Compute rewards and accuracies for each output
    rewards, accuracies = reward_model(outputs, query_batch_raw)

    # Compute token-level advantage for each token in each output
    advantages = calculate_grpo_advantage(rewards)

    #  Compute log probabilities for reference model and pre-update policy, no gradients here
    with torch.no_grad():
        old_log_probs = compute_log_probs(
            policy=policy_model,
            tokenizer=tokenizer,
            query_batch=query_batch_prompts,
            generated_ids=outputs_ids,
        )
        reference_model_log_probs = compute_log_probs(
            policy=reference_model,
            tokenizer=tokenizer,
            query_batch=query_batch_prompts,
            generated_ids=outputs_ids,
        )

    for i in range(mu):
        logger.info(f"Update iteration: {i+1}/{mu}")
        # Compute log probabilities for the current policy model, this needs gradients
        model_log_probs = compute_log_probs(
            policy=policy_model,
            tokenizer=tokenizer,
            query_batch=query_batch_prompts,
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
        loss.backward()

        # Update the policy
        optimizer.step()
        optimizer.zero_grad()

    return policy_model


def sample_outputs(policy, tokenizer, query_batch, G):
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
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        num_return_sequences=G,
        temperature=TEMPERATURE,
        return_dict_in_generate=True,
        output_scores=False,
    )

    # Tokenize the batch of queries
    tokenized_queries = tokenizer(
        query_batch, return_tensors="pt", padding=True, truncation=True
    )

    # Move the tokenized batch to the device of the policy
    tokenized_queries = {
        key: value.to(policy.device) for key, value in tokenized_queries.items()
    }

    # Sample G outputs from the policy for each query in query_batch
    with torch.no_grad():
        output = policy.generate(**tokenized_queries, generation_config=gen_config)

    output_ids = output.sequences
    # logger.info(f"Output IDs shape: {output_ids.shape}")

    # Separate the generated IDs from the input IDs
    generated_ids = output_ids[:, tokenized_queries["input_ids"].shape[1] :]
    logger.info(f"Generated IDs shape: {generated_ids.shape}")

    # Decode the generated IDs to text
    batch_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # logger.info(f"Batch responses: {batch_responses}")
    # Reshape the outputs to have shape (batch_size, G), each item contains the generated text and log probs
    batch_size = len(query_batch)
    responses_reshaped = [
        batch_responses[i * G : (i + 1) * G] for i in range(batch_size)
    ]
    logger.info(
        f"Responses shape: {len(responses_reshaped)}, {len(responses_reshaped[0])}"
    )

    # Reshape the generated IDs to have shape (batch_size, G, max_length)
    generated_ids_reshaped = generated_ids.view(batch_size, G, -1)
    logger.info(f"Generated IDs reshaped: {generated_ids_reshaped.shape}")

    return generated_ids_reshaped, responses_reshaped


def calculate_grpo_advantage(rewards):
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

    logger.info(f"Advantages shape: {advantages.shape}")
    return advantages


def compute_log_probs(policy, tokenizer, query_batch, generated_ids):
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
    # TODO: Chill out the comments in this function
    # Tokenize the batch of queries
    tokenized_queries = tokenizer(
        query_batch, return_tensors="pt", padding=True, truncation=True
    )
    query_ids = tokenized_queries["input_ids"]
    # Expand the query IDs to match the shape of generated IDs
    query_ids = query_ids.unsqueeze(1).expand(-1, generated_ids.shape[1], -1)
    logger.info(f"Query IDs shape: {query_ids.shape}")
    logger.info(f"Generated IDs shape: {generated_ids.shape}")

    # Concatenate the query IDs and generated IDs
    input_ids = torch.cat((query_ids, generated_ids), dim=2)
    logger.info(f"Input IDs shape: {input_ids.shape}")
    # Reshape the input IDs to have shape (batch_size * G, max_length)
    input_ids = input_ids.view(-1, input_ids.shape[-1])
    logger.info(f"Reshaped Input IDs shape: {input_ids.shape}")

    # Move the input IDs to the device of the policy
    input_ids = input_ids.to(policy.device)
    # Get the attention mask
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to(policy.device)
    logger.info(f"Attention mask shape: {attention_mask.shape}")

    # Run forward pass to get the logits
    outputs = policy(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    logger.info(f"Logits shape: {logits.shape}")

    # Separate the logits for the generated IDs
    generated_logits = logits[:, query_ids.shape[2] :]
    logger.info(f"Generated logits shape: {generated_logits.shape}")

    # Calculate log probabilities
    log_probs = F.log_softmax(generated_logits, dim=-1)
    logger.info(f"Log probabilities shape: {log_probs.shape}")
    # Get the log probabilities for the generated IDs
    generated_ids = generated_ids.view(-1, generated_ids.shape[-1])
    log_probs = log_probs.gather(2, generated_ids.unsqueeze(-1)).squeeze(-1)
    logger.info(f"Gathered log probabilities shape: {log_probs.shape}")

    # Reshape the log probabilities to have shape (batch_size, G, max_length)
    batch_size = len(query_batch)
    log_probs = log_probs.view(batch_size, -1, generated_ids.shape[-1])
    logger.info(f"Reshaped log probabilities shape: {log_probs.shape}")

    return log_probs


def kl_div_estimator(model_log_probs, ref_model_log_probs):
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
    model_log_probs, old_model_log_probs, ref_model_log_probs, advantages, eps, beta
):
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
    # TODO: Need to revisit model_log_probs and old_model_log_probs, model log probs aren't available to actually calculate the objective
    prob_ratios = torch.exp(model_log_probs - old_model_log_probs)
    logger.info(f"Prob ratios shape: {prob_ratios.shape}")
    clipped_ratios = torch.clamp(prob_ratios, 1 - eps, 1 + eps)
    logger.info(f"Clipped ratios shape: {clipped_ratios.shape}")
    # Expand the advantages to match the dimensions of prob_ratios
    advantages = advantages.unsqueeze(-1)
    logger.info(f"Advantages shape: {advantages.shape}")
    min_product = torch.min(prob_ratios * advantages, clipped_ratios * advantages)
    logger.info(f"Min product shape: {min_product.shape}")
    # Estimate KL
    kl_div = kl_div_estimator(model_log_probs, ref_model_log_probs)
    logger.info(f"KL divergence shape: {kl_div.shape}")
    # Combine the KL term into objective
    objective = min_product - beta * kl_div
    logger.info(f"Objective shape: {objective.shape}")
    # Take mean across all tokens and all outputs
    objective = torch.mean(objective, dim=[1, 2])
    logger.info(f"Final objective shape: {objective.shape}")
    return objective
