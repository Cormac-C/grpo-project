import torch
from transformers import GenerationConfig

MAX_NEW_TOKENS = 1024
TEMPERATURE = 1.0
STABILITY_CONST = 1e-8


def grpo_iteration(
    d_b, policy_model, reward_model, tokenizer, optimizer, G, eps, beta, mu
):
    """
    Perform one iteration of the GRPO algorithm.

    Args:
        d_b: Batch of queries.
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
    # Sample G outputs from the policy for each query in d_b
    outputs = sample_outputs(policy_model, tokenizer, d_b, G)

    # Compute rewards and accuracies for each output
    rewards, accuracies = calculate_rewards_and_accuracies(d_b, outputs, reward_model)

    # Compute token-level advantage for each token in each output
    advantages = calculate_grpo_advantage(outputs, rewards)

    for _ in range(mu):
        # Compute GRPO objective
        objective = calculate_grpo_objective(
            policy_model.log_probs(outputs),
            policy_model.old_log_probs(outputs),
            policy_model.ref_model_log_probs(outputs),
            advantages,
            eps,
            beta,
        )

        # Compute gradient of the GRPO objective
        loss = -objective
        loss.backward()

        # Update the policy
        optimizer.step()
        optimizer.zero_grad()

    return policy_model


def sample_outputs(policy, tokenizer, d_b, G):
    """
    Sample G outputs from the policy for each query in d_b.

    Args:
        policy: The current policy.
        tokenizer: The tokenizer for the policy model.
        d_b: Batch of queries.
        G: The number of outputs to sample.
        max_length: The maximum length of the generated sequences.

    Returns:
        A list of sampled outputs.
    """
    gen_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        num_return_sequences=G,
        temperature=TEMPERATURE,
    )

    with torch.no_grad():
        # Move the batch to the device of the policy
        d_b = {key: value.to(policy.device) for key, value in d_b.items()}

        tokenized_queries = tokenizer(d_b, return_tensors="pt", padding=True)

        # Sample G outputs from the policy for each query in d_b
        outputs = policy.generate(**tokenized_queries, generation_config=gen_config)

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(d_b["input_ids"], generated_ids)
        ]

        batch_responses = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

    # Reshape the outputs to have shape (batch_size, G, max_length)
    outputs = batch_responses.view(len(d_b), G, -1)

    return outputs


def calculate_rewards_and_accuracies(d_b, outputs, reward_model):
    """
    Calculate the rewards for each output across the batch of queries.

    Args:
        d_b: Batch of queries.
        outputs: The sampled outputs.
        reward_model: The reward model.
    Returns:
        A tensor of rewards for each output.
    """
    # Rewards are scalars so shape is (batch_size, G)
    rewards = torch.zeros(len(d_b), outputs.shape[1])
    accuracies = torch.zeros(len(d_b), outputs.shape[1])

    # Assume reward model can accomodate a batch of queries and outputs
    metrics = reward_model(d_b, outputs)
    rewards = metrics["reward_score"]
    accuracies = metrics["accuracy"]
    return rewards, accuracies


def calculate_grpo_advantage(outputs, rewards):
    """
    Calculate token-level advantage for each token of each output.
    Args:
        outputs: The sampled outputs.
        rewards: The rewards for each output.
    Returns:
        A tensor of advantages for each token in each output.
    """
    # TODO: revisit, maybe don't need to pass in outputs
    # Outputs have shape (batch_size, G, max_length)
    # Rewards have shape (batch_size, G)
    # Advantages have shape (batch_size, G, max_length)

    advantages = torch.zeros(outputs.shape)
    for i, output_group in enumerate(outputs):
        group_mean = torch.mean(rewards[i])
        group_std = torch.std(rewards[i])
        # Reponse advantage has shape (G), is normalized by group mean and std
        response_advantage = (rewards[i] - group_mean) / (group_std + STABILITY_CONST)
        # Spread response advantage across all tokens in the output
        advantages[i] = response_advantage.unsqueeze(1).expand(-1, outputs.shape[2])

    return advantages


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
        The GRPO objective value.
    """
    prob_ratios = torch.exp(model_log_probs - old_model_log_probs)
    clipped_ratios = torch.clamp(prob_ratios, 1 - eps, 1 + eps)
    min_product = torch.min(prob_ratios * advantages, clipped_ratios * advantages)
    # Estimate KL
    kl_div = kl_div_estimator(model_log_probs, ref_model_log_probs)
    # Combine the KL term into objective
    objective = min_product - beta * kl_div
    # Take mean across all tokens and all outputs
    objective = torch.mean(objective, dim=[0, 1])
    return objective
