import torch


def average_gradients_across_minibatches(
    model, loss_fn, data_states, data_actions, data_advantages, minibatch_size, create_graph=False
):
    """
    Compute gradients by splitting data into minibatches, accumulating gradients,
    then averaging. Useful for reducing estimation error while keeping memory low.

    Args:
        model: nn.Module to compute gradients for
        loss_fn: callable that takes (states, actions, advantages) and returns scalar loss
        data_states, data_actions, data_advantages: full batch tensors
        minibatch_size: size of each minibatch
        create_graph: if True, allow 2nd derivatives (for IRPO meta-gradients)

    Returns:
        Tuple of averaged gradient tensors, one per parameter
    """
    B = data_states.shape[0]
    all_gradients = []

    for start_idx in range(0, B, minibatch_size):
        end_idx = min(start_idx + minibatch_size, B)
        mb_states = data_states[start_idx:end_idx]
        mb_actions = data_actions[start_idx:end_idx]
        mb_advantages = data_advantages[start_idx:end_idx]

        loss = loss_fn(mb_states, mb_actions, mb_advantages)
        grads = torch.autograd.grad(
            loss, model.parameters(), create_graph=create_graph, allow_unused=True
        )
        grads = tuple(
            g if g is not None else torch.zeros_like(p)
            for p, g in zip(model.parameters(), grads)
        )
        all_gradients.append(grads)
        del loss, grads

    # Average gradients across all minibatches
    num_batches = len(all_gradients)
    averaged_grads = tuple(
        sum(all_gradients[i][param_idx] for i in range(num_batches)) / num_batches
        for param_idx in range(len(all_gradients[0]))
    )
    return averaged_grads


def average_loss_across_minibatches(loss_fn, data_states, data_actions, data_returns, minibatch_size):
    """
    Compute average loss across minibatches. Useful for critic updates to use
    all data points while keeping batch size manageable.

    Returns:
        Scalar tensor (average loss across all minibatches)
    """
    B = data_states.shape[0]
    all_losses = []

    for start_idx in range(0, B, minibatch_size):
        end_idx = min(start_idx + minibatch_size, B)
        mb_states = data_states[start_idx:end_idx]
        mb_actions = data_actions[start_idx:end_idx]
        mb_returns = data_returns[start_idx:end_idx]

        loss = loss_fn(mb_states, mb_actions, mb_returns)
        all_losses.append(loss.detach())

    avg_loss = sum(all_losses) / len(all_losses)
    return avg_loss


def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model, flat_params):
    pointer = 0
    for p in model.parameters():
        num_param = p.numel()
        p.data.copy_(flat_params[pointer : pointer + num_param].view_as(p))
        pointer += num_param


def compute_kl(old_actor, new_policy, obs):
    _, old_infos = old_actor(obs)
    _, new_infos = new_policy(obs)

    kl = torch.distributions.kl_divergence(old_infos["dist"], new_infos["dist"])
    return kl.mean()


def hessian_vector_product(kl_fn, model, damping, v):
    kl = kl_fn()
    # allow_unused=True: CNN params are detached inside the actor so kl has no
    # path to them; their gradient is None, which we replace with zeros.
    grads = torch.autograd.grad(kl, model.parameters(), create_graph=True, allow_unused=True)
    grads = tuple(g if g is not None else torch.zeros_like(p)
                  for p, g in zip(model.parameters(), grads))
    flat_grads = torch.cat([g.view(-1) for g in grads])
    g_v = (flat_grads * v).sum()
    hv = torch.autograd.grad(g_v, model.parameters(), allow_unused=True)
    hv = tuple(h if h is not None else torch.zeros_like(p)
               for p, h in zip(model.parameters(), hv))
    flat_hv = torch.cat([h.contiguous().view(-1) for h in hv])
    return flat_hv + damping * v


def conjugate_gradients(Av_func, b, nsteps=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        Avp = Av_func(p)
        alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        if new_rdotr < tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    return x


def estimate_advantages(
    rewards: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate advantages and returns using Generalized Advantage Estimation (GAE),
    handling infinite-horizon truncations correctly.

    Args:
        rewards (Tensor): Reward at each timestep, shape [T, 1]
        terminations (Tensor): Binary termination indicators (1 if truly terminated), shape [T, 1]
        truncations (Tensor): Binary truncation indicators (1 if time limit reached), shape [T, 1]
        values (Tensor): Value function estimates, shape [T, 1]
        gamma (float): Discount factor.
        gae (float): GAE lambda.

    Returns:
        advantages (Tensor): Estimated advantages, shape [T, 1]
        returns (Tensor): Estimated returns (value targets), shape [T, 1]
    """
    device = rewards.device  # Infer device from input tensor

    T = rewards.size(0)
    deltas = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    # Precompute episode boundaries (dones = terminations OR truncations)
    dones = torch.logical_or(terminations, truncations).float()

    prev_value = torch.tensor(0.0, device=device)
    prev_advantage = torch.tensor(0.0, device=device)

    for t in reversed(range(T)):
        # 1. Bootstrap mask: Only zero out next value if it's a TRUE termination
        bootstrap_mask = 1.0 - terminations[t]

        # 2. GAE chain mask: Break the advantage chain if the episode ended for ANY reason
        gae_mask = 1.0 - dones[t]

        # TD Error (Value survives truncations)
        deltas[t] = rewards[t] + gamma * prev_value * bootstrap_mask - values[t]

        # Advantage (Chain severed at any episode boundary)
        advantages[t] = deltas[t] + gamma * gae * prev_advantage * gae_mask

        prev_value = values[t]
        prev_advantage = advantages[t]

    returns = values + advantages
    return advantages, returns
