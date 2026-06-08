import torch
from utils.rl import average_gradients_across_minibatches
from policy.layers.ppo_networks import PPO_Actor

actor = PPO_Actor(input_dim=(1, 84, 84), hidden_dim=[128, 128], action_dim=4, is_discrete=True)
states = torch.rand(4, 1, 84, 84)
actions = torch.randint(0, 4, (4,))
advantages = torch.rand(4)

def actor_loss_fn(s, a, adv):
    _, metaData = actor(s)
    logprobs = actor.log_prob(metaData["dist"], a)
    return -(logprobs * adv).mean()

grads = average_gradients_across_minibatches(
    actor, actor_loss_fn, states, actions, advantages, minibatch_size=2, create_graph=True
)

for i, (name, p) in enumerate(actor.named_parameters()):
    g = grads[i]
    print(f"Element {i}: {name}, req_grad={g.requires_grad}, shape={g.shape}")
