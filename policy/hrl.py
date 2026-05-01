import numpy as np
import torch

from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from policy.ppo import PPO_Learner


class HRL_Learner(PPO_Learner):
    """
    High-level (option-selecting) PPO learner for hierarchical RL.

    Reuses the entire PPO update from `PPO_Learner` (collect → compute
    rewards → GAE → minibatch loop). Overrides only:
    - `__init__`: add option-policy storage and obs-RMS setup
    - `forward`: option-aware action selection
    - `_on_batch_collected`: refresh obs-RMS and record state visitations
    """

    def __init__(
        self,
        actor: PPO_Actor,
        critic: PPO_Critic,
        is_discrete: bool,
        nupdates: int,
        lr: float = 3e-4,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        eps_clip: float = 0.2,
        entropy_scaler: float = 1e-3,
        l2_reg: float = 1e-8,
        target_kl: float = 0.03,
        gamma: float = 0.99,
        gae: float = 0.9,
        K: int = 5,
        pos_idx: list = None,
        goal_idx: list = None,
        device: str = "cpu",
    ):
        super().__init__(
            actor=actor,
            critic=critic,
            is_discrete=is_discrete,
            nupdates=nupdates,
            lr=lr,
            num_minibatch=num_minibatch,
            minibatch_size=minibatch_size,
            eps_clip=eps_clip,
            entropy_scaler=entropy_scaler,
            l2_reg=l2_reg,
            target_kl=target_kl,
            gamma=gamma,
            gae=gae,
            K=K,
            device=device,
        )

        self.name = "HRL"
        self.action_dim = actor.action_dim
        self.policies = [None]  # populated by HRLTrainer

        self.setup_obs_rms(actor.input_shape, pos_idx=pos_idx, goal_idx=goal_idx)
        self.sync_obs_rms_to(self.actor, self.critic)

    def forward(
        self,
        state: np.ndarray,
        option_idx: int | None,
        deterministic: bool = False,
        **kwargs,
    ):
        state = self.preprocess_state(state)
        if option_idx is None:
            logits, metaData = self.actor(state, deterministic=deterministic)
            option_idx = torch.argmax(logits, dim=-1).item()
        else:
            logits = torch.tensor(np.full((1, self.action_dim), np.nan)).to(self.device)
            metaData = {
                "probs": torch.tensor(np.nan).to(self.device),
                "logprobs": torch.tensor(np.nan).to(self.device),
                "entropy": torch.tensor(np.nan).to(self.device),
                "dist": torch.tensor(np.nan).to(self.device),
            }

        is_option = option_idx < len(self.policies) - 1
        if is_option:
            a, _ = self.policies[option_idx].actor(state, deterministic=True)
            value = self.policies[option_idx].critic(state)

            option_termination = value.item() < 0
        else:
            a, _ = self.policies[option_idx](state, deterministic=True)
            option_termination = False

        return [option_idx, a], {
            "logits": logits,
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
            "dist": metaData["dist"],
            "is_option": is_option,
            "option_termination": option_termination,
        }

    def _on_batch_collected(self, states):
        self.update_obs_rms(states)
        self.sync_obs_rms_to(self.actor, self.critic)
        self.record_state_visitations(states)
