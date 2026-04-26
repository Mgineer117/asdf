import torch.nn as nn

from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from utils.functions import build_activation
from policy.maml import MAML_Learner
from trainer.onpolicy_trainer import OnPolicyTrainer
from utils.intrinsic_rewards import ALLOIntRewardFunctions, RandomIntRewardFunctions
from utils.sampler import OnlineSampler


class MAML_Algorithm(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(MAML_Algorithm, self).__init__()

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

        if self.args.int_reward_type == "allo":
            fn = ALLOIntRewardFunctions
        elif self.args.int_reward_type == "random":
            fn = RandomIntRewardFunctions
        else:
            NotImplementedError(
                f"Intrinsic reward type {self.args.int_reward_type} not implemented."
            )

        self.intrinsic_reward_fn = fn(
            logger=logger,
            writer=writer,
            args=args,
        )

        self.current_timesteps = self.intrinsic_reward_fn.current_timesteps

    def begin_training(self):
        # === Sampler === #
        sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.args.episode_len,
            batch_size=self.args.batch_size,
        )

        # === Meta-train using options === #'
        self.define_base_policy()
        trainer = OnPolicyTrainer(
            env=self.env,
            policy=self.policy,
            sampler=sampler,
            logger=self.logger,
            writer=self.writer,
            init_timesteps=self.current_timesteps,
            timesteps=self.args.timesteps,
            log_interval=self.args.log_interval,
            eval_num=self.args.eval_num,
            rendering=self.args.rendering,
            seed=self.args.seed,
        )
        final_steps = trainer.train()
        self.current_timesteps += final_steps

        return trainer.best_success_mean

    def define_base_policy(self):
        # === Define policy === #
        pos_idx = (
            self.args.pos_idx
            if getattr(self.args, "is_goal_conditioned", False)
            else None
        )
        goal_idx = (
            self.args.goal_idx
            if getattr(self.args, "is_goal_conditioned", False)
            else None
        )
        activation = build_activation(getattr(self.args, "actor_activation", None))
        actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.actor_fc_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            activation=activation,
            device=self.args.device,
        )
        critic = PPO_Critic(
            self.args.state_dim,
            hidden_dim=self.args.critic_fc_dim,
            activation=activation,
            device=self.args.device,
        )

        self.policy = MAML_Learner(
            actor=actor,
            critic=critic,
            intrinsic_reward_fn=self.intrinsic_reward_fn,
            timesteps=self.args.timesteps,
            num_exp_updates=self.args.num_exp_updates,
            base_policy_update_type=self.args.base_policy_update_type,
            lr=self.args.learning_rate,
            entropy_scaler=self.args.entropy_scaler,
            target_kl=self.args.target_kl,
            gamma=self.args.gamma,
            gae=self.args.gae,
            pos_idx=pos_idx,
            goal_idx=goal_idx,
            device=self.args.device,
        )

        if hasattr(self.env, "get_grid"):
            self.policy.actor.grid = self.env.get_grid()
