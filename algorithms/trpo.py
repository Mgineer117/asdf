import torch.nn as nn

from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from policy.trpo import TRPO_Learner
from trainer.onpolicy_trainer import OnPolicyTrainer
from utils.functions import build_activation
from utils.sampler import build_sampler


class TRPO_Algorithm(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(TRPO_Algorithm, self).__init__()

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

        # Rollout batch size is derived from minibatch settings (no --batch-size CLI).
        self.args.batch_size = int(args.minibatch_size * args.num_minibatch)
        self.args.nupdates = args.timesteps // self.args.batch_size

    def begin_training(self):
        # === Define policy === #
        self.define_policy()

        # === Sampler === #
        sampler = build_sampler(self.args)

        trainer = OnPolicyTrainer(
            env=self.env,
            policy=self.policy,
            sampler=sampler,
            logger=self.logger,
            writer=self.writer,
            init_timesteps=getattr(self.args, "resume_init_timesteps", 0),
            timesteps=self.args.timesteps,
            log_interval=self.args.log_interval,
            eval_num=self.args.eval_num,
            rendering=self.args.rendering,
            seed=self.args.seed,
        )

        trainer.train()

        return trainer.best_success_mean

    def define_policy(self):
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
        activation = build_activation(getattr(self.args, "activation", None))
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

        self.policy = TRPO_Learner(
            actor=actor,
            critic=critic,
            is_discrete=self.args.is_discrete,
            nupdates=self.args.nupdates,
            actor_lr=self.args.actor_lr,
            critic_lr=self.args.critic_lr,
            entropy_scaler=self.args.entropy_scaler,
            batch_size=self.args.batch_size,
            target_kl=self.args.target_kl,
            gamma=self.args.gamma,
            gae=self.args.gae,
            pos_idx=pos_idx,
            goal_idx=goal_idx,
            grad_batch_size=getattr(self.args, "minibatch_size", 256),
            device=self.args.device,
        )

        if hasattr(self.env, "get_grid"):
            self.policy.grid = self.env.get_grid()
