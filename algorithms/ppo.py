import torch.nn as nn

from policy.ppo import PPO_Learner
from policy.ppo_allo import PPO_ALLO_Learner
from trainer.onpolicy_trainer import OnPolicyTrainer
from utils.functions import build_actor_critic_pair
from utils.intrinsic_rewards import ALLOIntRewardFunctionG
from utils.sampler import OnlineSampler


class PPO_Algorithm(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(PPO_Algorithm, self).__init__()

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

        self.args.nupdates = args.timesteps // (
            args.minibatch_size * args.num_minibatch
        )

        self.use_allo = getattr(args, "int_reward_type", None) == "allo"
        if self.use_allo:
            if not getattr(args, "is_goal_conditioned", False):
                raise ValueError(
                    "PPO with --int-reward-type allo requires a goal-conditioned "
                    "environment (uses ALLOIntRewardFunctionG)."
                )
            self.intrinsic_reward_fn = ALLOIntRewardFunctionG(
                logger=logger,
                writer=writer,
                args=args,
                mode=getattr(args, "kernel_mode", "rbf"),
            )
        else:
            self.intrinsic_reward_fn = None

    def begin_training(self):
        # === Define policy === #
        self.define_policy()

        # === Sampler === #
        sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.args.episode_len,
            batch_size=int(self.args.minibatch_size * self.args.num_minibatch),
        )

        trainer = OnPolicyTrainer(
            env=self.env,
            policy=self.policy,
            sampler=sampler,
            logger=self.logger,
            writer=self.writer,
            timesteps=self.args.timesteps,
            log_interval=self.args.log_interval,
            eval_num=self.args.eval_num,
            rendering=self.args.rendering,
            seed=self.args.seed,
        )

        trainer.train()

        return trainer.best_success_mean

    def define_policy(self):
        actor, critic = build_actor_critic_pair(self.args)

        ppo_kwargs = dict(
            actor=actor,
            critic=critic,
            is_discrete=self.args.is_discrete,
            nupdates=self.args.nupdates,
            lr=self.args.learning_rate,
            num_minibatch=self.args.num_minibatch,
            minibatch_size=self.args.minibatch_size,
            eps_clip=self.args.eps_clip,
            entropy_scaler=self.args.entropy_scaler,
            target_kl=self.args.target_kl,
            gamma=self.args.gamma,
            gae=self.args.gae,
            K=self.args.K_epochs,
            device=self.args.device,
        )

        if self.use_allo:
            self.policy = PPO_ALLO_Learner(
                intrinsic_reward_fn=self.intrinsic_reward_fn,
                alpha=getattr(self.args, "int_reward_alpha", 1.0),
                alpha_final=getattr(self.args, "int_reward_alpha_final", 0.0),
                **ppo_kwargs,
            )
        else:
            self.policy = PPO_Learner(**ppo_kwargs)

        if hasattr(self.env, "get_grid"):
            self.policy.grid = self.env.get_grid()
