import torch.nn as nn

from policy.irpo import IRPO_G_Learner, IRPO_Learner
from policy.irpo_is import IRPO_IS_Learner
from policy.irpo_thompson import IRPO_Thompson_Learner
from policy.irpo_trpo_final import IRPO_TRPOFinal_Learner
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from trainer.onpolicy_trainer import OnPolicyTrainer
from utils.intrinsic_rewards import (
    ALLOIntRewardFunctions,
    ALLOIntRewardFunctionG,
    ArbitraryIntRewardFunctions,
    RandomIntRewardFunctions,
    RandomIntRewardFunctionsG,
)
from utils.sampler import OnlineSampler


class IRPO_Algorithm(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(IRPO_Algorithm, self).__init__()

        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

        mode = getattr(args, "kernel_mode", "cosine")
        self.goal_conditioned = getattr(args, "is_goal_conditioned", False)

        if self.args.int_reward_type == "allo":
            if self.goal_conditioned:
                self.intrinsic_reward_fn = ALLOIntRewardFunctionG(
                    logger=logger, writer=writer, args=args, mode=mode
                )
            else:
                self.intrinsic_reward_fn = ALLOIntRewardFunctions(
                    logger=logger, writer=writer, args=args
                )
        elif self.args.int_reward_type == "random":
            if self.goal_conditioned:
                self.intrinsic_reward_fn = RandomIntRewardFunctionsG(
                    logger=logger,
                    writer=writer,
                    args=args,
                    mode=getattr(args, "kernel_mode", "rbf"),
                )
            else:
                self.intrinsic_reward_fn = RandomIntRewardFunctions(
                    logger=logger,
                    writer=writer,
                    args=args,
                )
        elif self.args.int_reward_type == "arbitrary":
            self.intrinsic_reward_fn = ArbitraryIntRewardFunctions(
                logger=logger, writer=writer, args=args, target=5.0
            )
        else:
            raise NotImplementedError(
                f"Intrinsic reward type '{self.args.int_reward_type}' not implemented."
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
        actor_activation = self._build_actor_activation(
            getattr(self.args, "actor_activation", None)
        )
        actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.actor_fc_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            activation=actor_activation,
            device=self.args.device,
        )
        critic = PPO_Critic(self.args.state_dim, hidden_dim=self.args.critic_fc_dim)

        shared_kwargs = dict(
            actor=actor,
            critic=critic,
            beta=self.args.beta,
            intrinsic_reward_fn=self.intrinsic_reward_fn,
            temperature=self.args.temperature,
            noise_std=self.args.noise_std,
            num_exp_updates=self.args.num_exp_updates,
            base_policy_update_type=self.args.base_policy_update_type,
            lr=self.args.learning_rate,
            entropy_scaler=self.args.entropy_scaler,
            target_kl=self.args.target_kl,
            gamma=self.args.gamma,
            gae=self.args.gae,
            device=self.args.device,
            anneal_kl=self.args.anneal_kl,
        )

        if self.goal_conditioned:
            self.policy = IRPO_G_Learner(env_name=self.args.env_name, **shared_kwargs)
        else:
            irpo_type = getattr(self.args, "irpo_type", "irpo")
            # Main IRPO is Thompson-sampling argmax. `irpo_legacy` exposes the
            # original deterministic-aggregation path for reference / debugging.
            learner_cls = {
                "irpo": IRPO_Thompson_Learner,
                "irpo_thompson": IRPO_Thompson_Learner,
                "irpo_is": IRPO_IS_Learner,
                "irpo_trpo": IRPO_TRPOFinal_Learner,
                "irpo_trpo_final": IRPO_TRPOFinal_Learner,
                "irpo_legacy": IRPO_Learner,
            }.get(irpo_type, IRPO_Thompson_Learner)

            extra_kwargs = {
                "min_base_updates": getattr(self.args, "min_base_updates", 1),
            }
            if learner_cls is IRPO_TRPOFinal_Learner:
                extra_kwargs["trpo_switch_progress"] = getattr(
                    self.args, "trpo_switch_progress", 0.5
                )
            if learner_cls is IRPO_Learner:
                extra_kwargs.pop("min_base_updates", None)

            self.policy = learner_cls(
                aggregation_method=self.args.aggregation_method,
                **shared_kwargs,
                **extra_kwargs,
            )

        if hasattr(self.env, "get_grid"):
            self.policy.actor.grid = self.env.get_grid()

    def _build_actor_activation(self, activation_name):
        if activation_name is None:
            return nn.Tanh()

        activation_key = str(activation_name).strip().lower()
        if activation_key == "relu":
            return nn.ReLU()
        if activation_key == "tanh":
            return nn.Tanh()

        raise ValueError(
            f"Unsupported actor activation '{activation_name}'. Use 'relu' or 'tanh'."
        )
