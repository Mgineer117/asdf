import os

import torch.nn as nn

from policy.irpo import NUM_GOALS, IRPO_G_Learner, IRPO_Learner
from policy.layers.building_blocks import ConvVAEEncoder
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from trainer.onpolicy_trainer import OnPolicyTrainer
from utils.functions import build_activation
from utils.intrinsic_rewards import (
    ALLOIntRewardFunctions,
    ALLOIntRewardFunctionG,
    ArbitraryIntRewardFunctions,
    RandomIntRewardFunctions,
    RandomIntRewardFunctionsG,
)
from utils.sampler import build_sampler

_ATARI_ENVS = {"pacman", "amidar", "bankheist", "alien"}


class IRPO_Algorithm(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(IRPO_Algorithm, self).__init__()

        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

        mode = getattr(args, "kernel_mode", "cosine")
        self.goal_conditioned = getattr(args, "is_goal_conditioned", False)

        # Safeguard: ALLO on Atari requires a pre-saved VAE encoder so it can
        # train on consistent encoded features.  Run IRPO first (any int_reward_type)
        # to produce model/<env>/encoder/<seed>.pth, then run train_models.py.
        env_name_base = args.env_name.split("-")[0]
        if self.args.int_reward_type == "allo" and env_name_base in _ATARI_ENVS:
            encoder_path = os.path.join(
                "model", env_name_base, "encoder", f"{args.seed}.pth"
            )
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(
                    f"ALLO on Atari requires a pre-trained VAE encoder at "
                    f"'{encoder_path}'.  Run IRPO (any int_reward_type) first "
                    f"to train and save the encoder, then run train_models.py "
                    f"to train the ALLO model."
                )

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
        sampler = build_sampler(self.args)

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
        activation = build_activation(getattr(self.args, "actor_activation", None))
        # For Atari: CNN is trained by VAE only — IRPO must NOT backprop through it.
        # detach_cnn=True makes the actor detach CNN features before the MLP so
        # create_graph=True in IRPO only stores the small MLP graph (fixes OOM).
        env_name_base = self.args.env_name.split("-")[0]
        is_atari = env_name_base in _ATARI_ENVS

        actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.actor_fc_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            activation=activation,
            detach_cnn=is_atari,
            device=self.args.device,
        )
        critic = PPO_Critic(
            self.args.state_dim,
            hidden_dim=self.args.critic_fc_dim,
            activation=activation,
            device=self.args.device,
        )

        # Replace actor's plain CNN with ConvVAEEncoder (trained via VAE loss).
        vae_encoder = None
        if is_atari:
            H, W = self.args.state_dim  # raw grayscale (H, W)
            latent_dim = getattr(self.args, "encoder_dim", 256)
            vae_encoder = ConvVAEEncoder(
                input_shape=(1, H, W),
                latent_dim=latent_dim,
                device=self.args.device,
            )
            # Swap out the plain CNN; MLP head is compatible (same output dim=256).
            actor.feature_extractor = vae_encoder

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
            grad_batch_size=getattr(self.args, "minibatch_size", 256),
        )

        # IRPO_G_Learner is built for maze-family envs with a *discrete* goal
        # set (clustered via NUM_GOALS); continuous-goal envs like FetchReach
        # are goal-conditioned but don't fit that mould, so they fall through
        # to the standard IRPO_Learner path with a single intrinsic reward.
        use_g_learner = (
            self.goal_conditioned and self.args.env_name in NUM_GOALS
        )
        if use_g_learner:
            self.policy = IRPO_G_Learner(env_name=self.args.env_name, **shared_kwargs)
        else:
            self.policy = IRPO_Learner(
                aggregation_method=self.args.aggregation_method,
                vae_encoder=vae_encoder,
                **shared_kwargs,
            )

        if hasattr(self.env, "get_grid"):
            self.policy.actor.grid = self.env.get_grid()

