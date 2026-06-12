import os

import torch
import torch.nn as nn

from policy.irpo import NUM_GOALS, IRPO_G_Learner, IRPO_Learner
from policy.layers.building_blocks import CNN, MLP, ConvVAEEncoder
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

        # Rollout batch size is derived from minibatch settings (no --batch-size CLI).
        self.args.batch_size = int(args.minibatch_size * args.num_minibatch)

        mode = getattr(args, "kernel_mode", "cosine")
        self.goal_conditioned = getattr(args, "is_goal_conditioned", False)

        # Safeguard: ALLO on Atari trains its own CNN encoder (no VAE) first; the
        # IRPO policy then loads that encoder FROZEN.  So the ALLO encoder must
        # already exist — run train_models.py (int_reward_type=allo) beforehand.
        env_name_base = args.env_name.split("-")[0]
        if self.args.int_reward_type == "allo" and env_name_base in _ATARI_ENVS:
            encoder_path = os.path.join(
                "model", env_name_base, "allo_encoder", f"{args.seed}.pth"
            )
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(
                    f"ALLO on Atari requires a CNN encoder trained by ALLO at "
                    f"'{encoder_path}'.  Run train_models.py "
                    f"(int_reward_type=allo) first to train ALLO and export its "
                    f"encoder, then run IRPO with int_reward_type=allo."
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
        # For Atari: a single shared VAE encoder (CNN) is the visual representation,
        # initialized from pre-trained weights and then FINE-TUNED during IRPO by
        # the VAE objective on incoming policy samples. detach_cnn=True on both the
        # actor and critic ensures the policy/value gradients never train the CNN
        # (only the VAE optimizer does) and keeps IRPO's create_graph=True graphs
        # limited to the small MLP heads.
        env_name_base = self.args.env_name.split("-")[0]
        is_atari = env_name_base in _ATARI_ENVS

        cnn_mode = getattr(self.args, "cnn_mode", "simultaneous")
        detach_cnn = True if (is_atari and cnn_mode == "independent") else False

        actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.actor_fc_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            activation=activation,
            detach_cnn=detach_cnn,
            device=self.args.device,
        )
        critic = PPO_Critic(
            self.args.state_dim,
            hidden_dim=self.args.critic_fc_dim,
            activation=activation,
            detach_cnn=detach_cnn,
            device=self.args.device,
        )

        # Build the shared Atari visual encoder. The encoder source and whether it
        # is fine-tuned depend on int_reward_type:
        #   random : pretrain a FRESH VAE every run (no disk load), then keep
        #            fine-tuning it jointly via the VAE loss (train_vae=True).
        #   allo   : load the CNN encoder ALLO trained (no VAE) and use it FROZEN
        #            (train_vae=False); IRPO never updates these weights.
        # latent_dim must match PPO_Actor's CNN feature dim (256) so the MLP head
        # stays compatible.
        vae_encoder = None
        train_vae = True
        if is_atari:
            H, W = self.args.state_dim  # raw grayscale (H, W)
            latent_dim = 256

            if self.args.int_reward_type == "allo":
                # Frozen ALLO-trained CNN encoder (existence guaranteed by the
                # __init__ safeguard).
                encoder = CNN(
                    input_shape=(1, H, W),
                    features_dim=latent_dim,
                    initialization="default",
                    device=self.args.device,
                )
                allo_encoder_path = os.path.join(
                    "model", env_name_base, "allo_encoder", f"{self.args.seed}.pth"
                )
                encoder.load_state_dict(
                    torch.load(allo_encoder_path, map_location=self.args.device)
                )
                # Frozen in effect WITHOUT requires_grad_(False): the IRPO
                # gradient code calls autograd.grad(..., actor.parameters()),
                # which errors on params that don't require grad. The encoder is
                # never updated anyway — detach_cnn=True zeroes its policy/value
                # gradients, train_vae=False means no VAE optimizer, and its
                # params are excluded from the critic optimizers.
                vae_encoder = encoder
                train_vae = False
                print(
                    f"[IRPO] Loaded FROZEN ALLO encoder from {allo_encoder_path}"
                )
            else:
                # Fresh VAE, pretrained in-process every run, then joint-tuned.
                vae_encoder = ConvVAEEncoder(
                    input_shape=(1, H, W),
                    latent_dim=latent_dim,
                    device=self.args.device,
                )
                if self.args.int_reward_type == "random" and cnn_mode == "independent":
                    from pretrain_vae import train_vae_encoder

                    epochs = int(getattr(self.args, "vae_pretrain_epochs", 50))
                    samples = int(getattr(self.args, "vae_pretrain_samples", 100000))
                    batch = int(getattr(self.args, "vae_pretrain_batch_size", 256))
                    print(
                        f"[IRPO] Pretraining a fresh VAE encoder for "
                        f"int_reward_type=random ({epochs} epochs, {samples} samples)."
                    )
                    train_vae_encoder(
                        vae_encoder,
                        env=self.env,
                        num_epochs=epochs,
                        batch_size=batch,
                        num_samples=samples,
                        device=self.args.device,
                    )
                train_vae = (cnn_mode == "independent")
                if not train_vae:
                    print(f"[IRPO] cnn_mode={cnn_mode}. CNN encoder is trained end-to-end by RL like PPO. VAE loss is disabled.")

            # Share the SAME encoder as the feature extractor for both the actor
            # and the critic. The critic's MLP head is rebuilt to consume the
            # encoder's latent_dim (instead of its own 512-d CNN), so the whole
            # learner holds exactly ONE CNN.
            actor.feature_extractor = vae_encoder
            critic.feature_extractor = vae_encoder
            critic.model = MLP(
                latent_dim,
                self.args.critic_fc_dim,
                1,
                activation=activation,
                initialization="critic",
                device=self.args.device,
            )

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
                # random: joint VAE fine-tuning of a freshly pretrained encoder.
                # allo: frozen ALLO-trained CNN (no VAE loss).
                train_vae=train_vae,
                **shared_kwargs,
            )

        if hasattr(self.env, "get_grid"):
            self.policy.actor.grid = self.env.get_grid()

