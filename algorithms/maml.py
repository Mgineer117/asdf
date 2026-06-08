import torch.nn as nn

from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from utils.functions import build_activation
from policy.maml import MAML_Learner
from trainer.onpolicy_trainer import OnPolicyTrainer
from utils.intrinsic_rewards import ALLOIntRewardFunctions, RandomIntRewardFunctions
from utils.sampler import build_sampler


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
        
        env_name_base = self.args.env_name.split("-")[0]
        _ATARI_ENVS = {"pacman", "amidar", "bankheist", "alien"}
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

        vae_encoder = None
        train_vae = True
        if is_atari:
            from policy.layers.building_blocks import CNN, MLP, ConvVAEEncoder
            import os
            import torch

            H, W = self.args.state_dim
            latent_dim = 256

            if self.args.int_reward_type == "allo":
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
                vae_encoder = encoder
                train_vae = False
                print(
                    f"[MAML] Loaded FROZEN ALLO encoder from {allo_encoder_path}"
                )
            else:
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
                        f"[MAML] Pretraining a fresh VAE encoder for "
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
                    print(f"[MAML] cnn_mode={cnn_mode}. CNN encoder is trained end-to-end by RL like PPO. VAE loss is disabled.")

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
            vae_encoder=vae_encoder,
            train_vae=train_vae,
            grad_batch_size=getattr(self.args, "minibatch_size", 256),
        )

        if hasattr(self.env, "get_grid"):
            self.policy.actor.grid = self.env.get_grid()
