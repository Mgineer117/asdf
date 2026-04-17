from policy.htrpo import HTRPO_Learner
import torch
import torch.nn as nn

from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from policy.trpo import TRPO_Learner
from trainer.onpolicy_trainer import OnPolicyTrainer
from utils.sampler import OnlineSampler


class HTRPO_Algorithm(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(HTRPO_Algorithm, self).__init__()

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

        self.args.nupdates = args.timesteps // args.batch_size

    def begin_training(self):
        # === Define policy === #
        self.define_policy()

        # === Sampler === #
        sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.env.max_steps,
            batch_size=self.args.batch_size,
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
        actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.actor_fc_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            device=self.args.device,
        )
        critic = PPO_Critic(self.args.state_dim, hidden_dim=self.args.critic_fc_dim)

        env_name, _, _ = self.args.env_name.partition("-")
        self.policy = HTRPO_Learner(
            actor=actor,
            critic=critic,
            env_name=env_name,
            is_discrete=self.args.is_discrete,
            nupdates=self.args.nupdates,
            lr=self.args.learning_rate,
            entropy_scaler=self.args.entropy_scaler,
            batch_size=self.args.batch_size,
            target_kl=self.args.target_kl,
            gamma=self.args.gamma,
            gae=self.args.gae,
            device=self.args.device,
        )

        if hasattr(self.env, "get_grid"):
            self.policy.grid = self.env.get_grid()
