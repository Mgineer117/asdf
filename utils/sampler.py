import random
import time
from math import ceil
from queue import Empty

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn


class Base:
    def __init__(self, **kwargs):
        self.state_dim = kwargs.get("state_dim")
        self.action_dim = kwargs.get("action_dim")
        self.episode_len = kwargs.get("episode_len")
        self.batch_size = kwargs.get("batch_size")

    def get_reset_data(self, size: int):
        """Pre-allocate arrays for one worker's worth of samples."""
        data = dict(
            states=np.zeros(((size,) + self.state_dim), dtype=np.float32),
            next_states=np.zeros(((size,) + self.state_dim), dtype=np.float32),
            actions=np.zeros((size, self.action_dim), dtype=np.float32),
            rewards=np.zeros((size, 1), dtype=np.float32),
            terminations=np.zeros((size, 1), dtype=np.float32),
            truncations=np.zeros((size, 1), dtype=np.float32),
            logprobs=np.zeros((size, 1), dtype=np.float32),
            entropys=np.zeros((size, 1), dtype=np.float32),
        )
        return data


class OnlineSampler(Base):
    def __init__(
        self,
        state_dim: tuple,
        action_dim: int,
        episode_len: int,
        batch_size: int,
        num_workers: int = 4,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            episode_len=episode_len,
            batch_size=batch_size,
        )

        self.total_num_worker = num_workers
        self.worker_batch_size = ceil(batch_size / self.total_num_worker)

        if verbose:
            print("Sampling Parameters:")
            print(f"Total number of workers per policy: {self.total_num_worker}")
            print(f"Target samples per worker: {self.worker_batch_size}")

        torch.set_num_threads(1)

    def collect_samples(
        self,
        env,
        policies: list[nn.Module] | nn.Module,
        seed: int | None = None,
        deterministic: bool = False,
    ):
        """
        Collect on-policy samples in parallel for one or more policies.
        Each policy gets self.total_num_worker worker processes.
        Workers store raw states (no encoding); the policy's CNN runs in-process.
        """
        t_start = time.time()
        if not isinstance(policies, list):
            policies = [policies]

        num_policies = len(policies)
        workers_per_policy = self.total_num_worker
        total_expected_workers = num_policies * workers_per_policy

        # Move all policies to CPU before forking (CUDA can't be re-initialised in
        # child processes spawned by fork).
        original_devices = [p.device for p in policies]
        for p in policies:
            p.to_device(torch.device("cpu"))

        processes = []
        queue = mp.Queue()
        worker_memories = [None] * total_expected_workers

        for p_idx in range(num_policies):
            policy = policies[p_idx]
            for w_idx in range(workers_per_policy):
                global_worker_id = p_idx * workers_per_policy + w_idx
                worker_seed = seed + global_worker_id if seed is not None else None
                args = (global_worker_id, queue, env, policy, worker_seed, deterministic)
                p = mp.Process(target=self.collect_trajectory, args=args)
                processes.append(p)
                p.start()

        collected = 0
        while collected < total_expected_workers:
            try:
                g_id, data = queue.get(timeout=1200)
                if worker_memories[g_id] is None:
                    worker_memories[g_id] = data
                    collected += 1
            except Empty:
                print(
                    f"[Warning] Queue timeout. {collected}/{total_expected_workers} collected."
                )

        start_time = time.time()
        for p in processes:
            p.join(timeout=max(0.1, 30 - (time.time() - start_time)))
            if p.is_alive():
                p.terminate()
                p.join()
            p.close()

        try:
            queue.close()
            queue.join_thread()
        except Exception:
            pass

        # Merge per-worker data back into per-policy batches
        policy_memories = [{} for _ in range(num_policies)]
        for g_id, wm in enumerate(worker_memories):
            if wm is None:
                raise RuntimeError(f"Global worker {g_id} failed to return data.")
            p_idx = g_id // workers_per_policy
            target_mem = policy_memories[p_idx]
            for key, val in wm.items():
                if key in target_mem:
                    target_mem[key] = np.concatenate((target_mem[key], val), axis=0)
                else:
                    target_mem[key] = val

        t_end = time.time()

        # Restore policies to their original devices
        for p, device in zip(policies, original_devices):
            p.to_device(device)

        if num_policies == 1:
            policy_memories = policy_memories[0]

        return policy_memories, t_end - t_start

    def collect_trajectory(
        self,
        pid,
        queue,
        env,
        policy: nn.Module,
        seed: int,
        deterministic: bool = False,
    ):
        try:
            self._collect_trajectory_impl(pid, queue, env, policy, seed, deterministic)
        except BaseException as e:
            import sys
            import traceback
            sys.stderr.write(
                f"[sampler worker {pid}] CRASHED: {type(e).__name__}: {e}\n"
            )
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise

    def _collect_trajectory_impl(
        self,
        pid,
        queue,
        env,
        policy: nn.Module,
        seed: int,
        deterministic: bool = False,
    ):
        worker_seed = random.randint(0, 10000) + seed + pid
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)

        data = self.get_reset_data(self.worker_batch_size)
        step_count = 0
        ep_step = 0

        state, _ = env.reset(seed=worker_seed)

        while step_count < self.worker_batch_size:
            with torch.no_grad():
                a, metaData = policy(state, deterministic=deterministic)
                a = a.cpu().numpy().flatten()

            next_state, rew, term, trunc, _ = env.step(a)
            ep_step += 1

            if ep_step >= self.episode_len:
                trunc = True

            done = term or trunc

            data["states"][step_count] = state
            data["next_states"][step_count] = next_state
            data["actions"][step_count] = a
            data["rewards"][step_count] = rew
            data["terminations"][step_count] = term
            data["truncations"][step_count] = trunc
            data["logprobs"][step_count] = metaData["logprobs"].cpu().detach().numpy()
            data["entropys"][step_count] = metaData["entropy"].cpu().detach().numpy()

            step_count += 1

            if done:
                state, _ = env.reset(seed=worker_seed)
                ep_step = 0
            else:
                state = next_state

        if queue is not None:
            queue.put([pid, data])
        else:
            return data


class HLSampler(OnlineSampler):
    def __init__(
        self,
        state_dim: tuple,
        action_dim: int,
        episode_len: int,
        batch_size: int,
        max_option_len: int,
        gamma: float,
        num_workers: int = 4,
        verbose: bool = True,
    ) -> None:
        self.max_option_len = max_option_len
        self.gamma = gamma

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            episode_len=episode_len,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose,
        )

    def collect_trajectory(
        self,
        pid,
        queue,
        env,
        policy: nn.Module,
        seed: int | None = None,
        deterministic: bool = False,
    ):
        worker_seed = random.randint(0, 10000) + seed + pid
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)

        data = self.get_reset_data(self.worker_batch_size)
        step_count = 0
        ep_step = 0

        state, _ = env.reset(seed=worker_seed)

        while step_count < self.worker_batch_size:
            with torch.no_grad():
                [option_idx, a], metaData = policy(
                    state, option_idx=None, deterministic=deterministic
                )
                a = a.cpu().numpy().flatten()

            if metaData["is_option"]:
                r = 0
                option_termination = False

                for i in range(self.max_option_len):
                    next_state, rew, term, trunc, _ = env.step(a)
                    ep_step += 1

                    if ep_step >= self.episode_len:
                        trunc = True
                    done = term or trunc

                    r += (self.gamma**i) * rew

                    if done or option_termination:
                        rew = r
                        break
                    else:
                        with torch.no_grad():
                            [_, a], optionMetaData = policy(
                                next_state,
                                option_idx=option_idx,
                                deterministic=deterministic,
                            )
                            a = a.cpu().numpy().flatten()
                        option_termination = optionMetaData["option_termination"]

            else:
                next_state, rew, term, trunc, _ = env.step(a)
                ep_step += 1
                if ep_step >= self.episode_len:
                    trunc = True
                done = term or trunc

            data["states"][step_count] = state
            data["next_states"][step_count] = next_state
            data["actions"][step_count] = metaData["logits"]
            data["rewards"][step_count] = rew
            data["terminations"][step_count] = term
            data["truncations"][step_count] = trunc
            data["logprobs"][step_count] = metaData["logprobs"].cpu().detach().numpy()
            data["entropys"][step_count] = metaData["entropy"].cpu().detach().numpy()

            step_count += 1

            if done:
                state, _ = env.reset(seed=worker_seed)
                ep_step = 0
            else:
                state = next_state

        if queue is not None:
            queue.put([pid, data])
        else:
            return data


def build_sampler(args, verbose: bool = True, **overrides):
    return OnlineSampler(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        episode_len=args.episode_len,
        batch_size=overrides.get("batch_size", args.batch_size),
        verbose=verbose,
    )
