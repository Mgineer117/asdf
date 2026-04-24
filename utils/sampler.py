import random
import time
from datetime import date
from math import ceil
from queue import Empty

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

today = date.today()


class Base:
    def __init__(self, **kwargs):
        """
        Base class for the sampler.
        """
        self.state_dim = kwargs.get("state_dim")
        self.action_dim = kwargs.get("action_dim")
        self.episode_len = kwargs.get("episode_len")
        self.batch_size = kwargs.get("batch_size")

    def get_reset_data(self, size: int):
        """
        Pre-allocate arrays to the exact number of samples needed per worker.
        """
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

        torch.set_num_threads(1)  # Avoid CPU oversubscription

    def collect_samples(
        self,
        env,
        policies: list[nn.Module] | nn.Module,
        seed: int | None = None,
        deterministic: bool = False,
    ):
        """
        Collect samples in parallel for multiple policies.
        EACH policy gets self.total_num_worker processes.
        """
        t_start = time.time()
        if not isinstance(policies, list):
            policies = [policies]

        num_policies = len(policies)
        workers_per_policy = self.total_num_worker
        total_expected_workers = num_policies * workers_per_policy

        # Determine original devices to restore later
        original_devices = [p.device for p in policies]

        # Move all policies to CPU for multiprocessing pickling
        for p in policies:
            p.to_device(torch.device("cpu"))

        processes = []
        queue = mp.Queue()

        # Storage indexed by [policy_idx][worker_in_policy_idx]
        worker_memories = [None] * total_expected_workers

        # ✅ Spawn total_num_worker for EACH policy
        for p_idx in range(num_policies):
            policy = policies[p_idx]

            for w_idx in range(workers_per_policy):
                # Unique global ID for this specific worker-policy pair
                global_worker_id = p_idx * workers_per_policy + w_idx

                # Diverse seeding: Ensure no two workers share a seed
                worker_seed = seed + global_worker_id if seed is not None else None

                args = (
                    global_worker_id,
                    queue,
                    env,
                    policy,
                    worker_seed,
                    deterministic,
                )
                p = mp.Process(target=self.collect_trajectory, args=args)
                processes.append(p)
                p.start()

        # ✅ Collect from Queue
        collected = 0
        while collected < total_expected_workers:
            try:
                # Target collect_trajectory must return (global_worker_id, data)
                g_id, data = queue.get(timeout=1200)
                if worker_memories[g_id] is None:
                    worker_memories[g_id] = data
                    collected += 1
            except Empty:
                print(
                    f"[Warning] Queue timeout. {collected}/{total_expected_workers} collected."
                )

        # ✅ Cleanup processes
        start_time = time.time()
        for p in processes:
            p.join(timeout=max(0.1, 30 - (time.time() - start_time)))
            if p.is_alive():
                p.terminate()
                p.join()
            p.close()

        # ✅ Close the queue so its feeder thread / shared-memory handles are released
        try:
            queue.close()
            queue.join_thread()
        except Exception:
            pass

        # ✅ Merge memory PER POLICY
        policy_memories = [{} for _ in range(num_policies)]

        for g_id, wm in enumerate(worker_memories):
            if wm is None:
                raise RuntimeError(f"Global worker {g_id} failed to return data.")

            # Map the global ID back to which policy it belonged to
            p_idx = g_id // workers_per_policy
            target_mem = policy_memories[p_idx]

            for key, val in wm.items():
                if key in target_mem:
                    target_mem[key] = np.concatenate((target_mem[key], val), axis=0)
                else:
                    target_mem[key] = val

        t_end = time.time()

        # Restore policies to original devices
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
        # assign per-worker seed
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

        # Continuously sample until the exact worker_batch_size is met
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

            # If episode ends early, reset and keep going to reach worker_batch_size
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
