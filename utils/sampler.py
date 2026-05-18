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

        # If environment has a GPU-based encoder, move it to CPU temporarily
        # to prevent CUDA re-initialization errors in forked multiprocessing workers
        original_env_device = None
        if hasattr(env, "_encoder") and env._encoder is not None:
            original_env_device = getattr(env, "device", "cpu")
            env._encoder = env._encoder.cpu()
            env.device = "cpu"

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

        # Restore environment encoder to original device
        if original_env_device is not None:
            env.device = original_env_device
            env._encoder = env._encoder.to(original_env_device)

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
        # Surface worker exceptions: without this wrapper the parent waits
        # 20 min on queue.get and the actual traceback is lost.
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
            # Re-raise so the OS records a non-zero exit code; the parent's
            # `queue.get` will still time out, but at least the traceback
            # is in the log.
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


class VectorizedSampler(Base):
    """Sampler using gymnasium AsyncVectorEnv + batched GPU encoding.

    Designed for Atari environments where a pretrained image encoder converts
    raw pixel observations into compact feature vectors.  Instead of spawning
    mp.Process workers that each run a serial encode→policy→step loop, this
    sampler:

    1. Steps N environments in parallel via AsyncVectorEnv (C-level ALE,
       no Python GIL contention).
    2. Batch-encodes the N raw pixel frames in a single GPU forward pass.
    3. Batch-forwards through the policy to select actions for all N envs.

    This eliminates per-frame CPU encoding, mp.Queue serialization overhead,
    and process spawn/join costs.
    """

    def __init__(
        self,
        env_fn: callable,
        encoder: nn.Module,
        is_discrete: bool,
        device: str = "cpu",
        num_envs: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.env_fn = env_fn
        self.device = device
        self.encoder = encoder.to(self.device)
        self.is_discrete = is_discrete
        self.num_envs = num_envs

        self.encoder.eval()

        print("Sampling Parameters (VectorizedSampler):")
        print(f"  Number of parallel envs: {self.num_envs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Encoder device: {self.device}")

    # ------------------------------------------------------------------
    # Batch encoding
    # ------------------------------------------------------------------

    def _batch_encode(self, raw_obs: np.ndarray) -> np.ndarray:
        """Encode (N, H, W) or (N, H, W, C) uint8 pixels → (N, encoder_dim)."""
        if raw_obs.ndim == 3:  # grayscale (N, H, W)
            t = torch.from_numpy(raw_obs.astype(np.float32) / 255.0)
            t = t.unsqueeze(1)  # (N, 1, H, W)
        elif raw_obs.ndim == 4:  # colour (N, H, W, C)
            t = torch.from_numpy(raw_obs.astype(np.float32) / 255.0)
            t = t.permute(0, 3, 1, 2)  # (N, C, H, W)
        else:
            raise ValueError(f"Unexpected raw_obs shape: {raw_obs.shape}")

        with torch.no_grad():
            encoded = self.encoder(t.to(self.device))  # (N, encoder_dim)
        return encoded.cpu().numpy()

    # ------------------------------------------------------------------
    # Main collection loop
    # ------------------------------------------------------------------

    def collect_samples(
        self,
        env,
        policies: list[nn.Module] | nn.Module,
        seed: int | None = None,
        deterministic: bool = False,
    ):
        """Collect a batch of transitions using vectorized envs + batched encoding.

        The ``env`` argument is accepted for API compatibility with
        ``OnlineSampler`` but is **ignored** — we use ``self.env_fn`` to
        construct the internal ``AsyncVectorEnv``.
        """
        from gymnasium.vector import AsyncVectorEnv

        t_start = time.time()

        is_list = isinstance(policies, list)
        policy_list = policies if is_list else [policies]

        # ── Create vectorized env ──────────────────────────────────────────
        _env_fn = self.env_fn
        def _make_env(idx):
            def _init():
                return _env_fn()
            return _init

        vec_env = AsyncVectorEnv([_make_env(i) for i in range(self.num_envs)])
        batches = []

        for p_idx, policy in enumerate(policy_list):
            data = self.get_reset_data(self.batch_size)
            step_count = 0
            ep_steps = np.zeros(self.num_envs, dtype=int)

            seeds = (
                [seed + i + p_idx * self.num_envs for i in range(self.num_envs)]
                if seed is not None
                else None
            )
            raw_obs, _ = vec_env.reset(seed=seeds)  # (N, H, W)

            while step_count < self.batch_size:
                # 1. Batch encode raw pixels → feature vectors
                encoded = self._batch_encode(raw_obs)  # (N, encoder_dim)

                # 2. Batch policy forward
                with torch.no_grad():
                    actions, metaData = policy(encoded, deterministic=deterministic)
                    actions_np = actions.cpu().numpy()  # (N, action_dim)

                # 3. Convert to env actions
                if self.is_discrete:
                    env_actions = np.argmax(actions_np, axis=-1)  # (N,) ints
                else:
                    env_actions = actions_np

                # 4. Step all envs in parallel
                next_raw_obs, rewards, terms, truncs, infos = vec_env.step(env_actions)
                ep_steps += 1

                # 5. Enforce episode length
                for i in range(self.num_envs):
                    if ep_steps[i] >= self.episode_len:
                        truncs[i] = True

                # 6. Store transitions
                n_to_store = min(self.num_envs, self.batch_size - step_count)
                for i in range(n_to_store):
                    data["states"][step_count] = encoded[i]
                    data["actions"][step_count] = actions_np[i]
                    data["rewards"][step_count] = rewards[i]
                    data["terminations"][step_count] = float(terms[i])
                    data["truncations"][step_count] = float(truncs[i])
                    data["logprobs"][step_count] = (
                        metaData["logprobs"][i].cpu().detach().numpy()
                    )
                    data["entropys"][step_count] = (
                        metaData["entropy"][i].cpu().detach().numpy()
                    )
                    step_count += 1

                # 7. Reset ep_steps for terminated/truncated envs
                for i in range(self.num_envs):
                    if terms[i] or truncs[i]:
                        ep_steps[i] = 0

                raw_obs = next_raw_obs
            
            batches.append(data)

        vec_env.close()

        t_end = time.time()
        
        if is_list:
            return batches, t_end - t_start
        else:
            return batches[0], t_end - t_start


def build_sampler(args, verbose: bool = True, **overrides):
    """Factory: choose VectorizedSampler for Atari, OnlineSampler otherwise.

    ``overrides`` can supply e.g. ``batch_size=...`` to override ``args``.
    """
    common = dict(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        episode_len=args.episode_len,
        batch_size=overrides.get("batch_size", args.batch_size),
    )

    if getattr(args, "env_fn", None) is not None:
        return VectorizedSampler(
            env_fn=args.env_fn,
            encoder=args.encoder,
            is_discrete=args.is_discrete,
            device=args.device,
            **common,
        )

    return OnlineSampler(verbose=verbose, **common)



