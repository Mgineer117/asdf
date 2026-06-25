import datetime
import glob
import os
import shutil
import uuid

import torch

import wandb
from utils.functions import concat_csv_columnwise_and_delete, seed_all, setup_logger
from utils.get_args import get_args, override_args
from utils.get_envs import get_env

# it suppresses the wandb printing when it logs data
os.environ["WANDB_SILENT"] = "true"


def get_results_csv_path(args, seed):
    algo_dir = args.algo_name
    if args.algo_name == "irpo":
        int_reward_type = getattr(args, "int_reward_type", None)
        if int_reward_type == "random":
            algo_dir = "irpo_random"
    return os.path.join("results", args.env_name, algo_dir, f"{seed}.csv")


def find_latest_checkpoint(log_base_dir: str, env_name: str, algo_name: str, seed: int) -> str | None:
    """Search for the most-recently modified latest_full.pt checkpoint for this seed.

    The log directory layout is:
        log/train_log/<group>/<run_name>/checkpoint/latest_full.pt
    where <run_name> contains algo_name, env_name, and seed.
    We glob for all matching files and return the newest one.
    """
    pattern = os.path.join(log_base_dir, "**", "checkpoint", "latest_full.pt")
    candidates = glob.glob(pattern, recursive=True)

    seed_tag = f"seed:{seed}"
    matched = [p for p in candidates if algo_name in p and env_name in p and seed_tag in p]

    if not matched:
        return None
    # Return the most recently modified one
    return max(matched, key=os.path.getmtime)


def load_checkpoint_into_algo(algo, ckpt_path: str, device) -> int:
    """Load latest_full.pt into algo.policy and return the saved step count."""
    # weights_only=False: our own full-state checkpoint carries optimizer state
    # and numpy reward-normalizer stats, not just tensors, so PyTorch >=2.6's
    # weights_only=True default would reject it. Trusted (self-produced) source.
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_step = checkpoint.get("step", 0)
    policy_state = checkpoint.get("policy_state", {})

    try:
        # Restore the FULL training state (weights + critic optimizers + perf-gain
        # EMA + reward normalizer) so training continues rather than restarting
        # those from scratch. Falls back to a bare state_dict for legacy ckpts.
        if hasattr(algo.policy, "load_training_state"):
            algo.policy.load_training_state(policy_state)
        else:
            algo.policy.load_state_dict(policy_state, strict=False)
        print(f"[Resume] Loaded checkpoint from {ckpt_path} at step={saved_step}")
    except Exception as e:
        print(f"[Resume] WARNING: Could not fully load checkpoint state: {e}")

    return saved_step


def run(args, seed, unique_id, exp_time, result_csv_path=None, is_sweep=False):
    # fix seed
    seed_all(seed)

    # get env
    env = get_env(args)
    logger, writer = setup_logger(args, unique_id, exp_time, seed, is_sweep)

    # run algorithm
    if args.algo_name == "ppo":
        from algorithms.ppo import PPO_Algorithm

        algo = PPO_Algorithm(env=env, logger=logger, writer=writer, args=args)
    elif args.algo_name == "irpo":
        from algorithms.irpo import IRPO_Algorithm

        algo = IRPO_Algorithm(env=env, logger=logger, writer=writer, args=args)
    elif args.algo_name == "maml":
        from algorithms.maml import MAML_Algorithm

        algo = MAML_Algorithm(env=env, logger=logger, writer=writer, args=args)
    elif args.algo_name == "hrl":
        from algorithms.hrl import HRL

        algo = HRL(env=env, logger=logger, writer=writer, args=args)
    elif args.algo_name == "drnd":
        from algorithms.drnd import DRND_Algorithm

        algo = DRND_Algorithm(env=env, logger=logger, writer=writer, args=args)
    elif args.algo_name == "trpo":
        from algorithms.trpo import TRPO_Algorithm

        algo = TRPO_Algorithm(env=env, logger=logger, writer=writer, args=args)
    elif args.algo_name == "psne":
        from algorithms.psne import PSNE_Algorithm

        algo = PSNE_Algorithm(env=env, logger=logger, writer=writer, args=args)
    elif args.algo_name == "htrpo":
        from algorithms.htrpo import HTRPO_Algorithm

        algo = HTRPO_Algorithm(env=env, logger=logger, writer=writer, args=args)
    else:
        raise NotImplementedError(f"{args.algo_name} is not implemented.")

    # --- SLURM job-chain resume: load checkpoint if --resume-run was passed ---
    resume_run = getattr(args, "resume_run", False)
    if resume_run:
        algo.begin_training = _make_resume_begin_training(algo, args, seed)

    perf = algo.begin_training()

    if result_csv_path is not None and getattr(logger, "csv_file", None):
        os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)
        if os.path.exists(logger.csv_file):
            shutil.copy2(logger.csv_file, result_csv_path)

    # ✅ Memory cleanup
    del algo, env, logger, writer  # delete large references
    torch.cuda.empty_cache()  # release unreferenced GPU memory
    if not is_sweep:
        wandb.finish()  # end the sweep run to free up resources

    return perf


def _make_resume_begin_training(algo, args, seed):
    """Monkey-patch define_policy / define_base_policy so that after the policy
    is constructed, the checkpoint is loaded into it.  Then return the original
    begin_training — its own internal call to define_policy will trigger our
    patched version, avoiding the double-call bug."""

    # Identify which method creates the policy
    if hasattr(algo, "define_base_policy"):
        method_name = "define_base_policy"
    elif hasattr(algo, "define_policy"):
        method_name = "define_policy"
    else:
        print("[Resume] WARNING: algo has no define_policy; skipping checkpoint load.")
        return algo.begin_training

    original_define = getattr(algo, method_name)

    def _define_and_load():
        # 1. Run the original define to build algo.policy
        original_define()

        # 2. Now algo.policy exists — load checkpoint into it
        log_base = getattr(args, "logdir", "log/train_log")
        log_train = os.path.dirname(log_base)  # up to log/train_log
        ckpt_path = find_latest_checkpoint(
            log_train, args.env_name, args.algo_name, seed
        )
        if ckpt_path:
            saved_step = load_checkpoint_into_algo(algo, ckpt_path, args.device)
            args.resume_init_timesteps = saved_step
        else:
            print(
                f"[Resume] No checkpoint found under {log_train} for "
                f"{args.env_name}/{args.algo_name}/seed:{seed}. Starting fresh."
            )
            args.resume_init_timesteps = 0

    setattr(algo, method_name, _define_and_load)
    # Return the ORIGINAL begin_training — it will call our patched define method
    return algo.begin_training



if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    import torch.multiprocessing as mp
    mp.set_start_method("fork", force=True)

    init_args = get_args()
    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")

    # random.seed(init_args.seed)
    seed_pool = [1825, 410, 4507, 4013, 3658, 2287, 1680, 8936, 1425, 9675]
    # [410,4507,]
    if init_args.num_runs > len(seed_pool):
        seeds = seed_pool
        print(
            "[Warning] num_runs exceeds the length of seed_pool. Some seeds will be reused."
        )
    else:
        # random.sample(seed_pool, init_args.num_runs)
        if init_args.num_runs == 1:
            seeds = [seed_pool[init_args.seed]]
        else:
            seeds = seed_pool[: init_args.num_runs]

    print(f"-------------------------------------------------------")
    print(f"      Running ID: {unique_id}")
    print(f"      Running Seeds: {seeds}")
    print(f"      Time Begun   : {exp_time}")
    print(f"-------------------------------------------------------")

    for seed in seeds:
        args = override_args(init_args)
        args.seed = seed
        args.unique_id = unique_id

        result_csv_path = get_results_csv_path(args, seed)
        if not args.override_results and os.path.exists(result_csv_path):
            print(
                f"[Skip] Results already exist for {args.env_name}/{args.algo_name}/seed {seed}."
            )
            continue

        run(
            args,
            seed,
            unique_id,
            exp_time,
            result_csv_path=result_csv_path,
        )

    concat_csv_columnwise_and_delete(folder_path=args.logdir)

