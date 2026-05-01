import datetime
import os
import random
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
    return os.path.join("results", args.env_name, args.algo_name, f"{seed}.csv")


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


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)

    init_args = get_args()
    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")

    # random.seed(init_args.seed)
    seed_pool = [1825, 410, 4507, 4013, 3658, 2287, 1680, 8936, 1425, 9675]
    if init_args.num_runs > len(seed_pool):
        seeds = seed_pool
        print(
            "[Warning] num_runs exceeds the length of seed_pool. Some seeds will be reused."
        )
    else:
        seeds = random.sample(seed_pool, init_args.num_runs)

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
