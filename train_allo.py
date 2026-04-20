import datetime
import random
import uuid

import torch

from utils.functions import setup_logger
from utils.get_args import get_args, override_args
from utils.intrinsic_rewards import ALLOIntRewardFunctions

if __name__ == "__main__":

    torch.set_default_dtype(torch.float32)

    init_args = get_args()
    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")

    seeds = [1825, 410, 4507, 4013, 3658, 2287, 1680, 8936, 1425, 9675]
    print(f"-------------------------------------------------------")
    print(f"      Running ID: {unique_id}")
    print(f"      Running Seeds: {seeds}")
    print(f"      Time Begun   : {exp_time}")
    print(f"-------------------------------------------------------")

    for seed in seeds:
        args = override_args(init_args)
        args.seed = seed
        args.unique_id = unique_id
        args.algo_name = "irpo"
        args.int_reward_type = "allo"

        logger, writer = setup_logger(args, unique_id, exp_time, seed, False)

        irf = ALLOIntRewardFunctions(logger=logger, writer=writer, args=args)
