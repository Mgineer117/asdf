EPI_LENGTH = {
    "fourrooms-v1": 50,
    "fourrooms-v2": 50,
    "maze-v1": 100,
    "maze-v2": 100,
    "pointmaze-v1": 500,
    "pointmaze-v2": 500,
    "pointmaze-v3": 500,
    "pointmaze-v4": 500,
    "antmaze-v1": 1000,
    "antmaze-v2": 1000,
    "antmaze-v3": 1000,
    "antmaze-v4": 1000,
    "antmaze-v5": 1000,
    "fetchreach": 50,
    "fetchpush": 50,
    "pacman": 1000,
    "ant": 1000,
    "walker": 1000,
    "hopper": 1000,
    "halfcheetah": 1000,
}

POS_IDX = {
    "fourrooms": [0, 1],
    "maze": [0, 1],
    "pointmaze": [-4, -3],
    "antmaze": [-4, -3],
    "fetchreach": [-6, -5, -4],
    "pacman": None,
    "ant": [14],
    "walker": [9],
    "hopper": [6],
    "halfcheetah": [9],
}

GOAL_IDX = {
    "fourrooms": [2, 3],
    "maze": [2, 3],
    "pointmaze": [-2, -1],
    "antmaze": [-2, -1],
    "fetchreach": [-3, -2, -1],
    "pacman": None,
    "ant": [14],
    "walker": [9],
    "hopper": [6],
    "halfcheetah": [9],
}


POINTMAZE_MAPS = {
    "pointmaze-v1": [
        [1, 1, 1, 1, 1, 1],
        [1, "c", 1, "c", 0, 1],
        [1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ],
    "pointmaze-v2": [
        [1, 1, 1, 1, 1, 1],
        [1, "c", 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 1],
        [1, "c", 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ],
    "pointmaze-v3": [
        [1, 1, 1, 1, 1, 1],
        [1, "c", 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 1],
        [1, "c", 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 1],
        [1, "c", 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ],
    "pointmaze-v4": [
        [1, 1, 1, 1, 1, 1],
        [1, 0, "c", 1, "c", 1],
        [1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1],
        [1, "c", 1, "c", 0, 1],
        [1, 1, 1, 1, 1, 1],
    ],
}


ANTMAZE_MAPS = {
    "antmaze-v1": [
        [1, 1, 1, 1, 1],
        [1, "c", 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, "c", 0, "c", 1],
        [1, 1, 1, 1, 1],
    ],
    "antmaze-v2": [
        [1, 1, 1, 1, 1],
        [1, "c", 1, "c", 1],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, "c", 0, 1],
        [1, 1, 1, 1, 1],
    ],
    "antmaze-v3": [
        [1, 1, 1, 1, 1],
        [1, "c", 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, "c", 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, "c", 1],
        [1, 1, 1, 1, 1],
    ],
}
