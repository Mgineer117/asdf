import glob
import json

for algo_file in glob.glob("config/algos/*.json"):
    with open(algo_file, "r") as f:
        data = json.load(f)
    if "learning_rate" in data:
        data["atari_learning_rate"] = data["learning_rate"] / 20.0
    with open(algo_file, "w") as f:
        json.dump(data, f, indent=4)

print("Done updating atari_learning_rate to 1/20 of learning_rate")
