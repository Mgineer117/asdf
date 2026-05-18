import subprocess
import sys

# Algorithms to test
ALGOS = ["ppo", "irpo", "maml", "hrl", "drnd", "trpo", "psne", "htrpo"]

def run_test(algo):
    """
    Test that the algorithm runs for a tiny number of timesteps without crashing.
    """
    # For HRL / HTRPO / IRPO, some options-based / hierarchical algos might require specific configurations.
    # We will test them on pointmaze-v1 by default since it is continuous and fast.
    env = "pointmaze-v1"
    
    # We run with very few timesteps/minibatch updates just to verify no initialization or runtime crash.
    cmd = [
        sys.executable,
        "main.py",
        "--algo-name", algo,
        "--env-name", env,
        "--timesteps", "1000",
        "--sub-timesteps", "1000",
        "--hl-timesteps", "1000",
        "--batch-size", "128",
        "--minibatch-size", "32",
        "--num-minibatch", "4",
        "--num-runs", "1",
        "--override-results"
    ]
    
    print(f"\n=======================================================")
    print(f" Testing Algorithm: {algo}")
    print(f"=======================================================")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f" [PASS] {algo} completed successfully!")
        return True
    else:
        print(f" [FAIL] {algo} failed with exit code {result.returncode}")
        print("\n--- STDERR ---")
        print(result.stderr)
        print("\n--- STDOUT ---")
        print(result.stdout)
        return False

def main():
    passed = []
    failed = []
    
    for algo in ALGOS:
        success = run_test(algo)
        if success:
            passed.append(algo)
        else:
            failed.append(algo)
            
    print("\n=======================================================")
    print(" SUMMARY")
    print(f"=======================================================")
    print(f" Passed ({len(passed)}): {', '.join(passed)}")
    print(f" Failed ({len(failed)}): {', '.join(failed)}")
    
    if failed:
        sys.exit(1)
    else:
        print(" All algorithms executed successfully without errors!")
        sys.exit(0)

if __name__ == "__main__":
    main()
