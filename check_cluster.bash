#!/bin/bash

# --- Job Requirements ---
# Lowered CPUs per task from 8 to 2 to get you scheduled much faster!
# Lowered time from 3 days to 2 days so eng-research-gpu will accept it.
REQ_NODES=1
REQ_TASKS=6
REQ_CPUS_PER_TASK=4
REQ_GPUS=1
REQ_TIME="2-00:00:00"
TOTAL_CPUS=$((REQ_TASKS * REQ_CPUS_PER_TASK))

PARTITIONS=("IllinoisComputes-GPU" "eng-research-gpu" "csl")

echo "================================================================="
echo "📊 SLURM Partition Availability & Wait Time Estimator"
echo "================================================================="
echo "Job Profile: $REQ_NODES Node | $TOTAL_CPUS Total CPUs | $REQ_GPUS GPU | Time: $REQ_TIME"
echo "-----------------------------------------------------------------"

for p in "${PARTITIONS[@]}"; do
    echo "Checking Partition: $p..."

    # 1. Check current queue
    PENDING=$(squeue -p "$p" -t PD -h | wc -l)
    RUNNING=$(squeue -p "$p" -t R -h | wc -l)
    
    # 2. Check idle nodes
    IDLE_NODES=$(sinfo -p "$p" -h -t idle -O nodes | tr -d ' ' || echo "0")
    MIXED_NODES=$(sinfo -p "$p" -h -t mixed -O nodes | tr -d ' ' || echo "0")
    IDLE_NODES=${IDLE_NODES:-0}
    MIXED_NODES=${MIXED_NODES:-0}

    echo "  -> Queue: $RUNNING running, $PENDING pending."
    echo "  -> Nodes: $IDLE_NODES idle, $MIXED_NODES partially used."

    # 3. Create a temporary dummy script to get scheduler estimates
    # NOTE: Your account 'huytran1-ic' is now included!
    DUMMY_SCRIPT=".dummy_test_${p}.sh"
    cat <<EOF > "$DUMMY_SCRIPT"
#!/bin/bash
#SBATCH --account=huytran1-ic
#SBATCH --partition=$p
#SBATCH --nodes=$REQ_NODES
#SBATCH --ntasks=$REQ_TASKS
#SBATCH --cpus-per-task=$REQ_CPUS_PER_TASK
#SBATCH --gres=gpu:$REQ_GPUS
#SBATCH --time=$REQ_TIME
sleep 1
EOF

    # 4. Run test-only (sbatch outputs to stderr, so we redirect it 2>&1)
    TEST_OUT=$(sbatch --test-only "$DUMMY_SCRIPT" 2>&1)

    if [[ "$TEST_OUT" == *"error"* ]] || [[ "$TEST_OUT" == *"Invalid"* ]] || [[ "$TEST_OUT" == *"exceeds"* ]]; then
        echo -e "  -> Start Time: \033[0;31m❌ Ineligible\033[0m"
        echo "     Reason: $TEST_OUT"
    else
        # Extract the date/time string from the sbatch output
        START_TIME=$(echo "$TEST_OUT" | awk -F 'start at ' '{print $2}' | awk '{print $1}')
        if [ -z "$START_TIME" ]; then
            echo -e "  -> Start Time: \033[0;32m✅ Immediate (or check output: $TEST_OUT)\033[0m"
        else
            echo -e "  -> Start Time: \033[0;33m🕒 $START_TIME\033[0m"
        fi
    fi

    # Clean up dummy script
    rm -f "$DUMMY_SCRIPT"
    echo "-----------------------------------------------------------------"
done

echo "💡 Tip: Choose the partition with the earliest Start Time."
echo "================================================================="