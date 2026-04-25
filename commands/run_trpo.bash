for script in commands/trpo/*.sbatch; do
    echo "Submitting $script..."
    sbatch "$script"
    sleep 2
done