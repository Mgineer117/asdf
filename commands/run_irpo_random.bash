for script in commands/irpo_random/*.sbatch; do
    echo "Submitting $script..."
    sbatch "$script"
    sleep 2
done