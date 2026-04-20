for script in commands/hrl/*.sbatch; do
    echo "Submitting $script..."
    sbatch "$script"
    sleep 2
done