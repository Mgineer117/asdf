for script in commands/irpo/*.sbatch; do
    echo "Submitting $script..."
    sbatch "$script"
    sleep 2
done