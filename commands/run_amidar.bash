for script in commands/amidar/*.sbatch; do
    echo "Submitting $script..."
    sbatch "$script"
    sleep 2
done