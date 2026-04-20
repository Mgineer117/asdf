for script in commands/drnd/*.sbatch; do
    echo "Submitting $script..."
    sbatch "$script"
    sleep 2
done