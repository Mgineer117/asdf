for script in commands/htrpo/*.sbatch; do
    echo "Submitting $script..."
    sbatch "$script"
    sleep 2
done