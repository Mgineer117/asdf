for script in commands/pacman/*.sbatch; do
    echo "Submitting $script..."
    sbatch "$script"
    sleep 2
done