#!/bin/bash
#SBATCH --job-name=ablation_studies
#SBATCH --account=geos_extra
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-0
#SBATCH --output=logs/ablation_studies_%j.log
#SBATCH --error=logs/ablation_studies_%j.err

source /scratch/s1214882/gaza-damage-mapping/alex/bin/activate
cd /scratch/s1214882/gaza-damage-mapping
mkdir -p logs

echo "Starting ablation studies..."
echo "Start time: $(date)"

python3 src/classification/ablation_studies.py

echo "End time: $(date)"
echo "Done."
