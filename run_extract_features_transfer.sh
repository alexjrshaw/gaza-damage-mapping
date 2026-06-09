#!/bin/bash
#SBATCH --job-name=transfer_features
#SBATCH --account=geos_extra
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-4
#SBATCH --output=logs/transfer_features_%j.log
#SBATCH --error=logs/transfer_features_%j.err

source /scratch/s1214882/gaza-damage-mapping/alex/bin/activate
cd /scratch/s1214882/gaza-damage-mapping
mkdir -p logs

echo "Starting transfer city feature extraction..."
echo "Start time: $(date)"

python3 src/data/transfer_cities/extract_features_transfer.py

echo "End time: $(date)"
echo "Done."
