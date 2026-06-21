#!/bin/bash
#SBATCH --job-name=pixel_inference
#SBATCH --account=geos_extra
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-8
#SBATCH --output=logs/pixel_inference_%j.log
#SBATCH --error=logs/pixel_inference_%j.err

source /scratch/s1214882/gaza-damage-mapping/alex/bin/activate
cd /scratch/s1214882/gaza-damage-mapping
mkdir -p logs

echo "Starting pixel inference..."
echo "Model: rf_s1_2months_50trees_1x1_all7reducers_baseline"
echo "Start time: $(date)"

python3 src/inference/local_pixel_inference.py

echo "End time: $(date)"
echo "Done."
