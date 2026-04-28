#!/bin/bash
#SBATCH --job-name=gaza_features_splits
#SBATCH --account=geos_extra
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=7-0
#SBATCH --output=features_splits_log.txt
#SBATCH --error=features_splits_err.txt

cd /scratch/s1214882/gaza-damage-mapping
source alex/bin/activate

echo "=============================="
echo "Running random_all split..."
echo "=============================="
python3 src/data/sentinel1/extract_features_splits.py random_all

echo "=============================="
echo "Running random_per_aoi split..."
echo "=============================="
python3 src/data/sentinel1/extract_features_splits.py random_per_aoi

echo "All splits complete."
