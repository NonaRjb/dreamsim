#!/usr/bin/env bash
#SBATCH -A berzelius-2024-324
#SBATCH --mem 10GB
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00
#SBATCH --mail-type FAIL
#SBATCH --mail-user nonar@kth.se
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/dreamsim/logs/%J_slurm.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/dreamsim/logs/%J_slurm.err

data_path="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_meg/images"
save_path="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_meg/image_embeddings"
model_type="ensemble"  # "ensemble", "dino_vitb16", "clip_vitb32", "open_clip_vitb32", "dinov2_vitb14", and "synclr_vitb16"

cd /proj/rep-learning-robotics/users/x_nonra/dreamsim/

# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate ds
export PYTHONPATH="$PYTHONPATH:$(realpath ./dreamsim)"

python scripts/generate_things_meg_img_embeddings.py --data_path "$data_path" --save_path "$save_path" --model_type "$model_type" --model_name dreamsim
