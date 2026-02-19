#!/bin/bash
#SBATCH --job-name=multinode-test
#SBATCH --partition=gpu
#SBATCH --output=logs/multinode-test-%j.out
#SBATCH --error=logs/multinode-test-%j.err
#SBATCH --nodes=2                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:8                # number of GPUs per node
#SBATCH --time=01:59:00             # maximum execution time (HH:MM:SS)

######################
### Set environment ###
######################
module purge
module load <your_gpu_module>
module load miniconda/3
conda activate <your_conda_env>

export NCCL_SOCKET_IFNAME=<your_nccl_socket_ifname>
export NCCL_IB_HCA=<your_nccl_ib_hca>
export GPUS_PER_NODE=8

# Pre-cache HF assets to avoid download races and stale NFS locks
python -c "
from huggingface_hub import snapshot_download
snapshot_download('google-bert/bert-base-cased')
snapshot_download('nyu-mll/glue', repo_type='dataset')
snapshot_download('evaluate-metric/glue', repo_type='space')
"
find ~/.cache/huggingface -name "*.lock" -delete 2>/dev/null
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --machine_rank \$SLURM_NODEID \
    "
export ACCELERATE_DIR="<your_accelerate_dir>"
export SCRIPT="${ACCELERATE_DIR}/examples/complete_nlp_example.py"
export SCRIPT_ARGS=" \
    --mixed_precision bf16 \
    --checkpointing_steps epoch \
    --output_dir ${ACCELERATE_DIR}/examples/output \
    "
    
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
srun bash -c "$CMD"
