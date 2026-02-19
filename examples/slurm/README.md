# Slurm Multi-Node Training with Accelerate

This guide walks through running the [complete_nlp_example.py](../complete_nlp_example.py) (BERT fine-tuning on GLUE/MRPC) across multiple Slurm-managed GPU nodes using `accelerate launch`.

## 1. Environment Setup

```bash
# Load modules (names are cluster-specific)
module purge
module load <your_gpu_module>
module load miniconda/3

# Create and activate a conda environment
conda create -n accel python=3.12 -y
conda activate accel

# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch

# Clone the repo and install with test dependencies
git clone https://github.com/hang-zou/accelerate-hpc-test.git
cd accelerate-hpc-test
pip install -e ".[test_dev]"
```

The `[test_dev]` extra installs everything the NLP example needs: `transformers`, `datasets`, `evaluate`, `scipy`, `scikit-learn`, etc. (see [setup.py](../../setup.py)).

## 2. Pre-cache Hugging Face Assets

Run this **on the login node before submitting**. Compute nodes may lack internet access, and concurrent downloads across ranks cause NFS lock contention and timeouts.

> The job script repeats the `snapshot_download()` calls as a safety net, but pre-caching on the login node is faster and avoids wasting allocation time.

```bash
conda activate accel

python -c "
from huggingface_hub import snapshot_download
snapshot_download('google-bert/bert-base-cased')
snapshot_download('nyu-mll/glue', repo_type='dataset')
snapshot_download('evaluate-metric/glue', repo_type='space')
"

# Clean stale lock files (NFS creates these and doesn't always clean up)
find ~/.cache/huggingface -name "*.lock" -delete 2>/dev/null
```

> **Note:** `huggingface-cli download` may not be available in older `huggingface_hub` versions. The `snapshot_download()` Python API works reliably in all versions.

## 3. Configure `submit_multinode.sh`

Edit [`submit_multinode.sh`](submit_multinode.sh) and replace the placeholder values:

| Placeholder | Description | Example |
|---|---|---|
| `<your_gpu_module>` | Environment module that sets up GPU drivers/libraries | `intel_h200_gpu` |
| `<your_conda_env>` | Name of the conda environment from step 1 | `accel` |
| `<your_nccl_socket_ifname>` | (Optional) Network interface for NCCL inter-node communication (`ip link show` to find it). Try without first. | `ens3f0`, `ib0` |
| `<your_nccl_ib_hca>` | (Optional) InfiniBand HCA device for NCCL (`ibstat` to find it). Try without first. | `mlx5_0` |
| `<your_accelerate_dir>` | Absolute path to the cloned accelerate repo | `/home/user/accelerate` |

Also adjust these SBATCH directives as needed:

```bash
#SBATCH --nodes=2           # number of nodes
#SBATCH --gres=gpu:8        # GPUs per node (must match GPUS_PER_NODE below)
```

And make sure `GPUS_PER_NODE` matches `--gres`:

```bash
export GPUS_PER_NODE=8
```

### How it works

The script uses `srun` to launch one task per node (`--ntasks-per-node=1`). Each task runs `accelerate launch` which spawns `GPUS_PER_NODE` training processes. The `--machine_rank` is set via `$SLURM_NODEID` so each node knows its rank in the cluster. Communication uses the c10d rendezvous backend with the head node IP resolved from `$SLURM_JOB_NODELIST`.

No `accelerate config` YAML is needed — all configuration is passed via CLI arguments to `accelerate launch`.

## 4. Submit and Monitor

All commands below should be run from the `examples/slurm/` directory.

```bash
# Create the logs directory
mkdir -p logs

# Submit the job
sbatch submit_multinode.sh

# Check job status
squeue -u $USER
squeue -j <job_id>

# Watch the output log (once the job starts)
tail -f logs/multinode-test-<job_id>.out

# Check for errors
cat logs/multinode-test-<job_id>.err
```

## 5. Expected Output

The training script runs 3 epochs of BERT fine-tuning on GLUE/MRPC. You should see accuracy and F1 metrics printed after each epoch:

```
epoch 0: {'accuracy': 0.6838235294117647, 'f1': 0.8122270742358079}
epoch 1: {'accuracy': 0.6838235294117647, 'f1': 0.8122270742358079}
epoch 2: {'accuracy': 0.6936274509803921, 'f1': 0.8164464023494861}
```

With `--checkpointing_steps epoch`, checkpoints are saved under `examples/output/epoch_*/`.

## 6. Troubleshooting

### `No such file or directory` for the training script

Double-check `ACCELERATE_DIR` in `submit_multinode.sh`. It must be an **absolute path** with no stray characters (e.g. no accidental `$` prefix).

### `ImportError: ... dependencies['scipy', 'scikit-learn']`

Install the missing packages or reinstall with the test extras:

```bash
pip install -e ".[test_dev]"
```

### `httpx.ReadTimeout` / download failures on compute nodes

All ranks tried to download the same assets simultaneously. Pre-cache everything on the login node first (see step 2), then clean lock files:

```bash
find ~/.cache/huggingface -name "*.lock" -delete 2>/dev/null
```

### `OSError: [Errno 116] Stale file handle` during dataset loading

NFS lock contention between ranks on different nodes. The `FileLock` library doesn't work reliably on NFS. Fix:

```bash
find ~/.cache/huggingface -name "*.lock" -delete 2>/dev/null
```

The `submit_multinode.sh` script already does this automatically before training starts.

### `ValueError: another evaluation module instance is already using the local cache file`

The `evaluate` library's per-process cache files collide during distributed evaluation. The fix is adding `experiment_id` to the `evaluate.load()` call in `complete_nlp_example.py`:

```python
metric = evaluate.load("glue", "mrpc", experiment_id=f"{accelerator.process_index}")
```

> This fix is already applied in this fork.

### NCCL errors (`ncclRemoteError`, `ncclInternalError`)

These are usually a side effect of one rank crashing (e.g. from a download timeout). Fix the root cause (pre-cache assets, clean locks) and the NCCL errors will go away. If they persist, check that `NCCL_SOCKET_IFNAME` and `NCCL_IB_HCA` are set correctly for your cluster's network topology.

## Other Scripts

| Script | Description |
|---|---|
| [`submit_multigpu.sh`](submit_multigpu.sh) | Single-node multi-GPU training |
| [`submit_multicpu.sh`](submit_multicpu.sh) | Multi-CPU training |
| [`submit_multinode_fsdp.sh`](submit_multinode_fsdp.sh) | Multi-node training with FSDP (uses [`fsdp_config.yaml`](fsdp_config.yaml)) |
