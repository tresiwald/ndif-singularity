# NDIF Singularity Container — Qwen3.5

An Apptainer/Singularity definition that runs the upstream `ndif/ndif:latest` Docker image as an HPC-friendly image. It deploys **Qwen/Qwen3.5-4B** and **Qwen/Qwen3.5-2B**, caps the reasoning ("thinking") budget at **20,000 tokens**, generates a **persistent API key** on first run, and uses **all available GPUs and CPUs** for parallelization (Ray picks them up via `RAY_NUM_CPUS`/`RAY_NUM_GPUS`, which the runscript sets from `nproc` and `nvidia-smi`).

## Quick start

```bash
# 1. Build the image (on a Linux host or cluster build node)
singularity build ndif-qwen35.sif Singularity.def

# 2. Start the service — deploys both Qwen3.5 models, uses every GPU/CPU
singularity run --nv \
  --bind "$HOME/.cache/huggingface:$HOME/.cache/huggingface" \
  --bind "$PWD/ray-tmp:/tmp/ndif-ray" \
  ndif-qwen35.sif

# 3. In another shell: get the API key (generated on first run, then stable)
singularity run ndif-qwen35.sif apikey

# 4. Verify the deployment
python test_ndif_qwen.py --api-key "$(singularity run ndif-qwen35.sif apikey)"
```

The key is also printed in the service's startup log (`NDIF API key: ...`) and stored in `$HOME/.ndif/api_key` — `cat` that file works too. To use your own key instead, export `NDIF_API_KEY=...` before step 2.

On a SLURM cluster, run step 2 inside your job allocation (e.g. `srun --gres=gpu:4 ... singularity run --nv ...`); the container automatically sizes Ray to the CPUs/GPUs the job sees.

## Build

```bash
singularity build ndif-qwen35.sif Singularity.def
```

Use `apptainer build ...` if your cluster uses Apptainer instead.

If building Singularity/Apptainer from source complains about missing FUSE headers, install `libfuse-dev libfuse3-dev fuse3 pkg-config` (Debian/Ubuntu) or `fuse-devel fuse3-devel pkgconfig` (RHEL-family). On macOS run it inside a Linux VM.

## Run NDIF

```bash
singularity run --nv \
  --bind "$HOME/.cache/huggingface:$HOME/.cache/huggingface" \
  --bind "$PWD/ray-tmp:/tmp/ndif-ray" \
  ndif-qwen35.sif
```

On startup the runscript prints the deployed models, the detected CPU/GPU counts, and the API key. Defaults set by the container:

```bash
NDIF_SERVICE=all
NDIF_DEV_MODE=false                 # real API key required
NDIF_MODELS=Qwen/Qwen3.5-4B,Qwen/Qwen3.5-2B
NDIF_MAX_THINKING_TOKENS=20000
NDIF_API_KEY_FILE=$HOME/.ndif/api_key
RAY_NUM_CPUS=$(nproc)
RAY_NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
```

Ports match the Docker image: `5001` NDIF API, `27018` MinIO object store, `8265` Ray dashboard.

## API key

A key is generated on first run and stored (mode 600) in `$HOME/.ndif/api_key`. Print or pre-generate it with:

```bash
singularity run ndif-qwen35.sif apikey
```

Override it by exporting `NDIF_API_KEY` before `singularity run`.

## Thinking budget

`NDIF_MAX_THINKING_TOKENS` (default 20000) caps Qwen3.5's reasoning tokens; the smoke loader also writes `max_thinking_tokens` into each model's generation config so all deployments respect it. Override at run time:

```bash
NDIF_MAX_THINKING_TOKENS=10000 singularity run --nv ndif-qwen35.sif
```

## Smoke-load the models

Loads both Qwen models with plain Transformers (no service):

```bash
singularity run --nv ndif-qwen35.sif load
```

For gated models pass `HF_TOKEN` and bind your Hugging Face cache.

## Client checks

Minimal smoke test (single layer-0 trace on the vertical-asymptote test problem):

```bash
python test_ndif_qwen.py --api-key "$(singularity run ndif-qwen35.sif apikey)"
```

Internals + layerwise uncertainty features with nnsight (per-layer hidden states, attention outputs, and logit-lens entropy / confidence / top-1–top-2 margin):

```bash
python example_qwen_internals.py --api-key "$(singularity run ndif-qwen35.sif apikey)"
```

Both default to `Qwen/Qwen3.5-4B`; pass `--model Qwen/Qwen3.5-2B` for the smaller model. The default prompt in both is the test problem:

> Find the number of vertical asymptotes in the graph of
> y = ((x+8)(x+5)²(x+1)³x⁵(x−3)²) / ((x+7)(x+5)²(x+1)x(x−3)³(x−4)).

## Debugging Ray

If Ray fails during startup (`Starting Ray client server failed`), run:

```bash
bash diagnose_ndif.sh
```

or tail `ray_client_server_*.err/out`, `raylet.err/out`, `gcs_server.err/out`, and `dashboard_agent.log` under your bound `ray-tmp` directory.
