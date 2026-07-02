# NDIF Singularity Container

This workspace contains a draft Apptainer/Singularity definition for running the upstream `ndif/ndif:latest` Docker image as an HPC-friendly Singularity image. It defaults to loading/deploying the Hugging Face `gpt2` model.

## Build

If you are compiling SingularityCE/Apptainer from source and see:

```text
fuse / fuse3 (libfuse / libfuse3) headers are required to build squashfuse
```

install the FUSE development headers on the host before rerunning the Singularity setup:

```bash
# Debian/Ubuntu
sudo apt-get update
sudo apt-get install -y libfuse-dev libfuse3-dev fuse3 pkg-config

# RHEL/CentOS/Rocky/Alma/Fedora
sudo dnf install -y fuse-devel fuse3-devel pkgconfig
```

On macOS, Singularity/Apptainer is normally run inside a Linux VM because it depends on Linux kernel features.

```bash
apptainer build ndif-gpt2.sif Singularity.def
```

Use `singularity build ...` if your cluster still uses SingularityCE.

## Run NDIF

```bash
apptainer run --nv \
  --bind "$HOME/.cache/huggingface:$HOME/.cache/huggingface" \
  ndif-gpt2.sif
```

The runscript starts the same all-in-one service as the Docker image and sets:

```bash
NDIF_SERVICE=all
NDIF_DEV_MODE=true
HF_MODEL=gpt2
NDIF_DEPLOYMENTS=gpt2
HF_HOME=$HOME/.cache/huggingface
```

The important ports match the Docker image:

```text
5001   NDIF API
27018  MinIO object store
8265   Ray dashboard
```

## Smoke-load GPT-2

To verify the image can load a Hugging Face model without starting the service:

```bash
apptainer run --nv ndif-gpt2.sif load
```

## Use Another Model

```bash
HF_MODEL=openai-community/gpt2 apptainer run --nv ndif-gpt2.sif
```

For gated models, pass `HF_TOKEN` and bind your Hugging Face cache:

```bash
HF_TOKEN=hf_... HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
  apptainer run --nv \
  --bind "$HOME/.cache/huggingface:$HOME/.cache/huggingface" \
  ndif-gpt2.sif
```

## Client Check

Once the service is running:

```python
import nnsight

nnsight.CONFIG.API.HOST = "http://localhost:5001"
nnsight.CONFIG.set_default_api_key("any-key-works-in-dev-mode")

model = nnsight.LanguageModel("gpt2")
with model.trace("Hello world", remote=True):
    h6 = model.transformer.h[6].output[0].save()

print(h6.shape)
```
