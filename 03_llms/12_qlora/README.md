# QLoRA vs LoRA: what the GPU actually holds

This lab measures the infrastructure cost of keeping Qwen3-4B's **frozen**
base weights in bf16 (LoRA) or NF4 four-bit form (QLoRA). The training task,
adapter targets, optimizer, and ZeRO stage stay the same. This is a memory
comparison, not an evaluation of answer quality.

## What this demonstrates

DeepSpeed **ZeRO-2** shards gradients and optimizer state across ranks but
replicates frozen base weights. That makes the difference in base storage
visible per GPU. Do not switch the quantized arm to ZeRO-3: the version of
Transformers pinned here may skip ZeRO-3 parameter partitioning for quantized
weights, even when training exits successfully.

Both arms train rank-8 adapters on seven named Qwen linear projections using
the same four repeatable text samples. A repeated tiny training set gives a
controlled learning signal; it does not establish downstream model quality.
The measured peak includes model loading, training activations, adapter
optimizer state, and CUDA allocation. Measurements are **per GPU**, never
the sum over cards.

## Hardware requirements

| Resource | Starting budget | Notes |
|---|---:|---|
| GPU VRAM | 24 GB per GPU | Estimate only; verify on your actual card |
| GPUs | 1 | More ranks are optional, each still holds the base |
| Disk | 60 GB | Model weights, cache and environment |
| Host RAM | 48 GB | Loading and package installation |

An idealized 4.02B parameter base is about 8.04 GB in bf16 or 2.01 GB at
four bits; NF4 scales, unquantized layers, adapters and activations add to
the actual footprint. Full Adam fine-tuning of 4.02B weights alone would
need about 32 GB for two fp32 optimizer moments, before weights/gradients.
These are arithmetic estimates, **not measured output**.

## Environment & local testing

From this folder, on a machine with an appropriate CUDA GPU:

```bash
uv sync
uv run --locked python -c "import torch, deepspeed, peft, bitsandbytes"
uv run --locked deepspeed --num_gpus=1 train_qlora.py --max-steps 20
```

The committed `uv.lock` and explicit cu128 PyTorch index are part of the
example. Qwen3-4B weights download from Hugging Face on first use. No API
credentials are required for the public model. If your machine has no CUDA
GPU, you can still inspect the weight arithmetic and run the logic suite:

```bash
uv run --no-project python 03_llms/12_qlora/train_qlora.py --plan
uv run tests/test_qlora.py
./tests/run_all.sh
```

The last three commands are run from the **repository root**. A training
entry point without CUDA exits nonzero with a clear explanation.
`ALLOW_CPU=1` bypasses the preflight for debugging, but this 4B model
still cannot train on CPU. The `--plan` mode runs without torch.

## Running it

### CoreWeave / other SLURM clusters

On the login node, run `uv sync` in this folder once. Ensure the model
cache is available to compute nodes; set `HF_HOME` to a scratch directory
you own. Adjust the partition if necessary (`sinfo`). Then:

```bash
mkdir -p logs
sbatch run_deepspeed.sh --max-steps 20   # smoke test; cap per arm
sbatch run_deepspeed.sh                  # 40 optimizer steps per arm
squeue -u "$USER"
tail -f logs/qlora_<jobid>.out
scancel <jobid>                          # if you need to stop the job
```

### RunPod

From the repository root, inspect price and availability, launch with
automatic termination, and confirm no pod remains:

```bash
export RUNPOD_API_KEY=...
uv run runpod/runpod_ctl.py recommend 03_llms/12_qlora
uv run runpod/runpod_ctl.py run 03_llms/12_qlora \
    --dry-run --collect --wait --terminate --yes
uv run runpod/runpod_ctl.py pods
```

The controller's `--dry-run` imposes a wall-clock limit, including any
first model download; it may end before any optimizer step. To measure
both arms, use a normal run after checking the pod setup and set a suitable
`--wait-seconds`. `--volume` sizes the `/workspace` model cache; `--disk`
sizes a different container filesystem. Check the hourly cost before
renting hardware.

### Direct

```bash
uv run --locked deepspeed --num_gpus=1 train_qlora.py --max-steps 20
```

`--max-steps` counts optimizer steps **per arm**, not epochs. The default
40 steps per arm is still only a short demonstration. An uncapped claim
about convergence or model quality is not warranted.

## Reading the results

On rank 0, `comparison.json` holds a separate record for each GPU rank
and arm. `peak_allocated_bytes` is CUDA allocated memory after resetting
the peak before each arm, including weight loading; it differs from reserved
memory and system-level VRAM. `stored_tensor_bytes` sums parameter storage,
buffers and NF4 quantization state; it is not the same as the CUDA peak.
`logical_parameters` recovers original NF4 shapes from `quant_state.shape`,
because packed uint8 `numel()` would undercount. `trainable_parameters`
counts adapters. The first/last loss and five-step window averages describe
the tiny repeated corpus, not generalization.

The harness **raises** if both arms have the same representation, adapter
count or logical count, or if the QLoRA stored bytes / measured peak are
not smaller on an individual GPU. A capped run is labeled as capped.
Absolute GB values depend on card, driver, and libraries, so none are
invented here. A GPU run is needed to validate the measurements.
