# 2-GPU Setup Guide

This guide explains how to configure and run the evaluation framework with 2 GPUs at 95% memory utilization.

## GPU Configuration

### Hardware Requirements
- **2 GPUs** (NVIDIA A100, A6000, RTX 6000, or similar)
- **VRAM per GPU**: 40GB+ recommended
- **Total VRAM**: 80GB+

### Supported GPUs
- NVIDIA A100 (40GB or 80GB)
- NVIDIA A6000 (48GB)
- NVIDIA RTX 6000 (48GB)
- NVIDIA L40S (48GB)
- NVIDIA V100 (32GB) - minimum for smaller models

## Quick Start (2 minutes)

### Step 1: Verify GPU Availability

```bash
# Check available GPUs
nvidia-smi

# Expected output:
# GPU 0: [Model] with [VRAM]GB
# GPU 1: [Model] with [VRAM]GB
```

### Step 2: Start vLLM Server with 2 GPUs

```bash
# Option 1: Use the provided script (recommended)
./start_vllm_servers.sh start-8b

# Option 2: Start manually with environment variables
export CUDA_VISIBLE_DEVICES=0,1
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --dtype auto \
    &

# Wait for server to start
sleep 30
curl http://localhost:8000/health
```

### Step 3: Run Evaluations

```bash
# The evaluation script automatically uses 2 GPUs
export OPENAI_BASE_URL=http://localhost:8000/v1
./run_full_evaluation.sh all
```

## Detailed Configuration

### vLLM Server Parameters

```bash
python -m vllm.entrypoints.openai.api_server \
    --model META_LLAMA_MODEL \
    --port 8000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --dtype auto \
    --max-model-len 4096 \
    --swap-space 4
```

**Parameter Explanation**:

| Parameter | Value | Reason |
|---|---|---|
| `--gpu-memory-utilization` | 0.95 | Use 95% of available GPU memory |
| `--tensor-parallel-size` | 2 | Distribute model across 2 GPUs |
| `--dtype` | auto | Automatically select optimal dtype |
| `--max-model-len` | 4096 | Maximum sequence length |
| `--swap-space` | 4 | CPU swap space for context (GB) |

### Environment Variables

```bash
# Set which GPUs to use (0 and 1)
export CUDA_VISIBLE_DEVICES=0,1

# Optional: Set GPU memory fraction
export GPU_MEMORY_FRACTION=0.95

# Set for evaluation script
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=not-set
```

## Running Different Models

### Llama 3.1-8B (Recommended for Testing)
```bash
./start_vllm_servers.sh start-8b
# Fastest, requires less memory, good for quick tests
```

### Llama 3.1-70B (Full Model)
```bash
./start_vllm_servers.sh start-70b
# Requires more GPU memory, slower but more capable
# Ensure you have at least 80GB total VRAM
```

### Mistral-7B (Lightweight)
```bash
./start_vllm_servers.sh start-mistral
# Fast and efficient, requires less memory
```

### Qwen 3.5-27B
```bash
./start_vllm_servers.sh start-qwen
# Balanced model, requires moderate GPU memory
```

## Optimization Tips

### If You Get Out of Memory (OOM) Error

**Option 1: Reduce GPU Memory Utilization**
```bash
# Use 85% instead of 95%
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 2
```

**Option 2: Reduce Batch Size**
```bash
# Process fewer requests in parallel
--max-num-batched-tokens 4096
```

**Option 3: Reduce Max Model Length**
```bash
# Use shorter sequences
--max-model-len 2048
```

### For Maximum Throughput

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --max-model-len 4096 \
    --swap-space 4 \
    --enable-lora
```

### For Stability (Lower Utilization)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.80 \
    --tensor-parallel-size 2 \
    --dtype auto \
    --max-model-len 2048
```

## Evaluation Script Integration

The evaluation script automatically uses 2 GPUs:

```bash
# In run_full_evaluation.sh, the following is set:
export CUDA_VISIBLE_DEVICES=0,1
GPU_MEMORY_UTILIZATION=0.95
TENSOR_PARALLEL_SIZE=2
```

You can override these:

```bash
# Use different GPUs
export CUDA_VISIBLE_DEVICES=2,3
./run_full_evaluation.sh base

# Override memory utilization (not recommended, 0.95 is optimal)
# Edit run_full_evaluation.sh and change GPU_MEMORY_UTILIZATION
```

## Monitoring GPU Usage

### Real-time GPU Monitoring

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or use dedicated tools
nvidia-smi dmon

# Or Python script
python3 << 'EOF'
import subprocess
import time

while True:
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader']).decode()
    print(f"\n{time.strftime('%H:%M:%S')} - GPU Status:")
    for line in output.strip().split('\n'):
        print(f"  {line}")
    time.sleep(2)
EOF
```

### Check Server Status

```bash
# Check if server is running
curl -s http://localhost:8000/health | jq .

# Check server stats
curl -s http://localhost:8000/v1/models | jq .

# Monitor specific server
for i in {1..10}; do
    echo "Checking... $(date)"
    curl -s http://localhost:8000/health > /dev/null && echo "  OK" || echo "  FAIL"
    sleep 10
done
```

### GPU Memory Allocation

```bash
# Monitor GPU memory during evaluation
watch -n 1 'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader'

# Check allocated memory per process
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution 1: Reduce utilization**
```bash
--gpu-memory-utilization 0.85
```

**Solution 2: Reduce batch size**
```bash
--max-num-batched-tokens 4096
```

**Solution 3: Check for other processes**
```bash
nvidia-smi  # Look for other processes using GPU
```

### Issue: "RuntimeError: NCCL error"

**Solution**: Reduce tensor parallel size or use single GPU
```bash
--tensor-parallel-size 1  # Use only one GPU
```

### Issue: Server won't start

**Check logs**:
```bash
tail -f vllm_logs/vllm_server_8000.log
```

**Verify GPUs are available**:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

### Issue: Slow evaluation

**Check GPU utilization**:
```bash
nvidia-smi -l 1  # Refresh every 1 second
```

**If utilization is low**:
- Increase batch size: `--max-num-batched-tokens 8192`
- Reduce overhead: `--disable-log-requests`
- Use faster dtype: `--dtype float16`

## Performance Benchmarks

### Expected Throughput with 2 GPUs @ 0.95 Memory

| Model | Memory Used | Throughput (tokens/sec) | Batch Size |
|---|---|---|---|
| Llama 3.1-8B | ~38GB | 500-800 | 32-64 |
| Llama 3.1-70B | ~140GB (needs 2×80GB) | 100-200 | 4-8 |
| Mistral-7B | ~20GB | 800-1200 | 64-128 |
| Qwen 3.5-27B | ~60GB | 300-500 | 16-32 |

### Evaluation Timing with 2 GPUs

```
Llama 3.1-8B (2 GPUs @ 0.95):
- Base Attacks:      2-3 hours
- Repeated Prompt:   4-5 hours
- Multi-Turn:        4-5 hours
- PromptGuard:       2-3 hours
- SecAlign:          1 hour
- SFT:               1 hour
Total:              14-17 hours

vs Single GPU:       35-40 hours
```

## Advanced Configuration

### Using Multiple Servers

Run different models on different GPU pairs:

```bash
# Terminal 1: Llama 8B on GPUs 0,1
export CUDA_VISIBLE_DEVICES=0,1
./start_vllm_servers.sh start-8b

# Terminal 2: Mistral on GPUs 2,3 (if available)
export CUDA_VISIBLE_DEVICES=2,3
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.1 \
    --port 8001 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 &
```

Then run evaluations against specific servers:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
./run_full_evaluation.sh base

export OPENAI_BASE_URL=http://localhost:8001/v1
./run_full_evaluation.sh base
```

### Using with SLURM (HPC Clusters)

```bash
#!/bin/bash
#SBATCH --gpus=2
#SBATCH --mem=256GB
#SBATCH --time=24:00:00

export CUDA_VISIBLE_DEVICES=0,1

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 &

sleep 30

export OPENAI_BASE_URL=http://localhost:8000/v1
./run_full_evaluation.sh all
```

## Useful Commands

```bash
# Start server and log to file
nohup ./start_vllm_servers.sh start-8b > vllm.log 2>&1 &

# Check if server is responsive
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","prompt":"Hello","max_tokens":10}'

# Kill server by port
fuser -k 8000/tcp

# Kill server by PID
kill $(cat vllm_logs/server_8000.pid)

# Run evaluation with specific server
OPENAI_BASE_URL=http://localhost:8000/v1 ./run_full_evaluation.sh base

# Run evaluation and log output
./run_full_evaluation.sh all 2>&1 | tee evaluation_run_$(date +%Y%m%d_%H%M%S).log
```

## GPU Memory Layout

### With 2 GPUs @ 95% Utilization

```
GPU 0 (40GB)              GPU 1 (40GB)
├─ Model (part 1)  ~19GB  ├─ Model (part 2)  ~19GB
├─ KV Cache        ~12GB  ├─ KV Cache        ~12GB
├─ Batch Buffer     ~7GB  ├─ Batch Buffer     ~7GB
└─ Reserved         ~2GB  └─ Reserved         ~2GB
   Total: ~38GB             Total: ~38GB
```

## Next Steps

1. **Verify GPUs**: Run `nvidia-smi`
2. **Start Server**: `./start_vllm_servers.sh start-8b`
3. **Test Server**: `curl http://localhost:8000/health`
4. **Run Evaluation**: `./run_full_evaluation.sh base`
5. **Monitor**: Watch GPU usage with `watch -n 1 nvidia-smi`
6. **Analyze Results**: `python analyze_evaluation_results.py`

## Support

For issues:
1. Check GPU availability: `nvidia-smi`
2. Review vLLM logs: `tail -f vllm_logs/vllm_server_8000.log`
3. Test server: `curl http://localhost:8000/health`
4. Try single GPU: Remove `--tensor-parallel-size 2`

---

**Optimized for maximum throughput with 2 GPUs at 95% memory utilization!**
