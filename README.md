# Simplismart_ML_Assignment
MistralForCausalLM Inference Optimization
```markdown
# MistralForCausalLM Inference Optimization

This repository contains a highly optimized inference script for MistralForCausalLM models, specifically tuned for NVIDIA T4 GPUs (16GB VRAM). The implementation achieves 200+ tokens/sec throughput while handling 32 concurrent requests with 128 input and output tokens each.

## 🚀 Features

- High-throughput optimization: Exceeds 200 tokens/sec on a single T4 GPU
- Memory-efficient inference: Works within T4's 16GB VRAM constraints
- 4-bit quantization: Implements NF4 precision for reduced memory footprint
- Static KV cache: Optimized for memory coalescing and bandwidth utilization
- LoRA support: Compatible with LoRA fine-tuned models
- Interactive mode: Test the model with custom prompts
- Benchmark mode: Verify throughput on your hardware

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- NVIDIA T4 GPU (16GB VRAM) or similar
- CUDA 11.8+

## 📦 Installation

```bash
pip install -q transformers accelerate safetensors optimum einops flash-attn bitsandbytes
```

## 💻 Usage

### Running in Google Colab

1. Open the notebook in Google Colab
2. Ensure you're using a T4 GPU runtime
3. Run all cells

### Interactive Mode

After running the benchmark, the script enters interactive mode where you can test the model with custom prompts:

```
=== Interactive Mode ===
Type 'exit' to quit

Enter prompt: what is life from the pov of a fly/ or a prompt of your own pereference
```

## 🔍 Technical Details

### Optimization Techniques

1. Model Quantization
   - 4-bit NF4 precision via BitsAndBytesConfig
   - Double quantization for further memory reduction

2. Memory Access Optimization
   - Static KV cache implementation for memory coalescing
   - 128-token memory alignment to match T4's memory architecture
   - Optimal padding strategies for DRAM burst utilization

3. Batching Strategy
   - Single worker thread with batch size 32
   - Maximizes memory bandwidth utilization

4. Compilation Optimization
   - `torch.compile` with max-autotune for kernel fusion
   - TF32 precision where beneficial

5. Hardware-Specific Tuning
   - T4-specific CUDA optimizations
   - Tensor core utilization where possible

### Performance Metrics

- Throughput: 214 tokens/sec (exceeds 200 tokens/sec target)
- Memory usage: ~8.7GB during inference
- Batch size: 32 concurrent requests
- Latency: ~38 seconds for 128-token generation

## 📈 Benchmarking

The script includes a comprehensive benchmarking module to measure throughput:

```
=== Benchmark Results ===
Total requests: 32
Successful requests: 32
Total tokens processed: 8160
Total time: 38.09 seconds
Aggregate throughput: 214.25 tokens/sec
Average latency per request: 38.07 seconds
```

## 🧪 Architecture Considerations

This implementation is specifically optimized for NVIDIA T4 GPUs (Turing architecture, compute capability 7.5). Key architectural considerations:

- Memory bandwidth (320 GB/s) is the primary bottleneck
- Turing architecture doesn't support Flash Attention
- SDPA is not compatible with compute capability 7.5
- Static cache implementation is crucial for memory coalescing
- Single worker with large batch size outperforms multiple workers

##

- Hugging Face for the Transformers library
- Mistral AI for the MistralForCausalLM model
- NVIDIA for GPU architecture documentation
```
