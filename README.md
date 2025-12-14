# Flash-ColReduce

[![PyPI](https://img.shields.io/pypi/v/flash-colreduce)](https://pypi.org/project/flash-colreduce/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Flash-ColReduce** provides highly optimized Triton kernels for computing column-wise reductions of the attention matrix such as sums or means without materializing the full $O(N^2)$ attention weights.

This primitive is essential for KV-cache pruning, token importance estimation, and attention analysis in Large Language Models (LLMs) and Vision-Language Models (VLMs). It powers the visual token pruning in [SparseVILA](https://arxiv.org/abs/2510.17777).

## Highlights

- **🚀 Efficient**: Fused kernels compute column reductions in **$O(N)$ memory**, enabling 128k+ context lengths.
- **🧩 Flexible**: Supports **causal** and **non-causal** attention with irregular shapes ($M \neq N$).
- **✅ Exact**: Uses online softmax for numerical precision and correct causal masking.

## Prerequisites

- **Python**: 3.10+
- **PyTorch**: 2.1+ (with CUDA support)
- **Triton**: 3.0.0+
- **GPU**: NVIDIA GPU with Compute Capability 8.0+ (Ampere or newer recommended)

## Installation

Install from PyPI:
```bash
pip install flash-colreduce
```

Or build from source:
```bash
git clone https://github.com/z-lab/flash-colreduce.git
cd flash-colreduce
pip install -e .
```

## Usage

### 1. Non-Causal Attention

Compute a column-wise reduction of the attention matrix over the query dimension.

```python
import torch
from flash_colreduce import flash_colreduce

q = torch.randn(8, 16, 512, 64, device="cuda", dtype=torch.float16)
k = torch.randn(8, 16, 512, 64, device="cuda", dtype=torch.float16)

flash_colreduce(q, k, reduction="sum")   # Shape: (8, 16, 512)
flash_colreduce(q, k, reduction="mean")  # Shape: (8, 16, 512)
```

### 2. Causal Attention

Handle autoregressive attention where $M \neq N$. The kernel applies a right-aligned causal mask matching KV-cached decoding behavior.

```python
import torch
from flash_colreduce import flash_colreduce

q = torch.randn(1, 32, 128, 128, device="cuda", dtype=torch.float16)
k = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.float16)

flash_colreduce(q, k, is_causal=True, reduction="sum")  # Shape: (1, 32, 4096)
flash_colreduce(q, k, is_causal=True, reduction="mean")  # Shape: (1, 32, 4096)
```

## Performance

Flash-ColReduce achieves significant speedups and memory savings over naïve implementations. By fusing softmax and reduction into a single kernel, it avoids writing the massive $B \times H \times M \times N$ attention matrix to GPU memory.

![Benchmark Results on RTX Pro 6000 Blackwell](assets/benchmarks/rtx-pro-6000-blackwell.png)
*Benchmarked on NVIDIA RTX Pro 6000 Blackwell with FP16 precision*

## Development

### Project Structure
```
flash-colreduce/
├── flash_colreduce/       # Source code
│   ├── flash.py           # Triton kernels & API
│   └── naive.py           # Reference PyTorch implementations
├── benchmarks/            # Performance scripts
└── tests/                 # Correctness tests
```

### Running Tests
```bash
pip install -e ".[test]"
pytest -v
```

### Running Benchmarks
```bash
cd benchmarks
python run.py
```

## Citation

If you use Flash-ColReduce in your research, please cite the SparseVILA paper:

```bibtex
@inproceedings{khaki2025sparsevila,
  title = {{SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference}},
  author = {Khaki, Samir and Guo, Junxian and Tang, Jiaming and Yang, Shang and Chen, Yukang and Plataniotis, Konstantinos N and Lu, Yao and Han, Song and Liu, Zhijian},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2025}
}
```

## License

[MIT License](LICENSE)

## Acknowledgments

- **[FlashAttention](https://github.com/Dao-AILab/flash-attention)**: The tiling and online softmax approach is heavily inspired by FlashAttention.
- **[SparseVILA](https://arxiv.org/abs/2510.17777)**: The original project that motivated this primitive.
