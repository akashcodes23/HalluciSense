# Phase 56 — PyTorch Memory Forensics

## PyTorch CPU Runtime Analysis

- **Model**: `cross-encoder/nli-deberta-v3-small` (141M parameters).
- **Weight Footprint (fp32)**: $\approx 564\text{ MB}$.
- **Inference Mode**: `torch.inference_mode()` with `param.requires_grad = False`.
- **CPU Threads**: Configured via `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)`.
- **Gradient Tracking**: Disabled globally.
- **Deserialization Behavior**:
  - Unoptimized `from_pretrained`: Loads full serialized state dictionary into Python memory while allocating PyTorch module weights simultaneously, creating a transient spike of $\approx 1.1\text{ GB}$.
  - Hardened `from_pretrained(low_cpu_mem_usage=True)`: Directly maps parameters into model weights, eliminating the redundant state dictionary buffer.
