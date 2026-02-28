---
title: Advanced Attention Mechanisms
---

# Advanced Attention Mechanisms

Since the original Transformer's scaled dot-product attention, numerous variants have been proposed to address efficiency, long-range dependencies, and specific use cases.

## Efficient Attention

Standard self-attention has O(n²) complexity in sequence length, making it expensive for long sequences. Several approaches have been developed to reduce this cost.

Linear attention replaces the softmax with a kernel function, allowing the attention computation to be rewritten as a matrix product that runs in O(n) time. Performers use Random Fourier Features to approximate softmax attention in linear time. While faster, these approximations can lose some of the expressiveness of full attention.

Sparse attention patterns, used in models like Longformer and BigBird, restrict each token to attend to only a subset of other tokens. Longformer combines local sliding window attention with global attention for special tokens. BigBird adds random attention connections to ensure theoretical completeness. These models can handle sequences of 4096 to 16384 tokens efficiently.

Flash Attention is a hardware-aware algorithm that computes exact attention using tiling and recomputation to minimize memory I/O. It achieves 2-4x speedup over standard attention by keeping intermediate results in fast SRAM rather than slower HBM. Flash Attention 2 further improves parallelism and reduces non-matmul FLOPs.

## Multi-Query and Grouped-Query Attention

Multi-Query Attention (MQA), introduced by Shazeer, shares key and value projections across all attention heads while keeping separate query projections. This dramatically reduces memory bandwidth during autoregressive decoding with minimal quality loss.

Grouped-Query Attention (GQA) is a compromise between standard multi-head attention and MQA. It groups heads into clusters, with each group sharing one set of key-value projections. Llama 2 70B uses GQA with 8 KV heads shared across 64 query heads.

## Cross-Attention and Its Applications

Cross-attention allows a model to attend to a different sequence than the one it's processing. In encoder-decoder models, the decoder uses cross-attention to attend to encoder representations.

Vision-language models like Flamingo use cross-attention to fuse visual and textual features. The text decoder cross-attends to visual features extracted by a frozen vision encoder. This architecture enables few-shot visual question answering and image captioning.

## Attention in Modern LLMs

Modern large language models have adopted several attention innovations. Rotary Position Embeddings (RoPE) encode relative positions directly into the attention scores using rotation matrices. ALiBi adds a linear position-dependent bias to attention scores, avoiding the need for position embeddings entirely.

KV-cache optimization is critical for efficient inference. During autoregressive generation, the key and value vectors for previous tokens are cached to avoid redundant computation. Techniques like PagedAttention (used in vLLM) manage KV-cache memory like virtual memory pages, reducing waste from fragmentation.

Sliding window attention, used in Mistral, limits each token to attend to only a fixed window of recent tokens. This bounds memory usage during inference while maintaining good performance through the use of rolling KV-cache.
