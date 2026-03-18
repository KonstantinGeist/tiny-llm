# Qwen3-1.7B Architecture

Source: GGUF metadata loaded from `Qwen3-1.7B-Q8_0.gguf` (GGUF v3).

## Identity

| Field | Value |
|---|---|
| Name | Qwen3 1.7B Instruct |
| Base model | Qwen3 1.7B |
| Organization | Qwen |
| Architecture | qwen3 |
| Fine-tune | Instruct |
| Size label | 1.7B |
| File type | MOSTLY_Q8_0 (8) |
| GGUF version | 3 |

## Model Dimensions

| Parameter | Value |
|---|---|
| `d_model` (embedding length) | 2048 |
| `n_layers` (block count) | 28 |
| `d_ff` (feed-forward length) | 6144 |
| `n_heads_q` (attention heads) | 16 |
| `n_heads_kv` (KV heads, GQA) | 8 |
| `head_dim` (key_length / value_length) | 128 |
| GQA group size (n_heads_q / n_heads_kv) | 2 |
| Context length | 40960 |
| Vocabulary size | 151936 |

## Attention

Grouped-Query Attention (GQA) with 16 query heads and 8 KV heads (group size 2).
**No bias** on Q/K/V or output projections.
**Per-head RMSNorm** applied to Q and K after projection and before RoPE (Qwen3 addition).

| Tensor | Shape | Type |
|---|---|---|
| `attn_norm.weight` | [2048] | F32 (RMSNorm) |
| `attn_q.weight` | [2048, 2048] | Q8_0 |
| `attn_q_norm.weight` | [128] | F32 (per-head RMSNorm) |
| `attn_k.weight` | [2048, 1024] | Q8_0 |
| `attn_k_norm.weight` | [128] | F32 (per-head RMSNorm) |
| `attn_v.weight` | [2048, 1024] | Q8_0 |
| `attn_output.weight` | [2048, 2048] | Q8_0 |

K and V heads: 8 heads × 128 head_dim = 1024.

## RoPE

| Parameter | Value |
|---|---|
| Dimensions | 128 (= head_dim) |
| Base frequency | 1,000,000 |

## Feed-Forward Network (SwiGLU)

Gate-and-up projection with SiLU activation, followed by a down projection.
No bias on any FFN weights.

| Tensor | Shape | Type |
|---|---|---|
| `ffn_norm.weight` | [2048] | F32 (RMSNorm) |
| `ffn_gate.weight` | [2048, 6144] | Q8_0 |
| `ffn_up.weight` | [2048, 6144] | Q8_0 |
| `ffn_down.weight` | [6144, 2048] | Q8_0 |

## Normalisation

RMSNorm throughout. Epsilon: 1e-06.

- Pre-attention: `blk.N.attn_norm.weight` [2048] F32
- Post-Q-projection: `blk.N.attn_q_norm.weight` [128] F32 (applied per head)
- Post-K-projection: `blk.N.attn_k_norm.weight` [128] F32 (applied per head)
- Pre-FFN: `blk.N.ffn_norm.weight` [2048] F32
- Final output norm: `output_norm.weight` [2048] F32

## Quantization

All weight matrices are stored as Q8_0 (8-bit quantization).
Q8_0 format: blocks of 32 elements, each block = 2 bytes (f16 scale) + 32 bytes (int8 values).
Dequantization: `f32[i] = scale * int8[i]`.
All tensors are dequantized to float32 at load time for inference.

## Full Tensor Layout

310 tensors total: 1 embedding + 28 × 11 per-block + 1 output norm.

### Global

| Tensor | Shape | Type | Notes |
|---|---|---|---|
| `token_embd.weight` | [2048, 151936] | Q8_0 | Token embedding table |
| `output_norm.weight` | [2048] | F32 | Final RMSNorm |

No separate `output.weight`; the embedding table is weight-tied to the LM head.

### Per block (repeated for N = 0 … 27)

| Tensor | Shape | Type |
|---|---|---|
| `blk.N.attn_norm.weight` | [2048] | F32 |
| `blk.N.attn_q.weight` | [2048, 2048] | Q8_0 |
| `blk.N.attn_q_norm.weight` | [128] | F32 |
| `blk.N.attn_k.weight` | [2048, 1024] | Q8_0 |
| `blk.N.attn_k_norm.weight` | [128] | F32 |
| `blk.N.attn_v.weight` | [2048, 1024] | Q8_0 |
| `blk.N.attn_output.weight` | [2048, 2048] | Q8_0 |
| `blk.N.ffn_norm.weight` | [2048] | F32 |
| `blk.N.ffn_gate.weight` | [2048, 6144] | Q8_0 |
| `blk.N.ffn_up.weight` | [2048, 6144] | Q8_0 |
| `blk.N.ffn_down.weight` | [6144, 2048] | Q8_0 |

## Tokenizer

| Field | Value |
|---|---|
| Model | gpt2 (BPE) |
| Pre-tokenizer | qwen2 |
| Vocabulary size | 151936 tokens |
| Merge rules | 151387 |
| BOS token ID | 151643 |
| EOS token ID | 151645 |
| Padding token ID | 151643 |
| Add BOS automatically | false |
| Chat template | Jinja2 (ChatML-style `<\|im_start\|>` / `<\|im_end\|>`) |
