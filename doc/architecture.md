# Qwen2.5-0.5B-Instruct Architecture

Source: GGUF metadata loaded from `Qwen2.5-0.5B-Instruct-f16.gguf` (GGUF v3).

## Identity

| Field | Value |
|---|---|
| Name | Qwen2.5 0.5B Instruct |
| Base model | Qwen2.5 0.5B |
| Organization | Qwen |
| Architecture | qwen2 |
| Fine-tune | Instruct |
| Size label | 0.5B |
| License | Apache-2.0 |
| Languages | en |
| File type | MOSTLY_F16 (1) |
| GGUF version | 3 |

## Model Dimensions

| Parameter | Value |
|---|---|
| `d_model` (embedding length) | 896 |
| `n_layers` (block count) | 24 |
| `d_ff` (feed-forward length) | 4864 |
| `n_heads_q` (attention heads) | 14 |
| `n_heads_kv` (KV heads, GQA) | 2 |
| `head_dim` (d_model / n_heads_q) | 64 |
| GQA group size (n_heads_q / n_heads_kv) | 7 |
| Context length | 32768 |
| Vocabulary size | 151936 |
| Approximate parameter count | ~494M |

## Attention

Grouped-Query Attention (GQA) with 14 query heads and 2 KV heads (group size 7).
Each Q/K/V projection also carries a bias vector.

| Tensor | Shape | Type |
|---|---|---|
| `attn_norm.weight` | [896] | F32 (RMSNorm) |
| `attn_q.weight` | [896, 896] | F16 |
| `attn_q.bias` | [896] | F32 |
| `attn_k.weight` | [896, 128] | F16 |
| `attn_k.bias` | [128] | F32 |
| `attn_v.weight` | [896, 128] | F16 |
| `attn_v.bias` | [128] | F32 |
| `attn_output.weight` | [896, 896] | F16 |

K and V heads: 2 heads × 64 head_dim = 128.

## RoPE

| Parameter | Value |
|---|---|
| Dimensions | 64 (= head_dim) |
| Base frequency | 1,000,000 |

## Feed-Forward Network (SwiGLU)

Gate-and-up projection with SiLU activation, followed by a down projection.
No bias on FFN weights.

| Tensor | Shape | Type |
|---|---|---|
| `ffn_norm.weight` | [896] | F32 (RMSNorm) |
| `ffn_gate.weight` | [896, 4864] | F16 |
| `ffn_up.weight` | [896, 4864] | F16 |
| `ffn_down.weight` | [4864, 896] | F16 |

## Normalisation

RMSNorm throughout. Epsilon: 1e-06.

- Pre-attention: `blk.N.attn_norm.weight` [896] F32
- Pre-FFN: `blk.N.ffn_norm.weight` [896] F32
- Final output norm: `output_norm.weight` [896] F32

## Full Tensor Layout

290 tensors total: 1 embedding + 24 × 12 per-block + 1 output norm.

### Global

| Tensor | Shape | Type | Notes |
|---|---|---|---|
| `token_embd.weight` | [896, 151936] | F16 | Token embedding table |
| `output_norm.weight` | [896] | F32 | Final RMSNorm |

No separate `output.weight`; the embedding table is weight-tied to the LM head.

### Per block (repeated for N = 0 … 23)

| Tensor | Shape | Type |
|---|---|---|
| `blk.N.attn_norm.weight` | [896] | F32 |
| `blk.N.attn_q.weight` | [896, 896] | F16 |
| `blk.N.attn_q.bias` | [896] | F32 |
| `blk.N.attn_k.weight` | [896, 128] | F16 |
| `blk.N.attn_k.bias` | [128] | F32 |
| `blk.N.attn_v.weight` | [896, 128] | F16 |
| `blk.N.attn_v.bias` | [128] | F32 |
| `blk.N.attn_output.weight` | [896, 896] | F16 |
| `blk.N.ffn_norm.weight` | [896] | F32 |
| `blk.N.ffn_gate.weight` | [896, 4864] | F16 |
| `blk.N.ffn_up.weight` | [896, 4864] | F16 |
| `blk.N.ffn_down.weight` | [4864, 896] | F16 |

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
