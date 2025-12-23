# nano-vLLM Code Walkthrough Tutorial

This tutorial explains **nano-vLLM** as a *minimal, educational reimplementation of vLLM-style inference*.
The goal is to understand **how LLM inference works at the system level**, not to optimize performance.

We will focus on the **core design ideas**, the **key classes**, and **how data flows through the system**.

---

## 1. High-Level Architecture

nano-vLLM is organized around four core components:

```
LLMEngine
   ├── Scheduler
   │      ├── waiting queue
   │      ├── running queue
   │      └── BlockManager
   ├── ModelRunner
   │      ├── model + KV cache
   │      ├── prefill / decode preparation
   │      └── optional CUDA graphs
   └── Sequence objects (per request)
```

**Key idea:**  
KV cache is treated like *paged virtual memory* rather than a monolithic tensor.

---

## 2. LLMEngine: The Orchestrator

**File:** `engine/llm_engine.py`

### Responsibilities
- Accept user prompts
- Drive the inference loop
- Coordinate scheduler and model execution
- Collect finished outputs

### Key Methods

#### `__init__`
- Builds `Config`
- Spawns tensor-parallel worker processes (if enabled)
- Creates:
  - `ModelRunner`
  - `Scheduler`
  - HuggingFace tokenizer

#### `add_request(prompt, sampling_params)`
- Tokenizes prompt (if needed)
- Wraps it in a `Sequence`
- Pushes it into the scheduler

#### `step()`
One inference iteration:
1. Ask scheduler what to run
2. Run model (prefill or decode)
3. Postprocess tokens
4. Return finished sequences

#### `generate()`
- Convenience API
- Adds all prompts
- Repeatedly calls `step()` until all sequences finish
- Decodes token IDs to text

**Important:**  
`LLMEngine` does *not* manage memory or batching logic directly.
Those responsibilities live in `Scheduler` and `BlockManager`.

---

## 3. Scheduler: Batching, Prefill, Decode, Preemption

**File:** `engine/scheduler.py`

### Core Data Structures
- `waiting`: sequences not holding KV cache
- `running`: sequences actively decoding with KV cache

### Scheduling Strategy

#### Prefill Phase
- Admit new sequences from `waiting`
- Constraints:
  - `max_num_seqs`
  - `max_num_batched_tokens`
  - KV cache availability (`BlockManager.can_allocate`)
- Allocate KV blocks
- Move sequences to `running`

If any prefill happens → **no decode this step**.

#### Decode Phase
- Take sequences from `running`
- Each sequence generates **one token**
- If KV memory is insufficient:
  - Preempt another sequence
  - Free its blocks
  - Move it back to `waiting`

This implements **memory-aware continuous batching**.

#### Postprocess
- Append generated tokens
- Detect EOS or max length
- Deallocate KV cache for finished sequences

---

## 4. Sequence: Logical Request State

**File:** `engine/sequence.py`

A `Sequence` stores *metadata*, not tensors.

### Key Fields

| Field | Meaning |
|------|--------|
| `token_ids` | Prompt + generated tokens |
| `num_prompt_tokens` | Prompt length |
| `num_tokens` | Total tokens so far |
| `num_cached_tokens` | Tokens already cached in KV |
| `block_table` | Logical → physical KV block mapping |
| `status` | WAITING / RUNNING / FINISHED |

### Block-related Properties

- `num_blocks`: how many KV blocks needed
- `num_cached_blocks`: fully cached blocks
- `last_block_num_tokens`: tokens in last block

### Why This Matters
`Sequence` allows:
- Prefix caching
- Partial recomputation
- Safe preemption and resumption

---

## 5. BlockManager: Paged KV Cache

**File:** `engine/block_manager.py`

This is the **heart of nano-vLLM**.

### Key Idea
KV cache is split into **fixed-size blocks**.
Sequences reference blocks instead of owning contiguous memory.

### Core Structures

- `Block`
  - `block_id`
  - `ref_count`
  - `hash`
  - `token_ids`
- `free_block_ids`
- `used_block_ids`
- `hash_to_block_id` (prefix cache)

### Allocation (`allocate`)
- Iterate over sequence blocks
- Compute rolling hash for full blocks
- Reuse existing blocks if hash + content match
- Otherwise allocate new blocks
- Update `Sequence.block_table`

### Deallocation (`deallocate`)
- Decrement block ref-counts
- Free blocks when ref-count reaches zero
- Clear sequence block table

### Append (`may_append`)
Handles three cases:
1. Starting a new block
2. Completing a block (compute hash)
3. Appending inside a block

This enables **incremental decoding with reuse**.

---

## 6. ModelRunner: Model Execution Engine

**File:** `engine/model_runner.py`

### Responsibilities
- Initialize distributed environment (NCCL)
- Load model weights
- Allocate global KV cache tensor
- Prepare inputs for prefill and decode
- Execute model forward pass
- Sample next tokens

### KV Cache Allocation
- Estimate available GPU memory
- Compute bytes per KV block
- Allocate a large tensor:
  ```
  [K/V, layers, blocks, block_size, heads, head_dim]
  ```
- Attach slices to attention layers

### Prefill Preparation
- Pack variable-length sequences
- Build:
  - `input_ids`
  - `positions`
  - `cu_seqlens_q / cu_seqlens_k`
  - `slot_mapping`
  - `block_tables`
- Set runtime context for attention

### Decode Preparation
- One token per sequence
- Compute KV slot for the new token
- Reuse block tables for history

### CUDA Graphs (Optional)
- Capture decode graphs for common batch sizes
- Replay graphs for fast, low-overhead decoding

---

## 7. End-to-End Token Flow

### Prefill
```
prompt tokens
 → Scheduler allocates blocks
 → ModelRunner packs inputs
 → Attention writes KV into blocks
```

### Decode (per token)
```
last token
 → Scheduler checks memory
 → BlockManager appends KV slot
 → ModelRunner runs forward
 → Sampler picks next token
```

---

## 8. Key Takeaways

- nano-vLLM cleanly separates:
  - **policy** (Scheduler)
  - **memory** (BlockManager)
  - **execution** (ModelRunner)
- KV cache is virtualized via blocks
- Prefix reuse and preemption are first-class concepts
- This design mirrors real vLLM, but is readable

If you understand nano-vLLM, you understand **modern LLM inference systems**.

---
