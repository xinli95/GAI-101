# nano-vLLM: Key Mental Models Across Core Classes

This document distills the **core mental models** you should carry when reading or explaining nano‑vLLM.
Rather than restating code, it focuses on **how the classes think together** and **why responsibilities are split the way they are**.

---

## 1. The Single Unifying Idea

> **LLM inference is a memory‑management problem disguised as a modeling problem.**

nano‑vLLM is built around this insight.

Everything else follows from one question:

> *How do we maximize GPU utilization while safely sharing KV cache across many variable‑length requests?*

---

## 2. Mental Model Overview (Bird’s‑Eye View)

```
LLMEngine      → control plane
Scheduler      → policy & batching
Sequence       → logical request state
BlockManager   → KV memory virtualization
ModelRunner    → execution engine
```

Each class owns **one dimension** of the system and avoids leaking responsibility.

---

## 3. LLMEngine: “The Control Plane”

### Mental Model
**LLMEngine is orchestration, not intelligence.**

It does NOT:
- decide batching policy
- manage KV memory
- understand attention mechanics

It DOES:
- connect components
- advance the global clock (step by step)
- provide a user‑friendly API

### Think of it as
> A traffic controller that asks others *what should happen next* and *executes the plan*.

---

## 4. Scheduler: “The Policy Brain”

### Mental Model
**Scheduler answers one question repeatedly:**

> *Which sequences should run right now, and in what mode?*

It balances:
- fairness
- throughput
- memory pressure

### Two‑Phase Worldview
Scheduler always lives in one of two modes:

#### Prefill Mode
- Admit new sequences
- Allocate KV blocks
- Process many tokens at once

#### Decode Mode
- One token per sequence
- Memory‑constrained
- Preemption is allowed

### Key Insight
> **Prefill is token‑heavy; decode is memory‑heavy.**

Scheduler enforces this separation explicitly.

---

## 5. Sequence: “The Logical Illusion”

### Mental Model
**A Sequence pretends KV cache is continuous and infinite.**

But it is not.

The sequence only tracks:
- tokens
- block indices
- how much of itself is cached

### Sequence Never Knows
- where KV lives physically
- whether blocks are shared
- whether it was preempted

### Key Fields as Mental Anchors

| Field | Mental Meaning |
|-----|---------------|
| `token_ids` | What the model *thinks* it has seen |
| `num_cached_tokens` | What the GPU has actually stored |
| `block_table` | The illusion → reality mapping |
| `status` | Whether it is currently “alive” |

### Key Insight
> **Sequence is a pure logical abstraction.**  
> It survives preemption because it never owns memory directly.

---

## 6. BlockManager: “Virtual Memory for Attention”

### Mental Model
**BlockManager is a tiny operating system for KV cache.**

It provides:
- allocation
- deallocation
- reference counting
- content‑addressable reuse (hashing)

### Think in OS Terms

| OS Concept | nano‑vLLM Equivalent |
|---------|---------------------|
| Page | KV block |
| Page table | `Sequence.block_table` |
| Page cache | `hash_to_block_id` |
| Swap out | Preemption |
| Ref count | Shared prefixes |

### Why Hashing Matters
Hashing enables:
- prefix reuse
- zero recomputation for shared prompts
- safe sharing via reference counts

### Key Insight
> **KV cache is not tied to a request — it is a shared resource.**

---

## 7. Scheduler + BlockManager: The Critical Contract

### Contract Boundary
- Scheduler asks: *Can I allocate / append?*
- BlockManager answers: *Yes / No*

Scheduler never:
- touches block internals
- assumes memory layout

BlockManager never:
- decides who runs
- understands fairness

### Why This Matters
This separation makes:
- preemption safe
- policy changes easy
- memory bugs localized

---

## 8. ModelRunner: “The Execution Engine”

### Mental Model
**ModelRunner is a stateless executor with a shared memory backend.**

It assumes:
- Scheduler already decided legality
- BlockManager already mapped memory

### ModelRunner Responsibilities
- turn sequences into tensors
- translate logical tokens → physical KV slots
- run forward pass
- sample tokens

### Context System
The `set_context(...)` mechanism acts like:
> a thread‑local execution contract between scheduler logic and attention kernels

Attention layers do not ask *why* — they just consume context.

---

## 9. Prefill vs Decode: Two Different Worlds

### Prefill World
- many tokens
- variable lengths
- packed attention
- little reuse

### Decode World
- one token per sequence
- fixed patterns
- heavy reuse
- CUDA graphs possible

### Key Insight
> **Decode is where performance lives; prefill is where complexity lives.**

nano‑vLLM models this explicitly.

---

## 10. Preemption: The Hidden Superpower

### Mental Model
Preemption is possible because:

- Sequences don’t own memory
- Blocks are reference‑counted
- State is purely logical

### What Preemption Really Means
> *“You lose your KV cache, but not your identity.”*

The sequence can be reconstructed later.

---

## 11. Why This Design Scales

This architecture scales because:

- policy is isolated (Scheduler)
- memory is virtualized (BlockManager)
- execution is stateless (ModelRunner)
- requests are cheap abstractions (Sequence)

### Final Insight

> **nano‑vLLM works not because it is clever, but because each class refuses to do more than one job.**

This is the same design philosophy behind production vLLM.

---
