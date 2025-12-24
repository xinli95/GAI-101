# Paper Reading Notes: Search-R1

> **Paper:** _Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning_  
> **Goal of these notes:** capture the _core idea_, _system design_, _training/eval details_.

---

## 0. One-paragraph summary

Search-R1 trains a language model to **interleave reasoning with search actions** (issuing queries, reading retrieved evidence, and deciding when to stop) using **reinforcement learning** rather than supervised tool-use traces. The key training stability trick is to treat retrieved snippets as **environment observations** (masked out of the policy-gradient loss) while updating only on **model-generated tokens**, allowing the model to learn _when_ to search and _how_ to form queries based on an outcome-based reward computed from final answer correctness.

---

## 1. Motivation and problem framing

- **Why not plain RAG?** RAG often assumes a fixed retrieve-then-answer pipeline (e.g., top-k once), which is suboptimal for multi-hop or ambiguous questions.
- **Why not tool-use SFT?** Demonstration traces for multi-turn search are expensive and narrow; SFT can overfit to a rigid tool-calling pattern.
- **Search-R1 objective:** learn a **search policy**: decide _(a)_ whether to search, _(b)_ what query to issue, _(c)_ how many turns to search, and _(d)_ when to stop and answer.

---

## 2. High-level method

### 2.1 Interaction loop (trajectory)

A single rollout is a trajectory:

1. Input question **q**
2. Model generates reasoning tokens and possibly a **SEARCH(query)** action
3. Search engine returns retrieved text **d** (snippets / documents)
4. Model conditions on **q + (previous context) + d** and continues
5. Repeat for up to **T** search turns
6. Model outputs final answer **â**
7. Compute reward **R(â, a\*)** based on correctness

### 2.2 Key design choice: observations vs actions

- **Model-generated tokens** are actions → participate in policy gradient.
- **Retrieved text** is an environment observation → **masked** from the RL loss.

This avoids mis-crediting the retriever’s content as if the model “chose” those tokens.

---

## 3. Training details (what to look for)

### 3.1 Data format

- Training instances are typically _(question, gold answer)_ pairs.
- There are **no gold search traces** required.

Practical implication: your dataset must support **automatic answer checking** for reward computation.

### 3.2 Reward

- Outcome-based reward from final answer correctness.
- Dataset-dependent matching (Exact Match / F1 / normalized string match).

### 3.3 RL optimization

- Uses a PPO/GRPO-style update (reasoning-RL family).
- The policy learns query formulation and tool timing through trial-and-error.

### 3.4 Stability tricks / engineering knobs

- **Retrieved-token masking** (most important)
- Handling abnormal trajectories (e.g., infinite search loops)
- Train/inference mismatch mitigation (ensure tool interface is identical)
- Strict formatting for tool calls (token-in/token-out alignment)

---

## 4. Evaluation

Evaluation is standard QA scoring, with search enabled at inference:

- **Primary:** answer EM/F1/accuracy.
- **Diagnostics:** number of searches, average turns, stop condition behavior.

What I’d add when reproducing:

- Measure **over-search** rate on simple questions (wasted tool calls).
- Track **query quality** (duplicate queries, refinement patterns).

---

## 5. Worked example: Search-R1-style prompting & rollout

This example illustrates how **interleaved reasoning and search** works conceptually in Search-R1-style setups.

```text
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information.
If you find you lack some knowledge, you can call a search engine by <search> query </search>.

Question: paul walkers cars in fast and furious list?
<|im_end|>
<|im_start|>assistant
<think>
I need to search for Paul Walker's cars in the Fast and Furious movies.
</think>
<search> Paul Walker cars in Fast and Furious </search>

<information>
(Retrieved documents from search engine)
</information>

<think>
The retrieved information is incomplete; I may need another search or decide to answer.
</think>
<search> list of cars driven by Paul Walker in Fast and Furious </search>

<information>
(Retrieved documents from search engine)
</information>

<answer>
Dodge Charger
</answer>
<|im_end|>
```

**Key takeaway:**

- Search queries are _actions_ chosen by the model.
- Retrieved text is treated as _observations_.
- Only the model’s generated tokens (not retrieved text) receive policy-gradient updates.

---

## 6. Why it works (intuition)

- RL rewards the model for being correct, not for following a preset search script.
- Multi-turn search emerges because additional search calls are useful to increase answer correctness.
- Masking retrieved tokens prevents unstable gradients and keeps credit assignment clean.

---

## 7. Failure modes to watch

- **Search loops:** model keeps searching without converging.
- **Premature answering:** answers too early without sufficient evidence.
- **Query drift:** later queries become irrelevant due to noisy context.
- **Context overflow:** retrieved snippets crowd out reasoning context.

Mitigations:

- Cap max turns; add repetition penalties on identical queries.
- Summarize retrieved docs before appending (auxiliary summarizer).
- Force a structured tool-call format.

---
