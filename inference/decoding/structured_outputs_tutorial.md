# Structured Outputs in LLMs: A Practical Tutorial

> Goal: understand what *structured outputs* are, why they’re more reliable than “prompt → parse → retry”,
> and how *schema-constrained decoding* works under the hood.

## 1. What are structured outputs?

A **structured output** is an LLM response that is **guaranteed** to follow a predefined structure—most commonly a JSON object that conforms to a schema.

Instead of hoping the model returns valid JSON and then fixing it with retries, structured-output systems enforce the structure **during decoding**.

### Prompting vs. structured outputs

- **Traditional**: prompt the model to "return JSON" → parse → if invalid, retry / repair.
- **Structured outputs**: provide a schema/grammar → the decoder only allows tokens that keep the output valid.

That’s why structured outputs are typically much more reliable in production pipelines.

## 2. Why JSON is common (but not the only option)

Most providers use **JSON + JSON Schema** because it is:

- Easy to validate
- Widely supported (web services, databases, typed APIs)
- Friendly to downstream parsing (no custom parsers)

But the idea is more general: you can constrain decoding using **any formal grammar**, including custom DSLs.

## 3. Function calling is a special case of structured outputs

You can think of **function/tool calling** as **structured output + execution semantics**.

- Structured output: "produce data that matches this schema"
- Function calling: "produce data that matches this schema, and the runtime interprets it as a tool invocation"

Example conceptual schema:

```json
{
  "name": "get_weather",
  "arguments": { "city": "Seattle", "unit": "celsius" }
}
```

So: **every function call is structured output**, but not every structured output is a function call.

## 4. How schema-constrained decoding works

LLMs generate text token-by-token. At each step, the model produces **logits** over the vocabulary.

In unconstrained decoding, you sample from those logits (greedy / top-k / nucleus / etc.).

In **constrained decoding**, you first compute the set of tokens that are valid *given the schema and the current prefix*, then you **mask** all invalid tokens.

### The core algorithm

At step *t*:

1. Keep a **state** representing what the partial output means under the schema/grammar.
2. Compute `allowed_next_tokens(state)`.
3. Mask invalid tokens: set their logit to negative infinity.
4. Sample from the remaining valid tokens.

Mathematically, if the model gives logits $\ell_i$ for token $i$, we construct masked logits:

$$
\tilde{\ell}_i =
\begin{cases}
\ell_i, & i \in A \\
-\infty, & i \notin A
\end{cases}
$$

where $A$ is the allowed token set for the current state.

This guarantees the output stays valid by construction.

## 5. A minimal constrained decoder (from scratch)

Below is a **character-level** constrained decoder (easier to read than BPE token constraints).
It guarantees output matches this fixed schema:

```json
{"city":"<1–20 lowercase letters>","unit":"celsius"|"fahrenheit"}
```

### Key idea

- A tiny **finite-state machine (FSM)** encodes the schema.
- At each step, `allowed` is computed from the FSM state.
- The sampler masks all characters not in `allowed`.

```python
import math, random
random.seed(0)

VOCAB = list('{"}:,abcdefghijklmnopqrstuvwxyz')  # allowed chars

def logits(prefix: str):
    # toy 'LM' logits, prefix-dependent for variety
    r = random.Random(hash(prefix) & 0xffffffff)
    return [r.uniform(-2.0, 2.0) for _ in VOCAB]

def sample_masked(logits, allowed):
    probs = [math.exp(l) if c in allowed else 0.0 for c, l in zip(VOCAB, logits)]
    s = sum(probs)
    if s == 0:
        raise RuntimeError(f'Dead end, allowed={allowed}')
    x, acc = random.random() * s, 0.0
    for c, p in zip(VOCAB, probs):
        acc += p
        if acc >= x:
            return c
    return VOCAB[-1]

HEAD = '{"city":"'
MID  = '","unit":"'
TAIL = '"}'

def constrained_decode(max_steps=500):
    out, state, i = '', 'HEAD', 0
    city_len = 0
    unit_opts = ['celsius', 'fahrenheit']
    unit_typed = ''

    for _ in range(max_steps):
        if state == 'HEAD':
            allowed = {HEAD[i]}
        elif state == 'CITY':
            allowed = set('abcdefghijklmnopqrstuvwxyz')
            if city_len >= 1:
                allowed.add('"')
            if city_len >= 20:
                allowed = {'"'}
        elif state == 'MID':
            allowed = {MID[i]}
        elif state == 'UNIT':
            allowed = set()
            for lit in unit_opts:
                if lit.startswith(unit_typed):
                    if len(unit_typed) == len(lit):
                        allowed.add('"')
                    else:
                        allowed.add(lit[len(unit_typed)])
        elif state == 'TAIL':
            allowed = {TAIL[i]}
        else:
            raise RuntimeError('bad state')

        ch = sample_masked(logits(out), allowed)
        out += ch

        if state == 'HEAD':
            i += 1
            if i == len(HEAD):
                state, i = 'CITY', 0
        elif state == 'CITY':
            if ch == '"':
                state, i = 'MID', 0
            else:
                city_len += 1
        elif state == 'MID':
            i += 1
            if i == len(MID):
                state, i = 'UNIT', 0
        elif state == 'UNIT':
            if ch == '"':
                state, i = 'TAIL', 0
            else:
                unit_typed += ch
        elif state == 'TAIL':
            i += 1
            if i == len(TAIL):
                return out

    raise RuntimeError('did not terminate')

print(constrained_decode())
```

### What to notice

- The “model” can be random and still produce valid output because **constraints dominate**.
- `allowed` is the heart of the system.
- Real systems do the same thing at the **token** level (BPE), not the character level.

## 6. How production systems build `allowed_next_tokens`

In real structured outputs, constraints come from:

1. **JSON Schema** → compile to a grammar (or incremental validator)
2. **Grammar** → compile to an automaton / parser state machine
3. **Tokenizer-aware mapping** → map allowed *strings* to allowed *tokens*

This last step (tokenizer-aware mapping) is tricky and is why libraries exist.

## 7. Open-source libraries to study

Here are popular open-source options (good for reading code):

- **Outlines**: ergonomic structured generation (JSON, regex, CFG) using constrained decoding.
- **Guidance / llguidance**: grammar-first constraints and templating.
- **xgrammar**: fast grammar constraints, often used as a backend for structured outputs.
- **vLLM structured outputs**: integrates schema/grammar constraints into a high-performance serving engine.

If your goal is to learn the mechanics, Outlines is a great starting point because the abstraction is clean.

## 8. When structured outputs can still fail

Structured outputs can guarantee *format*, but not necessarily *semantic correctness*.

Examples:

- Wrong field value (e.g., temperature is unrealistic)
- Hallucinated content that still fits the schema
- Overly constrained schema causing low-quality answers

In practice you often combine:

- Structured outputs for format guarantees
- Additional validation checks (business rules)
- Optional post-processing (normalization)

## 9. Practical tips

- Keep schemas minimal: constrain what you *must* rely on.
- Use enums (`Literal`) for closed sets (tool names, categories).
- Prefer integers/floats only when you truly need them; numeric hallucinations are common.
- Separate extraction (structured) from reasoning (free text) if you need auditability.

## 10. Next steps

If you want to go deeper, the most educational extensions are:

1. Swap the character-level decoder for **token-level** constraints (BPE).
2. Auto-compile a tiny subset of JSON Schema to an FSM.
3. Add safety constraints (e.g., enums that avoid disallowed categories).

---

### Appendix: Glossary

- **Schema**: a specification of the structure and types of data (often JSON Schema).
- **Grammar**: a formal language definition; JSON can be described by a grammar.
- **FSM/DFA**: finite-state machines used to track what’s valid next.
- **Constrained decoding**: masking invalid tokens during generation.
