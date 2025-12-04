# Reading Notes: Building the Intent Engine — How Instacart Is Revamping Query Understanding with LLMs

Reference: https://tech.instacart.com/building-the-intent-engine-how-instacart-is-revamping-query-understanding-with-llms-3ac8051ae7ac

## 1. Overview

Instacart’s search experience depends on accurate Query Understanding (QU). The previous QU system relied on multiple specialized ML models for classification, rewrites, and tagging. While effective for common queries, this system struggled with long-tail searches that are rare, ambiguous, or uniquely phrased. Instacart rebuilt its QU pipeline using Large Language Models (LLMs) to improve generalization, reduce complexity, and enhance relevance for difficult queries.

## 2. Motivation

### 2.1 Challenges with Traditional QU

- User queries tend to be short, vague, and noisy.
- Tail queries lack historical signals, causing brittle predictions.
- Conversion data is noisy and does not reliably represent intent.
- Multiple independent models created a heavy engineering and maintenance burden.
- Complex or creatively phrased queries were hard to classify correctly.

### 2.2 Why LLMs

- LLMs possess strong general semantic knowledge and contextual reasoning abilities.
- A single model can unify many QU tasks traditionally handled by separate pipelines.
- LLMs understand synonyms, substitutions, and ingredient or product equivalences.
- They perform better in zero-shot and few-shot scenarios.

## 3. System Design: Instacart’s Layered LLM Approach

Instacart designed a multi-layer framework to adapt generic LLMs to Instacart’s domain.

### 3.1 Context-Engineering with Retrieval

- Inject Instacart-specific catalog, taxonomy, and historical behavioral signals directly into prompts.
- Ground the model’s reasoning in actual product data.
- Improve factual accuracy and reduce hallucination.

### 3.2 Output Guardrails and Validation

- Validate LLM outputs against internal product taxonomies.
- Apply semantic similarity filtering to ensure alignment with known categories.
- Prevent invalid or hallucinated product names from being surfaced.

### 3.3 Fine-Tuning on Proprietary Data

- Fine-tune smaller open-source LLMs on curated Instacart examples.
- Encodes domain knowledge into model weights.
- Reduces reliance on elaborate prompts.

## 4. Applications within Query Understanding

### 4.1 Query Category Classification

- Retrieve likely categories using historical signals.
- Re-rank using an LLM enhanced with contextual grounding.
- Apply semantic filtering for improved precision and recall.

### 4.2 Query Rewrites

- Generate expansions, synonyms, and broadened rewrites for sparse queries.
- Use structured prompts with few-shot examples.
- Achieved high coverage and precision in rewrite generation.

### 4.3 Semantic Role Labeling (SRL)

- Extract structured attributes such as product type, brand, and modifiers.
- Use a teacher–student system:
  - The offline teacher model (LLM + RAG) produces high-quality interpretations.
  - A fine-tuned lightweight student model handles real-time inference.

### 4.4 Latency and Productionization

- Fine-tuned Llama-based models with LoRA adapters.
- Achieved sub-300ms latency via caching, scaling, and adapter optimizations.
- Deployed infrastructure supports both high throughput and low latency.

## 5. Measured Impact

- Major improvements on tail queries (bottom 2 percent).
- More relevant results with fewer user complaints.
- Over 6 percent reduction in scroll depth needed to find items.
- Approximately 50 percent reduction in negative search-quality feedback.

## 6. Key Lessons

### 6.1 Domain Context Is Essential

Grounding LLMs with real catalog and taxonomy data is critical for accuracy.

### 6.2 Fine-Tuning > Prompting

Fine-tuning produced more consistent and reliable outputs than prompting or RAG-only methods.

### 6.3 Unified LLM Stack Simplifies Engineering

Replacing many model-specific components with one backbone reduces maintenance cost and improves consistency.

### 6.4 Production Constraints Are Real

Latency, compute cost, autoscaling, and caching strategies are essential to successful deployment.

## 7. Broader Implications

Instacart's work demonstrates a modern paradigm for e-commerce search: moving from specialized ML pipelines to unified LLM-based architectures. Their hybrid teacher–student pipeline, grounding strategies, and guardrail mechanisms offer a concrete blueprint for organizations seeking to modernize search and retrieval systems using LLMs.
