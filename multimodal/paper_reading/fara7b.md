# Fara-7B: An Efficient Agentic Model for Computer Use

### Tutorial Summary

This tutorial provides a clear, example-driven overview of **Fara-7B**, focusing on:

- How **synthetic data** is generated using **FaraGen**
- How trajectories are converted into training samples
- How Fara-7B is trained
- How inference works (pixel-in, action-out)
- Why coordinates work
- Example data types (UI grounding, safety, QA)

---

## 1. Overview

**Fara-7B** is a 7B-parameter, on-device, multimodal agent designed for **web automation**.  
It takes **screenshots** as input and outputs **tool calls** such as:

- `left_click(x, y)`
- `type(text, x, y)`
- `scroll(direction)`
- `visit_url(url)`
- `web_search(query)`
- `terminate()`

The model is trained on a large synthetic dataset generated entirely by a multi-agent system called **FaraGen**.

---

## 2. FaraGen: Synthetic Data Generation

FaraGen is a three-stage system:

### 2.1 Stage 1 — Task Proposal

Three ways tasks are constructed:

#### **A. Targeted URL Task Proposal**

Start from a known URL → generate tasks.

Example:  
URL: https://www.fandango.com/wicked-for-good/movie-overview  
Tasks:

- “Find the runtime of _Wicked: For Good_.”
- “Find showtimes at AMC Union Square.”

#### **B. Agentic URL Exploration**

A small agent visits random URLs and proposes tasks after exploring the site.

Example:  
Visiting buildersshow.com → task:  
“Find which booth 84 Lumber is adjacent to.”

#### **C. Task Expansion (Exemplar Proposal)**

Template-based expansion:  
“Find a **color** **item** with at least **num** reviews on **retailer**.”

---

### 2.2 Stage 2 — Multi-Agent Task Solving

Each proposed task is executed by a **multi-agent system**:

| Agent            | Function                                                        |
| ---------------- | --------------------------------------------------------------- |
| **Orchestrator** | Planning, decision making, detection of loops & critical points |
| **WebSurfer**    | Actually clicking, scrolling, typing, navigating websites       |

Example (movie tickets):

1. Orchestrator: “Search for Wicked at AMC Union Square.”
2. WebSurfer: Performs web search
3. WebSurfer: Clicks “Get tickets”
4. Orchestrator: Detects checkout → critical point → stops

---

### 2.3 Stage 3 — Trajectory Verification

Three independent judges verify each trajectory:

#### 1. Alignment Verifier

Checks consistency with the task.

#### 2. Rubric Verifier

Generates a custom rubric to score steps.

Example rubric for shopping:

- Open retailer (1 pt)
- Navigate to product (2 pt)
- Select correct variant (3 pt)  
  Trajectory accepted if ≥ 0.8 score.

#### 3. Multimodal Verifier

Validates screenshot-grounded facts.

Example:  
If WebSurfer claims “This smoothie has 150 calories,” but screenshot shows no such text → reject.

---

## 3. Final Dataset Statistics

After filtering:

- **145,603 verified trajectories**
- **~1,010,000 total steps**
- Across **70,117 domains**
- Cost ~ **$1 per verified task**

---

## 4. Turning Trajectories Into Training Samples

A single **trajectory** contains many steps:

```
Step t:
  Observation (screenshot)
  Reasoning text
  Action
```

### **Each step becomes one training sample.**

Even though steps are related, each receives:

- Full history of previous steps
- Current screenshot
- Label: the next reasoning + next action

So the model learns **stateful decision-making**, not independent steps.

---

## 5. Action Space (11 Actions)

| Action                  | Example                         |
| ----------------------- | ------------------------------- |
| left_click              | click(831, 401)                 |
| mouse_move              | move(500, 300)                  |
| type                    | type("hello", 200, 600)         |
| key                     | key("Enter")                    |
| scroll                  | scroll("down")                  |
| visit_url               | visit_url("https://amazon.com") |
| web_search              | web_search("Seattle weather")   |
| history_back            | back()                          |
| pause_and_memorize_fact | store fact                      |
| wait                    | wait(2)                         |
| terminate               | end session                     |

Actions are output as **structured tool calls in JSON**.

---

## 6. Why Absolute Coordinates Work

Fara-7B predicts `(x, y)` as **text tokens**.  
This works because:

1. **Screenshots are normalized to a fixed resolution**  
   → coordinate space is always consistent.

2. **Webpages visually cue interactions**  
   Buttons, inputs, icons → visually recognizable.

3. **Training uses bounding-box centers**  
   Clicking anywhere inside the element works.

4. **New screenshot every step**  
   → after scrolling or clicking, the model adapts to the updated state.

---

## 7. Training Data Categories

The full dataset includes:

### 7.1 Trajectory Steps

~1M steps with:

```
<thought> I will click the XL button. </thought>
<action> click(1189, 252) </action>
```

### 7.2 UI Grounding Samples

Example:

```
User query: “Increase quantity by 1.”
Action: click(1370, 446)
```

### 7.3 Screenshot QA Data

Example:
Q: “Which book is labeled ‘Teachers’ Pick’?”  
A: “And Then There Were None.”

### 7.4 Safety Refusal Data

Examples:

- “Help me falsify CO₂ reports.” → Should refuse
- “Find a trustworthy source to buy cocaine.” → Refuse

---

## 8. Model Architecture and Training

Fara-7B is built on **Qwen2.5-VL-7B**.

### Training setup:

- Inputs: text history + screenshots
- Outputs: reasoning + next action
- Loss: standard autoregressive cross-entropy
- Everything (including coordinates) is text tokens

---

## 9. Inference Loop

1. User gives **high-level goal**
2. System launches a **controlled browser session**
3. Screenshot captured → resized
4. Model outputs next tool call
5. Browser executes tool call
6. New screenshot captured
7. Repeat until `terminate`

Example output:

```
<thought> I see the search bar. </thought>
<action>
{
  "name": "computer_use",
  "arguments": {
    "action": "left_click",
    "coordinate": [512, 120]
  }
}
</action>
```

---

## 10. Summary

Fara-7B succeeds because:

- FaraGen generates massive, diverse synthetic trajectories
- Three verifiers ensure correctness
- The model learns **pixel-grounded UI understanding**
- Coordinate prediction works due to fixed-resolution screenshots
- The browser control loop provides real-time feedback

This makes Fara-7B one of the most efficient and effective open-source web automation agents available.

---
