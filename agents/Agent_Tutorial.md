# 🤖 Agentic Systems Tutorial — From ReAct to AutoGen

This tutorial introduces the **concept of agents** in LLM systems — what they are, how they reason and act, and how to implement them using **ReAct loops** and frameworks like **AutoGen**.

---

## 🧠 1. What Is an Agent?

An **agent** is an LLM-powered system that can **reason, act, and interact** with external environments.

Agents go beyond static text generation by combining:
- **Reasoning:** deciding what to do next,
- **Acting:** executing functions, tools, or code,
- **Observing:** reading results and updating its plan.

---

## 🔁 2. The ReAct Framework

**ReAct** (*Reason + Act*, Yao et al. 2022) alternates between reasoning and acting.

### Core Loop

```
Thought → Action → Observation → Thought → … → Final Answer
```

Example trace:

```
User: What’s the population of the capital of France divided by two?

Thought 1: I need to find the capital of France.
Action 1: lookup("capital of France")
Observation 1: "Paris"
Thought 2: I now need its population.
Action 2: lookup("population of Paris")
Observation 2: "2.1M"
Thought 3: Half of 2.1M is 1.05M.
Final Answer: 1.05M
```

---

## ⚙️ 3. Core Components

| Component | Role |
|------------|------|
| **LLM** | Generates reasoning (“Thoughts”) and structured tool calls (“Actions”) |
| **Tools / Functions** | Executable APIs that perform actions |
| **Runtime / Orchestrator** | Executes tools and feeds observations back |
| **Memory (optional)** | Stores prior steps or results |

---

## 🧩 4. Manual ReAct Example (no framework)

```python
# agent_react_sketch.py
from typing import Callable, Dict

tools: Dict[str, Callable] = {}

def register(name):
    def deco(f):
        tools[name] = f
        return f
    return deco

@register("lookup")
def lookup(query: str):
    db = {
        "capital of France": "Paris",
        "population of Paris": "2.1M"
    }
    return db.get(query.lower(), "unknown")

def simple_agent():
    print("Thought: I should find the capital of France.")
    result1 = tools["lookup"]("capital of France")
    print("Action: lookup('capital of France') →", result1)

    print("Thought: Now I find its population.")
    result2 = tools["lookup"]("population of Paris")
    print("Action: lookup('population of Paris') →", result2)

    print("Thought: Half of 2.1M is 1.05M.")
    print("Final Answer: 1.05M")

if __name__ == "__main__":
    simple_agent()
```

Output:
```
Thought: I should find the capital of France.
Action: lookup('capital of France') → Paris
Thought: Now I find its population.
Action: lookup('population of Paris') → 2.1M
Thought: Half of 2.1M is 1.05M.
Final Answer: 1.05M
```

---

## 🧠 5. Agents in AutoGen

AutoGen provides classes for multi-agent orchestration:
- `AssistantAgent` — the reasoning LLM.
- `UserProxyAgent` — executes functions or code on model’s behalf.

### Example

```python
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent(name="assistant", llm_config={"model": "gpt-4o"})
user = UserProxyAgent(name="user", code_execution_config={"use_docker": False})

def add(a: int, b: int) -> int:
    return a + b

assistant.register_for_llm(name="add", description="Add two integers")(add)
user.register_for_execution(name="add")(add)

user.initiate_chat(assistant, message="Please call add(2, 3) and tell me the result.")
```

The model decides when to call `add`, the runtime executes it, and returns the result.

---

## 🧮 6. Planner–Coder Multi-Agent Example

```python
from autogen import AssistantAgent, UserProxyAgent

planner = AssistantAgent(
    name="planner",
    system_message="You decompose tasks into steps.",
    llm_config={"model": "gpt-4o-mini"}
)

coder = UserProxyAgent(
    name="coder",
    code_execution_config={"use_docker": False},
    system_message="You write and run Python code."
)

planner.initiate_chat(coder, message="Calculate 1+2+3+4+5 and summarize.")
```

Planner sends plan → coder executes → result returned → planner finalizes.

---

## 🔗 7. Popular Frameworks

| Framework | Focus | Notes |
|------------|--------|-------|
| **AutoGen** | Multi-agent orchestration | Microsoft research framework |
| **LangChain** | Chain & tool management | Widely used in production |
| **Smolagents** | Lightweight, minimal ReAct loop | Hugging Face |
| **OpenDevin** | Developer GUI agent | Open source dev assistant |

---

## 🧭 8. Key Takeaways

- Agents = LLM + tools + orchestrator loop.  
- ReAct is the backbone pattern for reasoning + acting.  
- Frameworks like AutoGen abstract this into agents and roles.  
- Modern models (GPT-4o, Qwen3-VL) are *tool-aware*, meaning they can emit JSON calls directly.  
- True “agentic behavior” arises when these calls are **executed and observed**.

---

## 📚 9. References

- Yao et al., *“ReAct: Synergizing Reasoning and Acting in Language Models”*, 2022.  
- Microsoft AutoGen: [github.com/microsoft/autogen](https://github.com/microsoft/autogen)  
- Hugging Face Smolagents: [github.com/huggingface/smolagents](https://github.com/huggingface/smolagents)  
- OpenAI Function Calling Docs: [platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling)

---
