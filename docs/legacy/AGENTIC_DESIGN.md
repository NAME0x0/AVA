# AVA Agentic Design 🧠

**Project Vision:** To establish AVA as a compact, powerful, and locally-run agentic AI model capable of serving as a daily driver for advanced tasks on an NVIDIA RTX A2000 GPU with 4GB VRAM.

This document details the architecture of AVA's agentic capabilities, focusing on how it understands intent, uses tools, produces reliable outputs, and performs reasoning.

**Related Documents:**
*   `[ARCHITECTURE.md](./ARCHITECTURE.md)`: Overall system design.
*   `[OPTIMIZATION_STRATEGIES.md](./OPTIMIZATION_STRATEGIES.md)`: Model optimization and fine-tuning.
*   `[ROADMAP.md](./ROADMAP.md)`: Development phases.
*   `[UI_CONNECTIVITY.md](./UI_CONNECTIVITY.md)`: User interfaces and remote access.

---

## I. Core Agentic Loop

AVA's operation revolves around an agentic loop:
1.  **Perception:** Receive user input (CLI, GUI, remote).
2.  **Understanding (NLU):** Interpret the user's intent and extract key information. This is primarily handled by the core LLM.
3.  **Planning & Reasoning:** Decide on a course of action. This may involve:
    *   Determining if an external tool/function is needed.
    *   Breaking down a complex task into steps (Chain-of-Thought).
    *   Querying internal knowledge or deciding to access external data via MCP.
4.  **Action:**
    *   If a tool is needed, generate structured arguments and execute the tool (via `FunctionCaller`).
    *   If data is needed via MCP, interact with the MCP Host.
    *   If internal processing is sufficient, generate intermediate thoughts or a direct response.
5.  **Response Generation (NLG):** Formulate a coherent and structured response to the user, incorporating tool results or reasoned conclusions.
6.  **Learning (Implicit/Explicit):** Dialogue history is maintained, and future fine-tuning will incorporate feedback from interactions.

## II. Function Calling & Tool Use

*   **Objective:** Enable AVA to extend its capabilities by interacting with external tools, APIs, and local system functions.
*   **Methodology:**
    *   **Fine-tuning for Tool Detection:** The core LLM (Gemma 3n + QLoRA) will be fine-tuned on synthetic and real datasets to reliably detect when a query requires an external tool.
    *   **Structured Argument Generation:** The LLM will be trained to output a structured representation (ideally JSON, or a parsable string format) specifying the tool name and its arguments.
        *   Example LLM Output (String-based): `"[TOOL_CALL: get_weather(location=\"London\", unit=\"celsius\")]"`
        *   Example LLM Output (JSON-based, if native function calling supported/fine-tuned): 
            ```json
            {
              "tool_calls": [ 
                { "id": "call_abc123", "type": "function", 
                  "function": { "name": "get_weather", "arguments": "{\"location\": \"London\", \"unit\": \"celsius\"}"} 
                }
              ]
            }
            ```
    *   **Tool Execution:** A `FunctionCaller` module (see `src/ava_core/function_calling.py`) will be responsible for:
        *   Parsing the LLM's tool call request.
        *   Validating the tool name and arguments.
        *   Executing the corresponding Python function or API call.
        *   Returning the result to the agent loop (often fed back to the LLM for summarization/response generation).
*   **Tool Library (`src/tools/`):**
    *   Each tool will be a well-defined Python module/class (e.g., `calculator.py`, `web_search.py`, `calendar_api.py`).
    *   Tools must have clear input/output specifications.
*   **Error Handling:** Robust error handling for tool execution failures, invalid arguments, or API issues.

## III. Structured Output

*   **Objective:** Ensure AVA produces reliable, parsable outputs (e.g., JSON, formatted SQL, specific text structures) for downstream processing, tool interoperability, and predictable UI display.
*   **Methodology:**
    *   **Prompt Engineering:** Instructing the LLM to generate output in a specific format.
    *   **Fine-tuning for Format Adherence:** Training AVA on datasets where outputs conform to desired schemas.
    *   **Output Parsing & Validation:**
        *   Using libraries like LlamaIndex (Pydantic Programs, Output Parsers) to define schemas and parse LLM outputs against them.
        *   Custom parsing logic for simpler or non-JSON structures.
        *   Retry mechanisms or feedback loops if parsing fails (e.g., re-prompting the LLM with the error).
*   **Use Cases:** Generating JSON for API calls, formatting data for tables in the UI, producing SQL queries.

## IV. Reasoning Mechanisms

*   **Objective:** Enable AVA to perform multi-step reasoning, problem decomposition, and more complex thought processes.
*   **Methodology:**
    *   **Chain-of-Thought (CoT) Prompting:**
        *   Encouraging the LLM to generate explicit step-by-step reasoning before providing a final answer.
        *   Achieved through specific instructions in the system prompt or few-shot examples.
        *   The `ReasoningEngine` (see `src/ava_core/reasoning.py`) can assist in formulating or managing CoT processes.
    *   **(Advanced) Reflection/Self-Critique:** Potentially fine-tuning AVA to review its own reasoning or tool use plans, identify flaws, and correct them. This can be guided by a separate "critic" model or specific prompting techniques.
    *   **(Future) RL-of-Thoughts (RLoT):** Training a lightweight "navigator" model via reinforcement learning to adaptively select and combine reasoning blocks (primitive thoughts or tool calls) for complex tasks. This is a longer-term research direction.
*   **Integration:** Reasoning steps can be interleaved with tool use. For example, AVA might reason about what information is missing, decide to use a web search tool, reason about the search results, and then synthesize an answer.

## V. Model Context Protocol (MCP) Integration

*   **Objective:** Provide AVA with direct, real-time, and secure access to local files, APIs, and databases, as an alternative or complement to traditional RAG (Retrieval Augmented Generation) which relies on embeddings and vector search.
*   **Concept:** (As per [https://modelcontext.org/](https://modelcontext.org/))
    *   **AVA as MCP Host:** AVA's core logic will act as an MCP Host.
    *   **MCP Servers:** Lightweight servers that expose specific data sources (e.g., a file system server, a database server, an API wrapper server) via the MCP standard.
*   **Benefits for AVA:**
    *   **Reduced Hallucinations:** Direct access to fresh, factual data.
    *   **Lower Computational Cost:** Avoids embedding and vector search for many data access tasks.
    *   **Enhanced Security/Privacy:** Data can remain local; only necessary information is exchanged.
    *   **Streamlined Tool Orchestration:** Simplifies adding new data sources/tools if they conform to the MCP interface.
*   **Implementation (`src/mcp_integration/`):
    *   `mcp_host.py`: Logic for AVA to discover and communicate with MCP Servers.
    *   `mcp_servers/`: Example implementations of MCP Servers for common local resources (e.g., file access).
    *   AVA will be fine-tuned to recognize when an MCP interaction is more appropriate than a general tool call or relying on its internal knowledge.
*   **Workflow Example:**
    1.  User asks: "Summarize my document `meeting_notes_today.txt`."
    2.  AVA (as MCP Host), recognizing the local file reference, queries a local MCP File Server.
    3.  MCP File Server provides the content of `meeting_notes_today.txt`.
    4.  AVA summarizes the content and presents it to the user.

## VI. Agentic Workflow Orchestration

The `Agent` class (`src/ava_core/agent.py`) will be central to orchestrating these components:
*   It will manage the state of the interaction.
*   It will route user input through the NLU, Dialogue Manager, Reasoning Engine, Function Caller, and MCP Interface as needed.
*   It will synthesize information from various sources to generate the final response.

Continuous fine-tuning and evaluation will be critical to ensure these agentic components work together seamlessly and reliably within the 4GB VRAM constraint.

_(This document will be expanded with detailed architectural diagrams, implementation guides, and examples for each agentic component.)_ 