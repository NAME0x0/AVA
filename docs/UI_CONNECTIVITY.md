# AVA User Interfaces & Connectivity 🖥️↔️🌐

**Project Vision:** To establish AVA as a compact, powerful, and locally-run agentic AI model capable of serving as a daily driver for advanced tasks on an NVIDIA RTX A2000 GPU with 4GB VRAM.

This document details the design and implementation of AVA's user interfaces (CLI and GUI) and its remote access capabilities, ensuring a seamless and responsive user experience.

**Related Documents:**
*   `[ARCHITECTURE.md](./ARCHITECTURE.md)`: Overall system design.
*   `[AGENTIC_DESIGN.md](./AGENTIC_DESIGN.md)`: Agentic capabilities.
*   `[ROADMAP.md](./ROADMAP.md)`: Development phases.

---

## I. User Interface (UI) Philosophy

AVA aims to provide flexible interaction modes to suit different user preferences and tasks:
*   **Power & Scriptability:** A robust Command Line Interface (CLI) for advanced users, automation, and scripting.
*   **Ease of Use & Rich Interaction:** An intuitive Graphical User Interface (GUI) for conversational interaction and visualization of agentic processes.
*   **Responsiveness:** Both interfaces should provide quick feedback, leveraging token streaming for generated responses.

## II. Command Line Interface (CLI)

*   **Objective:** Provide a direct, efficient way to interact with AVA for tasks, scripting, and development.
*   **Technology:** Python, using libraries like `Typer` or `Click` (preferred for their ease of use and feature richness over `argparse`).
    *   `Typer` is chosen for its modern Python type hint integration.
*   **Core Functionality (`src/cli/main.py`):
    *   **Prompting:** `ava query "Your natural language query to AVA"`
    *   **Interactive Mode:** `ava chat` (for a persistent conversational session).
    *   **Tool Invocation (Directly, if needed for testing/specific scripts):** `ava tool_exec <tool_name> --arg1 value1 --arg2 value2`
    *   **Configuration Management:** `ava config set <key> <value>`, `ava config get <key>`
    *   **Status & Diagnostics:** `ava status`, `ava logs`
    *   **Model Management (Interfacing with Ollama):** `ava model list`, `ava model select <model_name>`
*   **Output Handling:**
    *   Plain text for simple responses.
    *   Structured output (e.g., JSON) via flags (`--output json`) for scriptability.
    *   Token streaming for long-generated responses to improve perceived responsiveness.
*   **Error Handling:** Clear error messages for invalid commands, tool failures, or connection issues.

## III. Graphical User Interface (GUI)

*   **Objective:** Offer an intuitive, visually appealing, and feature-rich interface for interacting with AVA, particularly for conversational and exploratory use cases.
*   **Recommended Foundation:** [Open WebUI](https://github.com/open-webui/open-webui)
    *   **Rationale:** Self-hosted, offline-capable, supports Ollama out-of-the-box, actively maintained, and provides a good baseline ChatGPT-like experience.
    *   GPU acceleration via Docker for Open WebUI is a key advantage.
*   **Integration with AVA:**
    *   AVA's core model (quantized Gemma 3n + QLoRA adapters) will be served locally via **Ollama**.
    *   Open WebUI will connect to this local Ollama instance as its backend LLM.
    *   The `Modelfile` for AVA in Ollama will define its system prompt, parameters, and how it should behave (see `docs/INSTALLATION.md`).
*   **Key Open WebUI Features to Leverage:**
    *   ChatGPT-style interface.
    *   LLM model selection (AVA will be one of the models).
    *   Parameter playground (temperature, top_p, etc.).
    *   Knowledge collections (document upload for RAG-like capabilities, complementing MCP).
    *   Live web search (if configured, can be a tool AVA uses).
*   **Advanced Agentic UI Inspirations (Potential Customizations or Future Enhancements to Open WebUI or a Custom UI):**
    *   **Interactive Tool Call Visualization:** Clearly showing when AVA decides to use a tool, the arguments, and the result.
    *   **Project Context Awareness:** (More complex) Allowing AVA to be aware of a user's current project directory or context for more relevant assistance (potentially via MCP).
    *   **Actionable Outputs:** Buttons or UI elements for actions based on AVA's output (e.g., copy code snippet, run generated command with confirmation, apply diff).
    *   **User Configurability:** Settings for agent behavior, permissions for tool use.
    *   **Feedback Mechanisms:** Easy ways for users to rate responses or report issues.
*   **Implementation (`src/gui/open_webui_integration.py` - if custom logic needed beyond Ollama config):
    *   Primarily involves configuring Ollama and Open WebUI correctly.
    *   Any custom backend logic required to bridge specific AVA features to Open WebUI if not directly supported.

## IV. Remote Access & Token Broadcasting

*   **Objective:** Allow users to access their local AVA instance securely from other devices (e.g., phone, another computer) over the internet.
*   **Methodology:**
    1.  **Local Server:** AVA's capabilities (via Ollama or a custom Python server if developed) will be exposed on a local network port.
    2.  **Secure Tunneling:**
        *   Utilize services like `Localtonet` or `ngrok` to create a secure public URL that tunnels traffic to the local AVA server.
        *   **Security:** This is critical. Implementations must include:
            *   Basic Authentication (username/password) on the tunnel or server.
            *   IP Whitelisting (if possible and practical for the user).
            *   HTTPS enforced by the tunneling service.
    3.  **Token Streaming for Responsive UI:**
        *   To avoid long waits for complete responses, especially over potentially slower remote connections, AVA's LLM responses will be streamed token by token.
        *   **Recommended Method:** Server-Sent Events (SSE) for efficient, uni-directional streaming of text data over HTTP. This is well-suited for LLM token streaming and is generally simpler to implement than WebSockets for this use case.
        *   The local server (Ollama or custom) must support SSE, and the client (Open WebUI, or custom remote clients) must be able to consume SSE streams.
*   **Benefits:**
    *   **Privacy & Control:** Inference still happens locally; only the interaction data passes through the tunnel.
    *   **Cost-Effective:** No need for cloud GPU hosting for inference.
    *   **Accessibility:** Use AVA from anywhere with an internet connection.

## V. Implementation Considerations

*   **Ollama as the Backbone:** For ease of setup and management of the local LLM, Ollama is the primary recommended serving mechanism. Both CLI and GUI (Open WebUI) will interface with Ollama.
*   **Configuration:** User-friendly configuration for CLI behavior, Ollama connection, Open WebUI settings, and remote access credentials (e.g., via `config/user_config.yaml` and `.env` files).
*   **Error Handling & Logging:** Robust logging for all UI and connectivity components to aid debugging.

This multi-faceted approach to UI and connectivity ensures that AVA is both powerful for advanced users and accessible for everyday interactions, whether local or remote. 