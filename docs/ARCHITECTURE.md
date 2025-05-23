# AVA System Architecture (Foundational)

**Current Focus:** This document outlines the architecture for the foundational version of AVA (Afsah's Virtual Assistant). This version prioritizes core functionality, reliability, and a solid framework for future enhancements.

Our long-term vision for AVA is expansive and detailed in `docs/ARCHITECTURE_VISION.md` and `docs/STRUCTURE_VISION.md`.

## I. Core Principles for Foundational AVA

*   **Simplicity:** Employ straightforward and well-understood technologies.
*   **Modularity:** Design components with clear responsibilities for easier development, testing, and future upgrades.
*   **Practicability:** Focus on features that can be reliably implemented with current resources.
*   **User-Centricity:** Prioritize a useful and intuitive user experience.

## II. Foundational Architecture Overview

The foundational AVA system is composed of the following key layers and components:

```mermaid
graph TD
    A[User Interface (CLI / Web)] --> B{Command Parser};
    B -- Text/Voice Input --> C[Interaction Manager];
    C --> D{Core Logic & LLM Backend};
    D --> E[Tool Interface];
    E --> F[Weather API];
    E --> G[Calendar API];
    E --> H[Reminders (Local DB)];
    E --> I[Computation Module];
    E --> J[Time/Date Module];
    F --> C;
    G --> C;
    H --> C;
    I --> C;
    J --> C;
    C --> K[Response Formatter];
    K -- Text/Voice Output --> A;
    L[Logging Service] --> C;
    L --> D;
    L --> E;
```

**Layers & Components:**

1.  **User Interface (UI) Layer:**
    *   **Description:** The primary point of interaction for the user.
    *   **Initial Implementation:** Command-Line Interface (CLI) built with Python (e.g., using `argparse`).
    *   **Potential Enhancement:** A simple web-based UI (e.g., using Flask or Streamlit).
    *   **Interaction Modalities:**
        *   **Text:** Direct text input.
        *   **Voice (Optional):** Speech-to-Text (STT) for input and Text-to-Speech (TTS) for output, integrated via libraries like `SpeechRecognition` and `gTTS` or `pyttsx3`.

2.  **Command Parser:**
    *   **Description:** Interprets user input to determine intent and extract relevant information.
    *   **Implementation:**
        *   **Rule-Based Matching:** Uses regular expressions or keyword spotting for predefined commands (e.g., "What is the weather?", "Set a reminder").
        *   **Natural Language Understanding (NLU) Fallback:** For more complex or ambiguous queries, the input can be passed to the Core Logic & LLM Backend for intent extraction and entity recognition.

3.  **Interaction Manager:**
    *   **Description:** Orchestrates the flow of information between the UI, Core Logic, and Tools. It routes parsed commands to the appropriate backend components and ensures responses are delivered back to the user.
    *   **Implementation:** Python-based logic.

4.  **Core Logic & LLM Backend:**
    *   **Description:** The "brain" of the foundational AVA. It handles general queries, conversation management, and directs tasks to specific tools when needed.
    *   **Implementation:**
        *   **Large Language Model (LLM):** Leverages a pre-trained LLM (e.g., via OpenAI API, Anthropic API, or a locally hosted model via Ollama).
        *   **System Prompt:** A carefully defined system prompt guides the LLM's persona, tone (polite, helpful, concise), and basic operational guidelines.
        *   **Session Management (Basic):** Maintains a short history of the conversation for contextual understanding.

5.  **Tool Interface & Modules:**
    *   **Description:** A standardized way for the Core Logic to interact with various tools that provide specific functionalities.
    *   **Implementation:** Each tool is a Python module with a defined interface.
    *   **Initial Tools:**
        *   **Weather Tool:** Connects to a public weather API (e.g., OpenWeatherMap) to fetch weather information.
        *   **Calendar Tool:** Integrates with a calendar service (e.g., Google Calendar API) to manage events.
        *   **Reminders Tool:** Stores and retrieves reminders, potentially using a local SQLite database or a simple file (e.g., JSON).
        *   **Computation Tool:** Handles basic mathematical calculations (either via LLM or a safe local evaluation method).
        *   **Time/Date Tool:** Provides current time and date information using Python's `datetime` module.

6.  **Response Formatter:**
    *   **Description:** Takes the output from the Core Logic or Tools and formats it into a user-friendly textual (or speech) response.
    *   **Implementation:** Python-based string formatting and logic.

7.  **Logging Service:**
    *   **Description:** Records key interactions, errors, and system events for debugging and monitoring.
    *   **Implementation:** Utilizes Python's `logging` module to write to log files.

## III. Data Flow Example (Weather Query)

1.  User types: "What's the weather like in London?" (or speaks it via STT).
2.  Command Parser identifies "weather" intent and "London" as the location.
3.  Interaction Manager routes this to the Core Logic.
4.  Core Logic, via the Tool Interface, calls the Weather Tool with "London".
5.  Weather Tool queries the OpenWeatherMap API.
6.  API returns weather data for London.
7.  Weather Tool processes data and returns it to the Core Logic.
8.  Core Logic passes the processed data to the Response Formatter.
9.  Response Formatter creates a sentence like: "The current weather in London is..."
10. UI displays the text (or speaks it via TTS).
11. Logging Service records the transaction.

## IV. Technology Stack (Foundational)

*   **Primary Language:** Python
*   **LLM Interaction:** APIs (OpenAI, Anthropic) or local via Ollama.
*   **CLI:** Python's `argparse`.
*   **Web UI (Optional):** Flask, Streamlit.
*   **Databases (for Reminders):** SQLite (initially).
*   **APIs:** Standard HTTP request libraries (e.g., `requests` in Python).
*   **Voice (Optional):** `SpeechRecognition`, `gTTS`, `pyttsx3`.

This foundational architecture provides a robust starting point for AVA, focusing on delivering core assistant functionalities effectively. Future iterations will build upon this base to incorporate more advanced features progressively. 