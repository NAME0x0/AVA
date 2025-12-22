# AVA API Examples

Complete examples for integrating with AVA programmatically.

## Table of Contents

- [Python SDK](#python-sdk)
- [HTTP API](#http-api)
- [WebSocket Streaming](#websocket-streaming)
- [JavaScript/TypeScript](#javascripttypescript)
- [cURL Examples](#curl-examples)

---

## Python SDK

### Basic Usage

```python
from ava import AVA
import asyncio

async def main():
    # Initialize AVA
    ava = AVA()
    await ava.start()

    # Simple chat (auto-routes to Medulla or Cortex)
    response = await ava.chat("What is Python?")
    print(f"Response: {response.text}")
    print(f"Used Cortex: {response.used_cortex}")
    print(f"Response time: {response.response_time_ms}ms")

    # Cleanup
    await ava.stop()

asyncio.run(main())
```

### Force Deep Thinking

```python
async def deep_analysis():
    ava = AVA()
    await ava.start()

    # Force Cortex for complex reasoning
    response = await ava.think("Compare and contrast Kant's and Hume's epistemology")
    print(response.text)

    await ava.stop()

asyncio.run(deep_analysis())
```

### With Configuration

```python
from ava import AVA, AVAConfig

async def custom_config():
    config = AVAConfig(
        simulation_mode=False,
        max_memory_turns=100,
        search_first=True,
    )

    ava = AVA(config=config)
    await ava.start()

    response = await ava.chat("What's the latest news about AI?")
    print(response.text)

    await ava.stop()

asyncio.run(custom_config())
```

### Conversation with Context

```python
async def conversation():
    ava = AVA()
    await ava.start()

    # AVA remembers context within a session
    await ava.chat("My name is Alex")
    await ava.chat("I'm learning Python")

    # This will use context from previous messages
    response = await ava.chat("What should I learn next?")
    print(response.text)  # Will consider that you're Alex learning Python

    await ava.stop()

asyncio.run(conversation())
```

### Using Tools

```python
async def with_tools():
    ava = AVA()
    await ava.start()

    # List available tools
    tools = ava.list_tools()
    for tool in tools:
        print(f"Tool: {tool.name} - {tool.description}")

    # Tools are used automatically when needed
    response = await ava.chat("What is 15% of 250?")
    print(response.text)  # Calculator tool may be used

    await ava.stop()

asyncio.run(with_tools())
```

---

## HTTP API

### Base URL

```
http://localhost:8085
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | System status |
| `/chat` | POST | Send message |
| `/think` | POST | Force deep thinking |
| `/tools` | GET | List available tools |
| `/ws` | WebSocket | Streaming chat |

### Request/Response Formats

#### POST /chat

**Request**:
```json
{
  "message": "What is Python?",
  "conversation_id": "optional-session-id"
}
```

**Response**:
```json
{
  "text": "Python is a high-level programming language...",
  "used_cortex": false,
  "cognitive_state": "flow",
  "entropy": 0.23,
  "response_time_ms": 145,
  "conversation_id": "abc123"
}
```

#### POST /think

**Request**:
```json
{
  "message": "Explain the implications of Gödel's incompleteness theorems"
}
```

**Response**:
```json
{
  "text": "Gödel's incompleteness theorems have profound implications...",
  "used_cortex": true,
  "cognitive_state": "creative",
  "entropy": 0.67,
  "response_time_ms": 8500
}
```

#### GET /status

**Response**:
```json
{
  "status": "healthy",
  "active_component": "medulla",
  "gpu_temperature_c": 45,
  "gpu_memory_used_mb": 2048,
  "total_requests": 156,
  "avg_response_time_ms": 234,
  "uptime_seconds": 3600
}
```

---

## WebSocket Streaming

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8085/ws');
```

### Message Types

**Send**:
```json
{
  "message": "Your question here",
  "force_deep": false
}
```

**Receive - Status**:
```json
{
  "type": "status",
  "status": "processing",
  "message": "Analyzing your request..."
}
```

**Receive - Chunk** (streaming):
```json
{
  "type": "chunk",
  "text": "Python is a ",
  "done": false
}
```

**Receive - Complete**:
```json
{
  "type": "response",
  "text": "Full response text...",
  "used_cortex": false,
  "done": true
}
```

**Receive - Error**:
```json
{
  "type": "error",
  "message": "Error description",
  "error_type": "timeout"
}
```

---

## JavaScript/TypeScript

### Fetch API

```typescript
async function chat(message: string): Promise<string> {
  const response = await fetch('http://localhost:8085/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error: ${response.status}`);
  }

  const data = await response.json();
  return data.text;
}

// Usage
const answer = await chat("What is machine learning?");
console.log(answer);
```

### WebSocket Client

```typescript
class AVAClient {
  private ws: WebSocket;
  private onMessage: (text: string) => void;

  constructor(onMessage: (text: string) => void) {
    this.onMessage = onMessage;
    this.ws = new WebSocket('ws://localhost:8085/ws');

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'chunk') {
        this.onMessage(data.text);
      } else if (data.type === 'response') {
        console.log('Complete:', data.text);
      } else if (data.type === 'error') {
        console.error('Error:', data.message);
      }
    };
  }

  send(message: string, forceDeep = false): void {
    this.ws.send(JSON.stringify({ message, force_deep: forceDeep }));
  }

  close(): void {
    this.ws.close();
  }
}

// Usage
const client = new AVAClient((chunk) => {
  process.stdout.write(chunk); // Stream chunks to console
});

client.send("Explain quantum computing");
```

### React Hook

```typescript
import { useState, useCallback, useEffect, useRef } from 'react';

interface UseAVAOptions {
  baseUrl?: string;
}

export function useAVA(options: UseAVAOptions = {}) {
  const { baseUrl = 'http://localhost:8085' } = options;
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const chat = useCallback(async (message: string): Promise<string> => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${baseUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      return data.text;
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Unknown error';
      setError(message);
      throw e;
    } finally {
      setIsLoading(false);
    }
  }, [baseUrl]);

  return { chat, isLoading, error };
}

// Usage in component
function ChatComponent() {
  const { chat, isLoading, error } = useAVA();
  const [response, setResponse] = useState('');

  const handleSubmit = async (message: string) => {
    const answer = await chat(message);
    setResponse(answer);
  };

  return (
    <div>
      {isLoading && <p>Thinking...</p>}
      {error && <p>Error: {error}</p>}
      {response && <p>{response}</p>}
    </div>
  );
}
```

---

## cURL Examples

### Health Check

```bash
curl http://localhost:8085/health
```

### Simple Chat

```bash
curl -X POST http://localhost:8085/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, AVA!"}'
```

### Force Deep Thinking

```bash
curl -X POST http://localhost:8085/think \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain the theory of relativity"}'
```

### Get Status

```bash
curl http://localhost:8085/status
```

### List Tools

```bash
curl http://localhost:8085/tools
```

### With Pretty Printing

```bash
curl -s http://localhost:8085/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 2+2?"}' | python -m json.tool
```

### Save Response to File

```bash
curl -X POST http://localhost:8085/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Write a poem about AI"}' \
  -o response.json
```

---

## Error Handling

### Python

```python
from ava import AVA, AVAError, ConnectionError, TimeoutError

async def safe_chat():
    ava = AVA()

    try:
        await ava.start()
        response = await ava.chat("Hello!")
        print(response.text)
    except ConnectionError:
        print("Could not connect to Ollama. Is it running?")
    except TimeoutError:
        print("Request timed out. Try a simpler question.")
    except AVAError as e:
        print(f"AVA error: {e}")
    finally:
        await ava.stop()
```

### JavaScript

```typescript
async function safeFetch(message: string): Promise<string | null> {
  try {
    const response = await fetch('http://localhost:8085/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
      signal: AbortSignal.timeout(30000), // 30s timeout
    });

    if (!response.ok) {
      if (response.status === 503) {
        console.error('AVA is not ready. Wait and retry.');
      } else {
        console.error(`HTTP Error: ${response.status}`);
      }
      return null;
    }

    const data = await response.json();
    return data.text;
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      console.error('Request timed out');
    } else {
      console.error('Network error:', error);
    }
    return null;
  }
}
```

---

*For more examples, see the `examples/` directory in the repository.*
