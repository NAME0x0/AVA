/**
 * WebSocket Hook for Streaming Responses
 * 
 * Provides real-time streaming of responses from the AVA backend
 * 
 * TODO: This hook is currently exported but not integrated into the main chat flow.
 * Future integration plan:
 * - Replace HTTP polling in useCoreStore with WebSocket streaming
 * - Add automatic reconnection with exponential backoff
 * - Support streaming for both Medulla (fast) and Cortex (slow) responses
 * - Track streaming progress for UI indicators
 * 
 * @see useCoreStore for current HTTP-based implementation
 */

import { useRef, useCallback, useState } from "react";
import { useCoreStore } from "@/stores/core";

interface StreamingMessage {
  type: "start" | "chunk" | "end" | "error";
  content?: string;
  done?: boolean;
  error?: string;
}

export function useStreamingChat() {
  const { backendUrl, addMessage, updateMessage, setIsGenerating } = useCoreStore();
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const wsUrl = backendUrl.replace("http", "ws") + "/chat/stream";
    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => {
      setIsConnected(true);
      console.log("WebSocket connected");
    };

    wsRef.current.onclose = () => {
      setIsConnected(false);
      console.log("WebSocket disconnected");
    };

    wsRef.current.onerror = (error) => {
      console.error("WebSocket error:", error);
      setIsConnected(false);
    };
  }, [backendUrl]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const sendStreamingMessage = useCallback(
    async (content: string): Promise<void> => {
      return new Promise((resolve, reject) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
          // Fall back to HTTP
          reject(new Error("WebSocket not connected"));
          return;
        }

        const messageId = Math.random().toString(36).substring(2, 15);
        let responseContent = "";

        // Add placeholder message
        addMessage({
          role: "assistant",
          content: "",
          isStreaming: true,
        });

        setIsGenerating(true);

        const handleMessage = (event: MessageEvent) => {
          try {
            const data: StreamingMessage = JSON.parse(event.data);

            switch (data.type) {
              case "start":
                // Stream starting
                break;

              case "chunk":
                responseContent += data.content || "";
                // Update the last message with accumulated content
                const messages = useCoreStore.getState().messages;
                const lastMessage = messages[messages.length - 1];
                if (lastMessage) {
                  updateMessage(lastMessage.id, {
                    content: responseContent,
                    isStreaming: !data.done,
                  });
                }
                break;

              case "end":
                setIsGenerating(false);
                wsRef.current?.removeEventListener("message", handleMessage);
                resolve();
                break;

              case "error":
                setIsGenerating(false);
                wsRef.current?.removeEventListener("message", handleMessage);
                reject(new Error(data.error));
                break;
            }
          } catch (error) {
            console.error("Failed to parse streaming message:", error);
          }
        };

        wsRef.current.addEventListener("message", handleMessage);
        wsRef.current.send(JSON.stringify({ message: content }));
      });
    },
    [addMessage, updateMessage, setIsGenerating]
  );

  return {
    isConnected,
    connect,
    disconnect,
    sendStreamingMessage,
  };
}
