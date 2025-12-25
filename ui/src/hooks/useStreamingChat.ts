/**
 * Streaming Chat Hook
 * 
 * Provides WebSocket-based streaming chat functionality with the AVA backend.
 * Handles connection management, message streaming, and reconnection logic.
 */

import { useState, useCallback, useRef, useEffect } from "react";
import { useCoreStore } from "@/stores/core";

export interface StreamingMessage {
    id: string;
    role: "user" | "assistant";
    content: string;
    isStreaming: boolean;
    timestamp: Date;
    metadata?: {
        usedCortex?: boolean;
        cognitiveState?: string;
        confidence?: number;
        toolsUsed?: string[];
        responseTimeMs?: number;
    };
}

export interface UseStreamingChatOptions {
    /** WebSocket endpoint path (default: /ws) */
    wsPath?: string;
    /** Auto-reconnect on disconnect */
    autoReconnect?: boolean;
    /** Reconnect delay in ms */
    reconnectDelay?: number;
    /** Max reconnect attempts */
    maxReconnectAttempts?: number;
}

export interface UseStreamingChatReturn {
    /** Send a message and receive streaming response */
    sendMessage: (message: string, forceThink?: boolean) => Promise<void>;
    /** Current messages in the conversation */
    messages: StreamingMessage[];
    /** Whether currently streaming a response */
    isStreaming: boolean;
    /** Whether WebSocket is connected */
    isConnected: boolean;
    /** Clear all messages */
    clearMessages: () => void;
    /** Manually connect to WebSocket */
    connect: () => void;
    /** Manually disconnect from WebSocket */
    disconnect: () => void;
    /** Connection error if any */
    error: string | null;
}

export function useStreamingChat(
    options: UseStreamingChatOptions = {}
): UseStreamingChatReturn {
    const {
        wsPath = "/ws",
        autoReconnect = true,
        reconnectDelay = 3000,
        maxReconnectAttempts = 5,
    } = options;

    const { backendUrl } = useCoreStore();

    const [messages, setMessages] = useState<StreamingMessage[]>([]);
    const [isStreaming, setIsStreaming] = useState(false);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectAttemptsRef = useRef(0);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const currentMessageIdRef = useRef<string | null>(null);

    // Generate unique message ID
    const generateId = useCallback(() => {
        return `msg_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
    }, []);

    // Get WebSocket URL from backend URL
    const getWsUrl = useCallback(() => {
        const url = new URL(backendUrl);
        const protocol = url.protocol === "https:" ? "wss:" : "ws:";
        return `${protocol}//${url.host}${wsPath}`;
    }, [backendUrl, wsPath]);

    // Handle incoming WebSocket messages
    const handleMessage = useCallback((event: MessageEvent) => {
        try {
            const data = JSON.parse(event.data);

            if (data.type === "token" || data.type === "chunk") {
                // Streaming token
                setMessages((prev) => {
                    const lastMessage = prev[prev.length - 1];
                    if (lastMessage?.isStreaming && lastMessage.role === "assistant") {
                        return [
                            ...prev.slice(0, -1),
                            {
                                ...lastMessage,
                                content: lastMessage.content + (data.token || data.content || ""),
                            },
                        ];
                    }
                    return prev;
                });
            } else if (data.type === "start") {
                // Start of response
                const messageId = generateId();
                currentMessageIdRef.current = messageId;
                setIsStreaming(true);
                setMessages((prev) => [
                    ...prev,
                    {
                        id: messageId,
                        role: "assistant",
                        content: "",
                        isStreaming: true,
                        timestamp: new Date(),
                    },
                ]);
            } else if (data.type === "end" || data.type === "complete") {
                // End of response
                setIsStreaming(false);
                setMessages((prev) => {
                    const lastMessage = prev[prev.length - 1];
                    if (lastMessage?.isStreaming) {
                        return [
                            ...prev.slice(0, -1),
                            {
                                ...lastMessage,
                                isStreaming: false,
                                metadata: {
                                    usedCortex: data.used_cortex,
                                    cognitiveState: data.cognitive_state,
                                    confidence: data.confidence,
                                    toolsUsed: data.tools_used,
                                    responseTimeMs: data.response_time_ms,
                                },
                            },
                        ];
                    }
                    return prev;
                });
                currentMessageIdRef.current = null;
            } else if (data.type === "error") {
                // Error from server
                setError(data.message || "Unknown error");
                setIsStreaming(false);
            } else if (data.text) {
                // Non-streaming response (fallback)
                const messageId = generateId();
                setMessages((prev) => [
                    ...prev,
                    {
                        id: messageId,
                        role: "assistant",
                        content: data.text,
                        isStreaming: false,
                        timestamp: new Date(),
                        metadata: {
                            usedCortex: data.used_cortex,
                            cognitiveState: data.cognitive_state,
                            confidence: data.confidence,
                            toolsUsed: data.tools_used,
                        },
                    },
                ]);
            }
        } catch (e) {
            console.error("Failed to parse WebSocket message:", e);
        }
    }, [generateId]);

    // Connect to WebSocket
    const connect = useCallback(() => {
        // Check if running in browser
        if (typeof window === "undefined") return;

        // Close existing connection
        if (wsRef.current) {
            wsRef.current.close();
        }

        try {
            const wsUrl = getWsUrl();
            const ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                setIsConnected(true);
                setError(null);
                reconnectAttemptsRef.current = 0;
            };

            ws.onmessage = handleMessage;

            ws.onclose = (event) => {
                setIsConnected(false);
                wsRef.current = null;

                // Auto-reconnect if enabled and not intentionally closed
                if (autoReconnect && event.code !== 1000) {
                    if (reconnectAttemptsRef.current < maxReconnectAttempts) {
                        reconnectTimeoutRef.current = setTimeout(() => {
                            reconnectAttemptsRef.current++;
                            connect();
                        }, reconnectDelay);
                    } else {
                        setError("Failed to reconnect after maximum attempts");
                    }
                }
            };

            ws.onerror = () => {
                setError("WebSocket connection error");
            };

            wsRef.current = ws;
        } catch (e) {
            setError(`Failed to create WebSocket: ${e}`);
        }
    }, [getWsUrl, handleMessage, autoReconnect, reconnectDelay, maxReconnectAttempts]);

    // Disconnect from WebSocket
    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
        }
        if (wsRef.current) {
            wsRef.current.close(1000, "User disconnected");
            wsRef.current = null;
        }
        setIsConnected(false);
    }, []);

    // Send a message
    const sendMessage = useCallback(
        async (message: string, forceThink: boolean = false) => {
            if (!message.trim()) return;

            // Add user message
            const userMessageId = generateId();
            setMessages((prev) => [
                ...prev,
                {
                    id: userMessageId,
                    role: "user",
                    content: message,
                    isStreaming: false,
                    timestamp: new Date(),
                },
            ]);

            // If WebSocket is connected, send via WebSocket
            if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(
                    JSON.stringify({
                        message,
                        force_think: forceThink,
                    })
                );
            } else {
                // Fallback to HTTP API
                try {
                    const endpoint = forceThink ? "/think" : "/chat";
                    const response = await fetch(`${backendUrl}${endpoint}`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ message }),
                    });

                    if (response.ok) {
                        const data = await response.json();
                        const assistantMessageId = generateId();
                        setMessages((prev) => [
                            ...prev,
                            {
                                id: assistantMessageId,
                                role: "assistant",
                                content: data.text || data.response || "",
                                isStreaming: false,
                                timestamp: new Date(),
                                metadata: {
                                    usedCortex: data.used_cortex,
                                    cognitiveState: data.cognitive_state,
                                    confidence: data.confidence,
                                    toolsUsed: data.tools_used,
                                    responseTimeMs: data.response_time_ms,
                                },
                            },
                        ]);
                    } else {
                        setError(`HTTP error: ${response.status}`);
                    }
                } catch (e) {
                    setError(`Failed to send message: ${e}`);
                }
            }
        },
        [backendUrl, generateId]
    );

    // Clear all messages
    const clearMessages = useCallback(() => {
        setMessages([]);
        setError(null);
    }, []);

    // Auto-connect on mount
    useEffect(() => {
        connect();
        return () => disconnect();
    }, [connect, disconnect]);

    return {
        sendMessage,
        messages,
        isStreaming,
        isConnected,
        clearMessages,
        connect,
        disconnect,
        error,
    };
}
