"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Zap, Clock, User, Bot, Copy, Check, Wrench, RefreshCw, AlertTriangle } from "lucide-react";
import { Message, useCoreStore } from "@/stores/core";
import { cn, getCognitiveStateColor, getCognitiveStateBgColor } from "@/lib/utils";

// Tool icon mapping
const TOOL_ICONS: Record<string, string> = {
  calculator: "🧮",
  web_search: "🔍",
  datetime: "📅",
  read_file: "📄",
  web_browse: "🌐",
  verify_fact: "✓",
};

interface MessageBubbleProps {
  message: Message;
}

// Format time consistently to avoid hydration mismatches
function formatTime(date: Date): string {
  const hours = date.getHours();
  const minutes = date.getMinutes();
  const ampm = hours >= 12 ? 'PM' : 'AM';
  const hour12 = hours % 12 || 12;
  const minuteStr = minutes.toString().padStart(2, '0');
  return `${hour12}:${minuteStr} ${ampm}`;
}

// Tool Badge Component
function ToolBadge({ name }: { name: string }) {
  const icon = TOOL_ICONS[name] || "🔧";
  return (
    <motion.span
      className="px-1.5 py-0.5 text-[10px] rounded bg-accent-dim/20 text-accent-primary inline-flex items-center gap-1"
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: "spring", stiffness: 500, damping: 15 }}
    >
      <span>{icon}</span>
      <span>{name.replace(/_/g, " ")}</span>
    </motion.span>
  );
}

// Streaming Cursor Component
function StreamingCursor() {
  return (
    <motion.span
      className="inline-block w-2 h-4 bg-accent-primary ml-0.5 align-middle"
      animate={{ opacity: [1, 0] }}
      transition={{ duration: 0.5, repeat: Infinity, repeatType: "reverse" }}
    />
  );
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const isStreaming = message.isStreaming;
  const isError = message.error;
  const retryMessage = useCoreStore((state) => state.retryMessage);
  const isGenerating = useCoreStore((state) => state.isGenerating);

  // Use client-side only rendering for timestamp to avoid hydration mismatch
  const [mounted, setMounted] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const copyToClipboard = useCallback(async () => {
    if (!message.content) return;
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  }, [message.content]);

  // Determine message style based on whether Cortex was used or error
  const messageStyle = isUser
    ? "message-user"
    : isError
    ? "message-error"
    : message.usedCortex
    ? "message-cortex"
    : "message-medulla";

  return (
    <motion.div
      className={cn(
        "flex gap-3 relative group",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      whileHover={{ scale: 1.005 }}
      transition={{ duration: 0.2 }}
    >
      {/* Avatar */}
      <div
        className={cn(
          "w-8 h-8 rounded-lg flex items-center justify-center shrink-0",
          isUser
            ? "bg-accent-primary/20"
            : message.usedCortex
            ? "bg-cortex-active/20"
            : "bg-medulla-active/20"
        )}
      >
        {isUser ? (
          <User className="w-4 h-4 text-accent-primary" />
        ) : (
          <Bot
            className={cn(
              "w-4 h-4",
              message.usedCortex ? "text-cortex-active" : "text-medulla-active"
            )}
          />
        )}
      </div>

      {/* Message Content */}
      <div
        className={cn(
          "flex flex-col max-w-[70%]",
          isUser ? "items-end" : "items-start"
        )}
      >
        {/* Header - Component indicator */}
        {!isUser && (
          <div className="flex items-center gap-2 mb-1 text-xs">
            {message.usedCortex ? (
              <span className="text-cortex-active flex items-center gap-1">
                <Brain className="w-3 h-3" />
                Cortex
              </span>
            ) : (
              <span className="text-medulla-active flex items-center gap-1">
                <Zap className="w-3 h-3" />
                Medulla
              </span>
            )}
            {message.policySelected && (
              <span className="text-text-muted">
                • {message.policySelected.replace(/_/g, " ").toLowerCase()}
              </span>
            )}
          </div>
        )}

        {/* Bubble */}
        <motion.div
          className={cn(
            "px-4 py-3 rounded-2xl border relative",
            messageStyle,
            isUser ? "rounded-tr-md" : "rounded-tl-md"
          )}
          layout
        >
          {/* Copy Button */}
          <AnimatePresence>
            {isHovered && message.content && !isStreaming && (
              <motion.button
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                onClick={copyToClipboard}
                className={cn(
                  "absolute top-2 right-2 p-1.5 rounded-lg",
                  "bg-neural-elevated/80 backdrop-blur-sm",
                  "hover:bg-neural-hover transition-colors",
                  "border border-neural-hover/50"
                )}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                title={copied ? "Copied!" : "Copy message"}
              >
                {copied ? (
                  <Check className="w-3.5 h-3.5 text-state-flow" />
                ) : (
                  <Copy className="w-3.5 h-3.5 text-text-muted" />
                )}
              </motion.button>
            )}
          </AnimatePresence>

          {isStreaming && !message.content ? (
            <div className="waveform">
              <div className="waveform-bar" />
              <div className="waveform-bar" />
              <div className="waveform-bar" />
            </div>
          ) : (
            <p className="text-sm whitespace-pre-wrap leading-relaxed">
              {message.content}
              {isStreaming && <StreamingCursor />}
            </p>
          )}
        </motion.div>

        {/* Footer - Metadata */}
        <div
          className={cn(
            "flex items-center gap-3 mt-1 text-[10px] text-text-muted",
            isUser ? "flex-row-reverse" : "flex-row"
          )}
        >
          {/* Timestamp - only render on client to avoid hydration mismatch */}
          <span suppressHydrationWarning>
            {mounted ? formatTime(message.timestamp) : "--:-- --"}
          </span>

          {/* Response time */}
          {!isUser && message.responseTimeMs && (
            <span className="flex items-center gap-0.5">
              <Clock className="w-2.5 h-2.5" />
              {message.responseTimeMs.toFixed(0)}ms
            </span>
          )}

          {/* Cognitive state badge */}
          {!isUser && message.cognitiveState && (
            <span
              className={cn(
                "px-1.5 py-0.5 rounded text-[9px] uppercase font-medium",
                getCognitiveStateBgColor(message.cognitiveState.label),
                getCognitiveStateColor(message.cognitiveState.label)
              )}
            >
              {message.cognitiveState.label}
            </span>
          )}

          {/* Surprise indicator */}
          {!isUser && message.surprise !== undefined && message.surprise > 1.5 && (
            <span className="text-memory-surprise">
              σ:{message.surprise.toFixed(1)}
            </span>
          )}
        </div>

        {/* Tool badges */}
        {!isUser && message.toolsUsed && message.toolsUsed.length > 0 && (
          <div className="flex gap-1 mt-1.5 flex-wrap">
            {message.toolsUsed.map((tool) => (
              <ToolBadge key={tool} name={tool} />
            ))}
          </div>
        )}

        {/* Retry button for failed messages */}
        {!isUser && isError && !isStreaming && (
          <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-2 mt-2"
          >
            <AlertTriangle className="w-3.5 h-3.5 text-status-error" />
            <span className="text-xs text-status-error">
              {message.errorType === "network" && "Network error"}
              {message.errorType === "timeout" && "Request timed out"}
              {message.errorType === "server" && "Server error"}
              {message.errorType === "unknown" && "Error"}
            </span>
            {message.originalContent && (message.retryCount || 0) < 3 && (
              <motion.button
                onClick={() => retryMessage(message.id)}
                disabled={isGenerating}
                className={cn(
                  "flex items-center gap-1 px-2 py-1 rounded text-xs",
                  "bg-accent-primary/20 text-accent-primary",
                  "hover:bg-accent-primary/30 transition-colors",
                  "disabled:opacity-50 disabled:cursor-not-allowed"
                )}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <RefreshCw className={cn("w-3 h-3", isGenerating && "animate-spin")} />
                Retry
                {(message.retryCount || 0) > 0 && (
                  <span className="opacity-60">({message.retryCount}/3)</span>
                )}
              </motion.button>
            )}
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}
