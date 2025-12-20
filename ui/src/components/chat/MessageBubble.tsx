"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Brain, Zap, Clock, User, Bot } from "lucide-react";
import { Message } from "@/stores/core";
import { cn, getCognitiveStateColor, getCognitiveStateBgColor } from "@/lib/utils";

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

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const isStreaming = message.isStreaming;
  
  // Use client-side only rendering for timestamp to avoid hydration mismatch
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  // Determine message style based on whether Cortex was used
  const messageStyle = isUser
    ? "message-user"
    : message.usedCortex
    ? "message-cortex"
    : "message-medulla";

  return (
    <div
      className={cn(
        "flex gap-3",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
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
            "px-4 py-3 rounded-2xl border",
            messageStyle,
            isUser ? "rounded-tr-md" : "rounded-tl-md",
            isStreaming && "animate-pulse"
          )}
          layout
        >
          {isStreaming && !message.content ? (
            <div className="waveform">
              <div className="waveform-bar" />
              <div className="waveform-bar" />
              <div className="waveform-bar" />
            </div>
          ) : (
            <p className="text-sm whitespace-pre-wrap leading-relaxed">
              {message.content}
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
      </div>
    </div>
  );
}
