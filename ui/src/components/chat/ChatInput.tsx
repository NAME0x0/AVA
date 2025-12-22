"use client";

import { useState, useRef, useEffect, KeyboardEvent } from "react";
import { motion } from "framer-motion";
import { Send, Loader2, Brain, Zap } from "lucide-react";
import { useCoreStore } from "@/stores/core";
import { cn } from "@/lib/utils";

export function ChatInput() {
  const { inputValue, setInputValue, sendMessage, isGenerating, connected, systemState } =
    useCoreStore();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [isFocused, setIsFocused] = useState(false);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        200
      )}px`;
    }
  }, [inputValue]);

  const handleSubmit = () => {
    if (!inputValue.trim() || isGenerating) return;
    sendMessage(inputValue.trim());
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="px-4 py-4 bg-neural-surface/50 backdrop-blur-sm border-t border-neural-hover">
      {/* Active component indicator above input */}
      <div className="flex items-center gap-2 mb-2 text-xs">
        <span className="text-text-muted">Response via:</span>
        <motion.div
          className={cn(
            "flex items-center gap-1 px-2 py-0.5 rounded-full transition-colors",
            systemState.activeComponent === "cortex"
              ? "bg-cortex-active/20 text-cortex-active"
              : "bg-medulla-active/20 text-medulla-active"
          )}
          animate={{
            scale: isGenerating ? [1, 1.05, 1] : 1,
          }}
          transition={{
            duration: 0.5,
            repeat: isGenerating ? Infinity : 0,
          }}
        >
          {systemState.activeComponent === "cortex" ? (
            <>
              <Brain className="w-3 h-3" />
              <span>Cortex (Deep)</span>
            </>
          ) : (
            <>
              <Zap className="w-3 h-3" />
              <span>Medulla (Fast)</span>
            </>
          )}
        </motion.div>
      </div>

      {/* Input container */}
      <motion.div
        className={cn(
          "relative flex items-end gap-2 rounded-2xl border transition-all duration-200",
          isFocused
            ? "bg-neural-elevated border-accent-primary/50 shadow-neural"
            : "bg-neural-elevated border-neural-hover",
          !connected && "opacity-60"
        )}
      >
        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder={
            connected ? "Message AVA..." : "Waiting for backend connection..."
          }
          disabled={!connected || isGenerating}
          rows={1}
          aria-label="Message input"
          aria-describedby="input-hint"
          aria-disabled={!connected || isGenerating}
          className={cn(
            "flex-1 bg-transparent px-4 py-3 text-sm text-text-primary",
            "placeholder:text-text-muted resize-none outline-none",
            "max-h-[200px] min-h-[44px]",
            "focus:ring-0 focus:outline-none"
          )}
        />

        {/* Send button */}
        <motion.button
          onClick={handleSubmit}
          disabled={!inputValue.trim() || isGenerating || !connected}
          aria-label={isGenerating ? "Generating response" : "Send message"}
          aria-busy={isGenerating}
          className={cn(
            "m-2 p-2.5 rounded-xl transition-all duration-200",
            inputValue.trim() && connected
              ? "bg-accent-primary text-neural-void hover:bg-accent-dim"
              : "bg-neural-hover text-text-muted cursor-not-allowed",
            "focus:outline-none focus:ring-2 focus:ring-accent-primary focus:ring-offset-2 focus:ring-offset-neural-elevated"
          )}
          whileHover={inputValue.trim() && connected ? { scale: 1.05 } : {}}
          whileTap={inputValue.trim() && connected ? { scale: 0.95 } : {}}
        >
          {isGenerating ? (
            <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" />
          ) : (
            <Send className="w-4 h-4" aria-hidden="true" />
          )}
        </motion.button>
      </motion.div>

      {/* Hint text */}
      <div id="input-hint" className="flex items-center justify-between mt-2 text-[10px] text-text-muted">
        <span>Press Enter to send, Shift+Enter for new line</span>
        {isGenerating && (
          <span className="text-accent-primary animate-pulse" role="status" aria-live="polite">
            Generating response...
          </span>
        )}
      </div>
    </div>
  );
}
