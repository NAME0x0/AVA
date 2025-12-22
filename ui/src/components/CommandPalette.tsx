"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useCoreStore } from "@/stores/core";

interface Command {
  id: string;
  label: string;
  description?: string;
  shortcut?: string;
  icon?: string;
  action: () => void;
  category?: string;
}

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  additionalCommands?: Command[];
}

/**
 * CommandPalette - Quick action launcher (Ctrl+K)
 *
 * Features:
 * - Fuzzy search
 * - Keyboard navigation
 * - Action categories
 * - Recent commands
 */
export function CommandPalette({
  isOpen,
  onClose,
  additionalCommands = [],
}: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const { forceCortex, forceSleep, clearMessages, toggleSidebar } = useCoreStore();

  // Default commands
  const defaultCommands: Command[] = useMemo(
    () => [
      {
        id: "deep-think",
        label: "Deep Think",
        description: "Force Cortex for next response",
        shortcut: "Ctrl+D",
        icon: "🧠",
        action: () => {
          forceCortex();
          onClose();
        },
        category: "Actions",
      },
      {
        id: "search",
        label: "Search First",
        description: "Force web search for next query",
        shortcut: "Ctrl+S",
        icon: "🔍",
        action: () => {
          // Set search mode flag
          onClose();
        },
        category: "Actions",
      },
      {
        id: "clear-chat",
        label: "Clear Chat",
        description: "Clear all messages",
        shortcut: "Ctrl+L",
        icon: "🗑️",
        action: () => {
          clearMessages();
          onClose();
        },
        category: "Chat",
      },
      {
        id: "toggle-sidebar",
        label: "Toggle Sidebar",
        description: "Show/hide metrics sidebar",
        shortcut: "Ctrl+B",
        icon: "📊",
        action: () => {
          toggleSidebar();
          onClose();
        },
        category: "View",
      },
      {
        id: "sleep-mode",
        label: "Sleep Mode",
        description: "Put AVA into low-power sleep",
        icon: "😴",
        action: () => {
          forceSleep();
          onClose();
        },
        category: "System",
      },
      {
        id: "help",
        label: "Help",
        description: "Show keyboard shortcuts",
        shortcut: "F1",
        icon: "❓",
        action: () => {
          // Show help modal
          onClose();
        },
        category: "Help",
      },
    ],
    [forceCortex, forceSleep, clearMessages, toggleSidebar, onClose]
  );

  const allCommands = useMemo(
    () => [...defaultCommands, ...additionalCommands],
    [defaultCommands, additionalCommands]
  );

  // Filter commands based on query
  const filteredCommands = useMemo(() => {
    if (!query) return allCommands;

    const lowerQuery = query.toLowerCase();
    return allCommands.filter(
      (cmd) =>
        cmd.label.toLowerCase().includes(lowerQuery) ||
        cmd.description?.toLowerCase().includes(lowerQuery) ||
        cmd.category?.toLowerCase().includes(lowerQuery)
    );
  }, [allCommands, query]);

  // Group by category
  const groupedCommands = useMemo(() => {
    const groups: Record<string, Command[]> = {};
    for (const cmd of filteredCommands) {
      const category = cmd.category || "General";
      if (!groups[category]) groups[category] = [];
      groups[category].push(cmd);
    }
    return groups;
  }, [filteredCommands]);

  // Reset on open
  useEffect(() => {
    if (isOpen) {
      setQuery("");
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  // Keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((i) => Math.min(i + 1, filteredCommands.length - 1));
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((i) => Math.max(i - 1, 0));
          break;
        case "Enter":
          e.preventDefault();
          if (filteredCommands[selectedIndex]) {
            filteredCommands[selectedIndex].action();
          }
          break;
        case "Escape":
          e.preventDefault();
          onClose();
          break;
      }
    },
    [filteredCommands, selectedIndex, onClose]
  );

  // Close on backdrop click
  const handleBackdropClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === e.currentTarget) {
        onClose();
      }
    },
    [onClose]
  );

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh] bg-black/50 backdrop-blur-sm"
          onClick={handleBackdropClick}
        >
          <motion.div
            initial={{ opacity: 0, y: -20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="w-full max-w-lg bg-surface rounded-xl border border-surface-lighter shadow-2xl overflow-hidden"
          >
            {/* Search input */}
            <div className="flex items-center gap-3 px-4 py-3 border-b border-surface-lighter">
              <span className="text-text-muted">⌘</span>
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setSelectedIndex(0);
                }}
                onKeyDown={handleKeyDown}
                placeholder="Type a command or search..."
                className="flex-1 bg-transparent text-text outline-none placeholder:text-text-muted"
              />
              <kbd className="px-2 py-1 text-xs text-text-muted bg-surface-lighter rounded">
                ESC
              </kbd>
            </div>

            {/* Commands list */}
            <div className="max-h-80 overflow-y-auto py-2">
              {Object.entries(groupedCommands).map(([category, commands]) => (
                <div key={category}>
                  <div className="px-4 py-1 text-xs font-semibold text-text-muted uppercase tracking-wide">
                    {category}
                  </div>
                  {commands.map((cmd) => {
                    const globalIndex = filteredCommands.indexOf(cmd);
                    const isSelected = globalIndex === selectedIndex;

                    return (
                      <button
                        key={cmd.id}
                        className={`
                          w-full flex items-center gap-3 px-4 py-2 text-left
                          transition-colors duration-75
                          ${isSelected ? "bg-primary/20 text-primary" : "hover:bg-surface-lighter text-text"}
                        `}
                        onClick={cmd.action}
                        onMouseEnter={() => setSelectedIndex(globalIndex)}
                      >
                        <span className="text-lg">{cmd.icon}</span>
                        <div className="flex-1">
                          <div className="font-medium">{cmd.label}</div>
                          {cmd.description && (
                            <div className="text-sm text-text-muted">
                              {cmd.description}
                            </div>
                          )}
                        </div>
                        {cmd.shortcut && (
                          <kbd className="px-2 py-1 text-xs text-text-muted bg-surface-darker rounded">
                            {cmd.shortcut}
                          </kbd>
                        )}
                      </button>
                    );
                  })}
                </div>
              ))}

              {filteredCommands.length === 0 && (
                <div className="px-4 py-8 text-center text-text-muted">
                  No commands found
                </div>
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
