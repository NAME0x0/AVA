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
 * - Fuzzy search with scoring
 * - Keyboard navigation
 * - Action categories
 * - Full accessibility (ARIA)
 */
export function CommandPalette({
  isOpen,
  onClose,
  additionalCommands = [],
}: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [searchMode, setSearchMode] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const { forceCortex, forceSleep, clearMessages, toggleSidebar, setInputValue } = useCoreStore();

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
          setSearchMode(true);
          setInputValue("[Search] ");
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
          setShowHelp(true);
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

  // Fuzzy search with scoring
  const filteredCommands = useMemo(() => {
    if (!query) return allCommands;

    const lowerQuery = query.toLowerCase();

    // Score-based fuzzy matching
    const scored = allCommands
      .map((cmd) => {
        const labelLower = cmd.label.toLowerCase();
        const descLower = cmd.description?.toLowerCase() || "";

        let score = 0;

        // Exact match in label
        if (labelLower === lowerQuery) score += 100;
        // Starts with query
        else if (labelLower.startsWith(lowerQuery)) score += 50;
        // Contains query
        else if (labelLower.includes(lowerQuery)) score += 25;
        // Description match
        else if (descLower.includes(lowerQuery)) score += 10;
        // Initials match (e.g., "dt" matches "Deep Think")
        else {
          const initials = cmd.label
            .split(" ")
            .map((w) => w[0]?.toLowerCase())
            .join("");
          if (initials.includes(lowerQuery)) score += 30;
        }

        return { cmd, score };
      })
      .filter(({ score }) => score > 0)
      .sort((a, b) => b.score - a.score)
      .map(({ cmd }) => cmd);

    return scored;
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
      setShowHelp(false);
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
          className="fixed inset-0 z-50 flex items-start justify-center pt-[10vh] sm:pt-[15vh] px-4 bg-black/50 backdrop-blur-sm"
          onClick={handleBackdropClick}
          role="dialog"
          aria-modal="true"
          aria-label="Command palette"
        >
          <motion.div
            initial={{ opacity: 0, y: -20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="w-full max-w-lg bg-neural-surface rounded-xl border border-neural-hover shadow-2xl overflow-hidden"
          >
            {/* Search input */}
            <div className="flex items-center gap-3 px-4 py-3 border-b border-neural-hover">
              <span className="text-text-muted" aria-hidden="true">⌘</span>
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
                aria-label="Search commands"
                aria-controls="command-list"
                aria-activedescendant={filteredCommands[selectedIndex]?.id}
                className="flex-1 bg-transparent text-text-primary outline-none placeholder:text-text-muted"
              />
              <kbd className="hidden sm:block px-2 py-1 text-xs text-text-muted bg-neural-elevated rounded">
                ESC
              </kbd>
            </div>

            {/* Commands list */}
            <div
              ref={listRef}
              id="command-list"
              role="listbox"
              aria-label="Available commands"
              className="max-h-80 overflow-y-auto py-2"
            >
              {Object.entries(groupedCommands).map(([category, commands]) => (
                <div key={category} role="group" aria-label={category}>
                  <div className="px-4 py-1 text-xs font-semibold text-text-muted uppercase tracking-wide">
                    {category}
                  </div>
                  {commands.map((cmd) => {
                    const globalIndex = filteredCommands.indexOf(cmd);
                    const isSelected = globalIndex === selectedIndex;

                    return (
                      <button
                        key={cmd.id}
                        id={cmd.id}
                        role="option"
                        aria-selected={isSelected}
                        className={`
                          w-full flex items-center gap-3 px-4 py-2 text-left
                          transition-colors duration-75
                          ${isSelected ? "bg-accent-primary/20 text-accent-primary" : "hover:bg-neural-elevated text-text-primary"}
                          focus:outline-none focus:bg-accent-primary/20
                        `}
                        onClick={cmd.action}
                        onMouseEnter={() => setSelectedIndex(globalIndex)}
                      >
                        <span className="text-lg" aria-hidden="true">{cmd.icon}</span>
                        <div className="flex-1 min-w-0">
                          <div className="font-medium truncate">{cmd.label}</div>
                          {cmd.description && (
                            <div className="text-sm text-text-muted truncate">
                              {cmd.description}
                            </div>
                          )}
                        </div>
                        {cmd.shortcut && (
                          <kbd className="hidden sm:block px-2 py-1 text-xs text-text-muted bg-neural-elevated rounded shrink-0">
                            {cmd.shortcut}
                          </kbd>
                        )}
                      </button>
                    );
                  })}
                </div>
              ))}

              {filteredCommands.length === 0 && (
                <div className="px-4 py-8 text-center text-text-muted" role="status">
                  No commands found for "{query}"
                </div>
              )}
            </div>

            {/* Help panel - inline keyboard shortcuts */}
            <AnimatePresence>
              {showHelp && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="border-t border-neural-hover overflow-hidden"
                >
                  <div className="px-4 py-3 bg-neural-elevated/50">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-sm font-semibold text-text-primary">Keyboard Shortcuts</h3>
                      <button
                        onClick={() => setShowHelp(false)}
                        className="text-text-muted hover:text-text-primary transition-colors text-sm"
                        aria-label="Close help"
                      >
                        ✕
                      </button>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      {[
                        { key: "Ctrl+K", desc: "Command Palette" },
                        { key: "Ctrl+D", desc: "Deep Think (Cortex)" },
                        { key: "Ctrl+S", desc: "Force Search" },
                        { key: "Ctrl+L", desc: "Clear Chat" },
                        { key: "Ctrl+B", desc: "Toggle Sidebar" },
                        { key: "Enter", desc: "Send Message" },
                        { key: "Shift+Enter", desc: "New Line" },
                        { key: "ESC", desc: "Close" },
                      ].map(({ key, desc }) => (
                        <div key={key} className="flex items-center gap-2">
                          <kbd className="px-1.5 py-0.5 text-xs bg-neural-surface rounded text-text-muted min-w-[70px]">
                            {key}
                          </kbd>
                          <span className="text-text-secondary">{desc}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
