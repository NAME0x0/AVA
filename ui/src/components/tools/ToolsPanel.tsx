"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Wrench, Calculator, Search, Calendar, FileText, Globe, CheckCircle } from "lucide-react";
import { useCoreStore } from "@/stores/core";
import { cn } from "@/lib/utils";

// Tool definitions with metadata
const TOOLS_CONFIG = [
  {
    id: "calculator",
    name: "Calculator",
    description: "Perform mathematical calculations",
    icon: Calculator,
    color: "text-blue-400",
    bgColor: "bg-blue-400/10",
  },
  {
    id: "web_search",
    name: "Web Search",
    description: "Search the internet for information",
    icon: Search,
    color: "text-green-400",
    bgColor: "bg-green-400/10",
  },
  {
    id: "datetime",
    name: "Date & Time",
    description: "Get current date and time information",
    icon: Calendar,
    color: "text-purple-400",
    bgColor: "bg-purple-400/10",
  },
  {
    id: "read_file",
    name: "Read File",
    description: "Read content from local files",
    icon: FileText,
    color: "text-yellow-400",
    bgColor: "bg-yellow-400/10",
  },
  {
    id: "web_browse",
    name: "Web Browse",
    description: "Browse and extract web page content",
    icon: Globe,
    color: "text-cyan-400",
    bgColor: "bg-cyan-400/10",
  },
  {
    id: "verify_fact",
    name: "Verify Fact",
    description: "Cross-reference and verify information",
    icon: CheckCircle,
    color: "text-emerald-400",
    bgColor: "bg-emerald-400/10",
  },
];

interface Tool {
  name: string;
  description: string;
  enabled: boolean;
}

export function ToolsPanel() {
  const { toolsPanelOpen, setToolsPanelOpen, backendUrl } = useCoreStore();
  const [tools, setTools] = useState<Tool[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch tools from backend
  useEffect(() => {
    if (toolsPanelOpen) {
      fetchTools();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toolsPanelOpen, backendUrl]);

  const fetchTools = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${backendUrl}/tools`);
      if (response.ok) {
        const data = await response.json();
        setTools(data.tools || []);
      } else {
        // Use fallback tools if backend unavailable
        setTools(
          TOOLS_CONFIG.map((t) => ({
            name: t.id,
            description: t.description,
            enabled: true,
          }))
        );
      }
    } catch (e) {
      // Use fallback tools
      setTools(
        TOOLS_CONFIG.map((t) => ({
          name: t.id,
          description: t.description,
          enabled: true,
        }))
      );
    } finally {
      setLoading(false);
    }
  };

  const getToolConfig = (toolName: string) => {
    return TOOLS_CONFIG.find((t) => t.id === toolName) || {
      id: toolName,
      name: toolName.replace(/_/g, " "),
      description: "Custom tool",
      icon: Wrench,
      color: "text-gray-400",
      bgColor: "bg-gray-400/10",
    };
  };

  return (
    <AnimatePresence>
      {toolsPanelOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/30 backdrop-blur-sm z-40"
            onClick={() => setToolsPanelOpen(false)}
          />

          {/* Panel */}
          <motion.aside
            initial={{ width: 0, opacity: 0, x: 100 }}
            animate={{ width: 360, opacity: 1, x: 0 }}
            exit={{ width: 0, opacity: 0, x: 100 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="fixed right-0 top-0 h-full z-50 bg-neural-surface/95 backdrop-blur-md border-l border-neural-hover overflow-hidden"
          >
            <div className="h-full flex flex-col">
              {/* Header */}
              <div className="flex items-center justify-between p-4 border-b border-neural-hover">
                <div className="flex items-center gap-2">
                  <Wrench className="w-5 h-5 text-accent-primary" />
                  <h2 className="text-lg font-semibold text-text-primary">Available Tools</h2>
                </div>
                <motion.button
                  onClick={() => setToolsPanelOpen(false)}
                  className="p-2 rounded-lg hover:bg-neural-hover transition-colors"
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <X className="w-5 h-5 text-text-muted" />
                </motion.button>
              </div>

              {/* Content */}
              <div className="flex-1 overflow-y-auto p-4">
                <p className="text-sm text-text-muted mb-4">
                  These tools are available to AVA for answering your questions. Tools are
                  automatically selected based on the query context.
                </p>

                {loading ? (
                  <div className="flex items-center justify-center py-8">
                    <motion.div
                      className="w-8 h-8 border-2 border-accent-primary border-t-transparent rounded-full"
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    />
                  </div>
                ) : error ? (
                  <div className="text-center py-8 text-state-confusion">{error}</div>
                ) : (
                  <div className="space-y-3">
                    {tools.map((tool, index) => {
                      const config = getToolConfig(tool.name);
                      const Icon = config.icon;

                      return (
                        <motion.div
                          key={tool.name}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: index * 0.05 }}
                          className={cn(
                            "p-4 rounded-xl border transition-all",
                            tool.enabled
                              ? "border-neural-hover bg-neural-elevated/50"
                              : "border-neural-hover/50 bg-neural-surface/50 opacity-60"
                          )}
                          whileHover={{ scale: 1.02, borderColor: "rgba(0, 212, 200, 0.3)" }}
                        >
                          <div className="flex items-start gap-3">
                            <motion.div
                              className={cn("p-2 rounded-lg", config.bgColor)}
                              whileHover={{
                                boxShadow: `0 0 20px ${config.color.replace("text-", "").replace("-400", "")}40`,
                              }}
                            >
                              <Icon className={cn("w-5 h-5", config.color)} />
                            </motion.div>
                            <div className="flex-1">
                              <div className="flex items-center justify-between">
                                <h3 className="font-medium text-text-primary capitalize">
                                  {config.name}
                                </h3>
                                <span
                                  className={cn(
                                    "px-2 py-0.5 rounded text-xs",
                                    tool.enabled
                                      ? "bg-state-flow/20 text-state-flow"
                                      : "bg-text-muted/20 text-text-muted"
                                  )}
                                >
                                  {tool.enabled ? "Enabled" : "Disabled"}
                                </span>
                              </div>
                              <p className="text-sm text-text-muted mt-1">
                                {tool.description || config.description}
                              </p>
                            </div>
                          </div>
                        </motion.div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Footer */}
              <div className="p-4 border-t border-neural-hover">
                <motion.button
                  onClick={fetchTools}
                  className="w-full py-2 rounded-lg bg-accent-primary/20 text-accent-primary text-sm font-medium hover:bg-accent-primary/30 transition-colors"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Refresh Tools
                </motion.button>
              </div>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
