"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Bug, AlertCircle, ExternalLink, Copy, Check, Info } from "lucide-react";
import { invoke } from "@tauri-apps/api/tauri";
import { cn } from "@/lib/utils";

interface SystemInfo {
  cpu: string;
  memory_total_mb: number;
  memory_used_mb: number;
  gpu: string | null;
}

interface BugReport {
  version: string;
  os: string;
  category: string;
  error_message: string;
  stack_trace: string | null;
  steps: string | null;
  screenshot_path: string | null;
  system_info: SystemInfo;
}

interface BugReportDialogProps {
  isOpen: boolean;
  onClose: () => void;
  errorMessage?: string;
  stackTrace?: string;
}

export function BugReportDialog({
  isOpen,
  onClose,
  errorMessage = "",
  stackTrace,
}: BugReportDialogProps) {
  const [report, setReport] = useState<BugReport | null>(null);
  const [steps, setSteps] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isReportable, setIsReportable] = useState(true);
  const [notReportableReason, setNotReportableReason] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen && errorMessage) {
      checkReportability();
      createReport();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, errorMessage]);

  const checkReportability = async () => {
    try {
      const [reportable, reason] = await invoke<[boolean, string | null]>(
        "is_error_reportable",
        { errorMessage }
      );
      setIsReportable(reportable);
      setNotReportableReason(reason);
    } catch (err) {
      console.error("Failed to check reportability:", err);
      setIsReportable(true);
    }
  };

  const createReport = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const bugReport = await invoke<BugReport>("create_bug_report", {
        errorMessage,
      });
      if (stackTrace) {
        bugReport.stack_trace = stackTrace;
      }
      setReport(bugReport);
    } catch (err) {
      console.error("Failed to create bug report:", err);
      setError("Failed to create bug report");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async () => {
    if (!report) return;

    setIsLoading(true);
    try {
      const reportWithSteps = { ...report, steps: steps || null };
      await invoke("open_bug_report_url", { report: reportWithSteps });
      onClose();
    } catch (err) {
      console.error("Failed to open bug report:", err);
      setError("Failed to open browser. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopyInfo = async () => {
    if (!report) return;

    const info = `
AVA Bug Report
==============
Version: ${report.version}
OS: ${report.os}
Error: ${report.error_message}
CPU: ${report.system_info.cpu}
Memory: ${report.system_info.memory_used_mb}MB / ${report.system_info.memory_total_mb}MB
GPU: ${report.system_info.gpu || "Unknown"}

Stack Trace:
${report.stack_trace || "N/A"}

Steps to Reproduce:
${steps || "Not provided"}
    `.trim();

    try {
      await navigator.clipboard.writeText(info);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case "ApplicationError":
        return "text-state-confusion";
      case "BackendError":
        return "text-amber-400";
      case "UserConfiguration":
      case "MissingDependency":
      case "NetworkError":
        return "text-text-muted";
      default:
        return "text-text-primary";
    }
  };

  const getCategoryLabel = (category: string) => {
    switch (category) {
      case "ApplicationError":
        return "Application Error";
      case "BackendError":
        return "Backend Error";
      case "UserConfiguration":
        return "Configuration Issue";
      case "MissingDependency":
        return "Missing Dependency";
      case "NetworkError":
        return "Network Error";
      default:
        return "Unknown";
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
            onClick={onClose}
          />

          {/* Dialog */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.2 }}
            className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-full max-w-lg"
          >
            <div className="bg-neural-surface border border-neural-hover rounded-xl shadow-2xl overflow-hidden">
              {/* Header */}
              <div className="flex items-center justify-between p-4 border-b border-neural-hover">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-state-confusion/20">
                    <Bug className="w-5 h-5 text-state-confusion" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-text-primary">
                      Report a Bug
                    </h2>
                    <p className="text-xs text-text-muted">
                      Help us improve AVA
                    </p>
                  </div>
                </div>
                <motion.button
                  onClick={onClose}
                  className="p-2 rounded-lg hover:bg-neural-hover transition-colors"
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <X className="w-5 h-5 text-text-muted" />
                </motion.button>
              </div>

              {/* Content */}
              <div className="p-4 space-y-4 max-h-[60vh] overflow-y-auto">
                {/* Not Reportable Warning */}
                {!isReportable && notReportableReason && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-start gap-3 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20"
                  >
                    <Info className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm text-amber-400 font-medium">
                        This may not be a bug
                      </p>
                      <p className="text-xs text-text-muted mt-1">
                        {notReportableReason}
                      </p>
                    </div>
                  </motion.div>
                )}

                {/* Error Message */}
                {errorMessage && (
                  <div className="p-3 rounded-lg bg-neural-elevated/50">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertCircle className="w-4 h-4 text-state-confusion" />
                      <span className="text-sm font-medium text-text-primary">
                        Error
                      </span>
                      {report && (
                        <span
                          className={cn(
                            "text-xs px-2 py-0.5 rounded",
                            getCategoryColor(report.category),
                            "bg-neural-hover"
                          )}
                        >
                          {getCategoryLabel(report.category)}
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-text-muted font-mono break-all">
                      {errorMessage}
                    </p>
                  </div>
                )}

                {/* System Info */}
                {report && (
                  <div className="p-3 rounded-lg bg-neural-elevated/50">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-text-primary">
                        System Information
                      </span>
                      <motion.button
                        onClick={handleCopyInfo}
                        className="p-1.5 rounded hover:bg-neural-hover transition-colors"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        {copied ? (
                          <Check className="w-4 h-4 text-state-flow" />
                        ) : (
                          <Copy className="w-4 h-4 text-text-muted" />
                        )}
                      </motion.button>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <span className="text-text-muted">Version:</span>{" "}
                        <span className="text-text-primary">{report.version}</span>
                      </div>
                      <div>
                        <span className="text-text-muted">OS:</span>{" "}
                        <span className="text-text-primary">{report.os}</span>
                      </div>
                      <div className="col-span-2">
                        <span className="text-text-muted">CPU:</span>{" "}
                        <span className="text-text-primary">
                          {report.system_info.cpu}
                        </span>
                      </div>
                      <div>
                        <span className="text-text-muted">Memory:</span>{" "}
                        <span className="text-text-primary">
                          {report.system_info.memory_used_mb}MB /{" "}
                          {report.system_info.memory_total_mb}MB
                        </span>
                      </div>
                      <div>
                        <span className="text-text-muted">GPU:</span>{" "}
                        <span className="text-text-primary">
                          {report.system_info.gpu || "Unknown"}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Steps to Reproduce */}
                <div>
                  <label className="block text-sm font-medium text-text-primary mb-2">
                    Steps to Reproduce
                  </label>
                  <textarea
                    value={steps}
                    onChange={(e) => setSteps(e.target.value)}
                    placeholder="1. I was doing...\n2. Then I clicked...\n3. The error appeared..."
                    className="w-full h-24 px-3 py-2 rounded-lg bg-neural-elevated border border-neural-hover text-text-primary text-sm resize-none focus:outline-none focus:border-accent-primary placeholder:text-text-muted"
                  />
                </div>

                {/* Error Display */}
                {error && (
                  <div className="p-3 rounded-lg bg-state-confusion/10 border border-state-confusion/20">
                    <p className="text-sm text-state-confusion">{error}</p>
                  </div>
                )}
              </div>

              {/* Footer */}
              <div className="flex items-center justify-end gap-3 p-4 border-t border-neural-hover">
                <motion.button
                  onClick={onClose}
                  className="px-4 py-2 rounded-lg text-sm font-medium text-text-muted hover:text-text-primary hover:bg-neural-hover transition-colors"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Cancel
                </motion.button>
                <motion.button
                  onClick={handleSubmit}
                  disabled={isLoading || !report}
                  className={cn(
                    "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
                    isLoading || !report
                      ? "bg-neural-hover text-text-muted cursor-not-allowed"
                      : "bg-accent-primary text-white hover:bg-accent-primary/90"
                  )}
                  whileHover={!isLoading && report ? { scale: 1.02 } : {}}
                  whileTap={!isLoading && report ? { scale: 0.98 } : {}}
                >
                  <ExternalLink className="w-4 h-4" />
                  {isLoading ? "Opening..." : "Open GitHub Issue"}
                </motion.button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
