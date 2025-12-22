"use client";

import { useState, useCallback, useRef, useEffect, ReactNode } from "react";

interface SplitPaneProps {
  left: ReactNode;
  right: ReactNode;
  defaultRatio?: number; // 0-1, percentage for left pane
  minRatio?: number;
  maxRatio?: number;
  className?: string;
  onRatioChange?: (ratio: number) => void;
}

/**
 * SplitPane - Resizable horizontal split view
 *
 * Features:
 * - Drag to resize
 * - Double-click to reset
 * - Min/max constraints
 * - Smooth transitions
 */
export function SplitPane({
  left,
  right,
  defaultRatio = 0.65,
  minRatio = 0.3,
  maxRatio = 0.85,
  className = "",
  onRatioChange,
}: SplitPaneProps) {
  const [ratio, setRatio] = useState(defaultRatio);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging || !containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      let newRatio = x / rect.width;

      // Clamp to min/max
      newRatio = Math.max(minRatio, Math.min(maxRatio, newRatio));

      setRatio(newRatio);
      onRatioChange?.(newRatio);
    },
    [isDragging, minRatio, maxRatio, onRatioChange]
  );

  const handleDoubleClick = useCallback(() => {
    setRatio(defaultRatio);
    onRatioChange?.(defaultRatio);
  }, [defaultRatio, onRatioChange]);

  // Global mouse event listeners
  useEffect(() => {
    if (isDragging) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    }

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  return (
    <div ref={containerRef} className={`flex h-full ${className}`}>
      {/* Left pane */}
      <div
        className="h-full overflow-hidden"
        style={{ width: `${ratio * 100}%` }}
      >
        {left}
      </div>

      {/* Divider */}
      <div
        className={`
          relative w-1 flex-shrink-0 cursor-col-resize
          bg-neural-hover hover:bg-primary/50
          transition-colors duration-150
          ${isDragging ? "bg-primary" : ""}
        `}
        onMouseDown={handleMouseDown}
        onDoubleClick={handleDoubleClick}
      >
        {/* Drag handle indicator */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
          <div className="flex flex-col gap-0.5">
            <div className="w-1 h-1 rounded-full bg-text-muted" />
            <div className="w-1 h-1 rounded-full bg-text-muted" />
            <div className="w-1 h-1 rounded-full bg-text-muted" />
          </div>
        </div>
      </div>

      {/* Right pane */}
      <div
        className="h-full overflow-hidden"
        style={{ width: `${(1 - ratio) * 100}%` }}
      >
        {right}
      </div>
    </div>
  );
}
