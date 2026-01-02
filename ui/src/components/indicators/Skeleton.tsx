/**
 * Loading Skeleton Components
 * ===========================
 *
 * Provides skeleton loading states for better perceived performance.
 * Uses CSS animations with tailored shapes for different content types.
 */

import { cn } from "@/lib/utils";

interface SkeletonProps {
  className?: string;
}

/**
 * Base skeleton component with pulse animation.
 */
export function Skeleton({ className }: SkeletonProps) {
  return (
    <div
      className={cn(
        "animate-pulse rounded-md bg-white/5",
        className
      )}
    />
  );
}

/**
 * Skeleton for a chat message bubble.
 */
export function MessageSkeleton({ isUser = false }: { isUser?: boolean }) {
  return (
    <div className={cn("flex gap-3 p-4", isUser && "flex-row-reverse")}>
      {/* Avatar */}
      <Skeleton className="h-8 w-8 rounded-full shrink-0" />
      
      {/* Message content */}
      <div className={cn("flex flex-col gap-2", isUser ? "items-end" : "items-start")}>
        {/* Header */}
        <Skeleton className="h-4 w-20" />
        
        {/* Message lines */}
        <div className="space-y-2">
          <Skeleton className="h-4 w-64" />
          <Skeleton className="h-4 w-48" />
          <Skeleton className="h-4 w-56" />
        </div>
      </div>
    </div>
  );
}

/**
 * Skeleton for the metrics panel.
 */
export function MetricsSkeleton() {
  return (
    <div className="space-y-4 p-4">
      {/* Title */}
      <Skeleton className="h-6 w-32" />
      
      {/* Metric items */}
      {[...Array(4)].map((_, i) => (
        <div key={i} className="flex items-center justify-between">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-4 w-16" />
        </div>
      ))}
      
      {/* Chart placeholder */}
      <Skeleton className="h-24 w-full rounded-lg" />
    </div>
  );
}

/**
 * Skeleton for the settings panel.
 */
export function SettingsSkeleton() {
  return (
    <div className="space-y-6 p-4">
      {/* Section header */}
      <Skeleton className="h-6 w-24" />
      
      {/* Setting items */}
      {[...Array(3)].map((_, i) => (
        <div key={i} className="space-y-2">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-10 w-full rounded" />
        </div>
      ))}
    </div>
  );
}

/**
 * Skeleton for the tools panel.
 */
export function ToolsSkeleton() {
  return (
    <div className="space-y-4 p-4">
      {/* Title */}
      <Skeleton className="h-6 w-24" />
      
      {/* Tool items */}
      {[...Array(5)].map((_, i) => (
        <div key={i} className="flex items-center gap-3">
          <Skeleton className="h-8 w-8 rounded" />
          <div className="flex-1 space-y-1">
            <Skeleton className="h-4 w-24" />
            <Skeleton className="h-3 w-full" />
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * Full-page loading skeleton for initial app load.
 */
export function AppSkeleton() {
  return (
    <div className="flex h-screen bg-gradient-to-br from-zinc-950 via-zinc-900 to-zinc-950">
      {/* Left sidebar */}
      <div className="w-64 border-r border-white/5 p-4">
        <SettingsSkeleton />
      </div>
      
      {/* Main chat area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="h-14 border-b border-white/5 flex items-center px-4">
          <Skeleton className="h-6 w-48" />
        </div>
        
        {/* Messages */}
        <div className="flex-1 p-4 space-y-4">
          <MessageSkeleton />
          <MessageSkeleton isUser />
          <MessageSkeleton />
        </div>
        
        {/* Input area */}
        <div className="border-t border-white/5 p-4">
          <Skeleton className="h-12 w-full rounded-lg" />
        </div>
      </div>
      
      {/* Right sidebar */}
      <div className="w-72 border-l border-white/5 p-4">
        <MetricsSkeleton />
      </div>
    </div>
  );
}

export default Skeleton;
