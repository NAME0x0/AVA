"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertTriangle, RefreshCw, Home, Bug } from "lucide-react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

/**
 * ErrorBoundary - Catches React errors and displays a fallback UI
 *
 * Features:
 * - Catches JavaScript errors anywhere in the child component tree
 * - Logs error information for debugging
 * - Displays a user-friendly error message
 * - Provides recovery options (refresh, go home)
 */
export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
    this.setState({ errorInfo });

    // You could also log to an error reporting service here
    // logErrorToService(error, errorInfo);
  }

  private handleRefresh = () => {
    window.location.reload();
  };

  private handleGoHome = () => {
    window.location.href = "/";
  };

  private handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  public render() {
    if (this.state.hasError) {
      // Custom fallback provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default error UI
      return (
        <div className="min-h-screen bg-neural-void flex items-center justify-center p-4">
          <div className="max-w-md w-full bg-neural-surface border border-neural-hover rounded-2xl p-6 shadow-xl">
            {/* Error Icon */}
            <div className="flex justify-center mb-4">
              <div className="w-16 h-16 rounded-full bg-red-500/20 flex items-center justify-center">
                <AlertTriangle className="w-8 h-8 text-red-500" />
              </div>
            </div>

            {/* Error Title */}
            <h1 className="text-xl font-semibold text-text-primary text-center mb-2">
              Something went wrong
            </h1>

            {/* Error Description */}
            <p className="text-text-muted text-center mb-4">
              AVA encountered an unexpected error. Don&apos;t worry, your data is safe.
            </p>

            {/* Error Details (collapsible) */}
            {this.state.error && (
              <details className="mb-4 p-3 bg-neural-elevated rounded-lg border border-neural-hover">
                <summary className="text-sm text-text-muted cursor-pointer flex items-center gap-2">
                  <Bug className="w-4 h-4" />
                  Technical Details
                </summary>
                <div className="mt-2 text-xs font-mono text-red-400 overflow-x-auto">
                  <div className="font-semibold">{this.state.error.name}</div>
                  <div className="opacity-80">{this.state.error.message}</div>
                  {this.state.errorInfo && (
                    <pre className="mt-2 text-[10px] opacity-60 whitespace-pre-wrap">
                      {this.state.errorInfo.componentStack}
                    </pre>
                  )}
                </div>
              </details>
            )}

            {/* Action Buttons */}
            <div className="flex flex-col gap-2">
              <button
                onClick={this.handleRefresh}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-accent-primary text-neural-void rounded-xl font-medium hover:bg-accent-dim transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                Refresh Page
              </button>

              <button
                onClick={this.handleReset}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-neural-elevated text-text-primary rounded-xl font-medium hover:bg-neural-hover transition-colors"
              >
                Try Again
              </button>

              <button
                onClick={this.handleGoHome}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 text-text-muted hover:text-text-primary transition-colors"
              >
                <Home className="w-4 h-4" />
                Go to Home
              </button>
            </div>

            {/* Help Text */}
            <p className="mt-4 text-xs text-text-muted text-center">
              If this keeps happening, please{" "}
              <a
                href="https://github.com/NAME0x0/AVA/issues"
                target="_blank"
                rel="noopener noreferrer"
                className="text-accent-primary hover:underline"
              >
                report the issue
              </a>
              .
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * withErrorBoundary - HOC to wrap a component with ErrorBoundary
 */
export function withErrorBoundary<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  fallback?: ReactNode
) {
  return function WithErrorBoundary(props: P) {
    return (
      <ErrorBoundary fallback={fallback}>
        <WrappedComponent {...props} />
      </ErrorBoundary>
    );
  };
}
