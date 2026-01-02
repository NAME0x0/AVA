'use client';

/**
 * Toast Notification System
 * =========================
 *
 * Provides non-intrusive notifications for backend status changes
 * and other important events.
 */

import { useState, useEffect, useCallback, createContext, useContext } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  CheckCircle,
  XCircle,
  AlertTriangle,
  Info,
  X,
  Wifi,
  WifiOff,
} from 'lucide-react';
import { cn } from '@/lib/utils';

type ToastType = 'success' | 'error' | 'warning' | 'info' | 'connection';

interface Toast {
  id: string;
  type: ToastType;
  title: string;
  message?: string;
  duration?: number;
}

interface ToastContextValue {
  toasts: Toast[];
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

/**
 * Hook to access toast notifications.
 */
export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
}

/**
 * Toast Provider - Wrap your app with this to enable toast notifications.
 */
export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = useCallback((toast: Omit<Toast, 'id'>) => {
    const id = Math.random().toString(36).slice(2, 11);
    setToasts((prev) => [...prev, { ...toast, id }]);
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ toasts, addToast, removeToast }}>
      {children}
      <ToastContainer />
    </ToastContext.Provider>
  );
}

/**
 * Toast container - renders all active toasts.
 */
function ToastContainer() {
  const { toasts, removeToast } = useToast();

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-md">
      <AnimatePresence mode="popLayout">
        {toasts.map((toast) => (
          <ToastItem key={toast.id} toast={toast} onClose={() => removeToast(toast.id)} />
        ))}
      </AnimatePresence>
    </div>
  );
}

/**
 * Individual toast item.
 */
function ToastItem({ toast, onClose }: { toast: Toast; onClose: () => void }) {
  useEffect(() => {
    const duration = toast.duration ?? 5000;
    if (duration > 0) {
      const timer = setTimeout(onClose, duration);
      return () => clearTimeout(timer);
    }
  }, [toast.duration, onClose]);

  const icons: Record<ToastType, React.ReactNode> = {
    success: <CheckCircle className="w-5 h-5 text-green-400" />,
    error: <XCircle className="w-5 h-5 text-red-400" />,
    warning: <AlertTriangle className="w-5 h-5 text-yellow-400" />,
    info: <Info className="w-5 h-5 text-blue-400" />,
    connection: toast.title.includes('Connected') ? (
      <Wifi className="w-5 h-5 text-green-400" />
    ) : (
      <WifiOff className="w-5 h-5 text-red-400" />
    ),
  };

  const bgColors: Record<ToastType, string> = {
    success: 'bg-green-500/10 border-green-500/20',
    error: 'bg-red-500/10 border-red-500/20',
    warning: 'bg-yellow-500/10 border-yellow-500/20',
    info: 'bg-blue-500/10 border-blue-500/20',
    connection: toast.title.includes('Connected')
      ? 'bg-green-500/10 border-green-500/20'
      : 'bg-red-500/10 border-red-500/20',
  };

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95, transition: { duration: 0.1 } }}
      className={cn(
        'flex items-start gap-3 p-4 rounded-lg border backdrop-blur-sm',
        'shadow-lg shadow-black/20',
        bgColors[toast.type]
      )}
    >
      <div className="shrink-0">{icons[toast.type]}</div>
      <div className="flex-1 min-w-0">
        <p className="font-medium text-white/90 text-sm">{toast.title}</p>
        {toast.message && (
          <p className="text-sm text-white/60 mt-1">{toast.message}</p>
        )}
      </div>
      <button
        onClick={onClose}
        className="shrink-0 text-white/40 hover:text-white/80 transition-colors"
      >
        <X className="w-4 h-4" />
      </button>
    </motion.div>
  );
}

/**
 * Hook to monitor backend connection and show toasts.
 */
export function useBackendStatusToasts() {
  const { addToast } = useToast();
  const [lastStatus, setLastStatus] = useState<'connected' | 'disconnected' | null>(null);

  const notifyConnected = useCallback(() => {
    if (lastStatus !== 'connected') {
      addToast({
        type: 'connection',
        title: 'Connected to AVA Backend',
        message: 'All systems operational',
        duration: 3000,
      });
      setLastStatus('connected');
    }
  }, [addToast, lastStatus]);

  const notifyDisconnected = useCallback((reason?: string) => {
    if (lastStatus !== 'disconnected') {
      addToast({
        type: 'connection',
        title: 'Disconnected from Backend',
        message: reason || 'Attempting to reconnect...',
        duration: 5000,
      });
      setLastStatus('disconnected');
    }
  }, [addToast, lastStatus]);

  const notifyError = useCallback((error: string) => {
    addToast({
      type: 'error',
      title: 'Backend Error',
      message: error,
      duration: 7000,
    });
  }, [addToast]);

  return { notifyConnected, notifyDisconnected, notifyError };
}

export default ToastProvider;
