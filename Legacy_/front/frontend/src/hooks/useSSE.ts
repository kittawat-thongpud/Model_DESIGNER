/**
 * SSE Hooks — React hooks for Server-Sent Events streaming.
 *
 * useTrainingStream: subscribes to /api/train/{jobId}/stream for live epoch metrics.
 * useLogStream: subscribes to /api/logs/stream for live system log entries.
 * useJobLogStream: subscribes to /api/jobs/{jobId}/logs/stream for live job logs.
 */
import { useEffect, useRef, useCallback } from 'react';

const API_BASE = 'http://localhost:8000';

// ─── Types ──────────────────────────────────────────────────────────────────

export interface TrainingEvent {
  type: 'epoch' | 'complete' | 'stopped' | 'error';
  job_id: string;
  epoch?: number;
  total_epochs?: number;
  train_loss?: number;
  train_accuracy?: number;
  val_loss?: number | null;
  val_accuracy?: number | null;
  lr?: number;
  epoch_time?: number;
  gpu_memory_mb?: number | null;
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  status: string;
  weight_id?: string;
  message?: string;
}

export interface LogEvent {
  timestamp: string;
  category: string;
  level: string;
  message: string;
  data: Record<string, unknown>;
}

// ─── Generic SSE hook ───────────────────────────────────────────────────────

interface UseSSEOptions<T> {
  /** Full URL path (without base), e.g. "/api/train/abc123/stream" */
  url: string;
  /** Called for each incoming event */
  onEvent: (event: T) => void;
  /** Called when the stream ends (terminal event or error) */
  onEnd?: () => void;
  /** Whether the hook should be active */
  enabled?: boolean;
}

function useSSE<T>({ url, onEvent, onEnd, enabled = true }: UseSSEOptions<T>) {
  const eventSourceRef = useRef<EventSource | null>(null);
  const onEventRef = useRef(onEvent);
  const onEndRef = useRef(onEnd);

  // Keep callback refs fresh without re-subscribing
  useEffect(() => { onEventRef.current = onEvent; }, [onEvent]);
  useEffect(() => { onEndRef.current = onEnd; }, [onEnd]);

  useEffect(() => {
    if (!enabled) return;

    const es = new EventSource(`${API_BASE}${url}`);
    eventSourceRef.current = es;

    const handleMessage = (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data) as T;
        onEventRef.current(data);
      } catch {
        // Ignore unparseable messages
      }
    };

    // Listen to all named event types
    const eventTypes = ['epoch', 'complete', 'stopped', 'error', 'message'];
    for (const type of eventTypes) {
      es.addEventListener(type, handleMessage);
    }

    // Also handle unnamed events
    es.onmessage = handleMessage;

    es.onerror = () => {
      // EventSource auto-reconnects by default; if the server closes
      // the stream (terminal event), readyState will be CLOSED.
      if (es.readyState === EventSource.CLOSED) {
        onEndRef.current?.();
      }
    };

    return () => {
      es.close();
      eventSourceRef.current = null;
    };
  }, [url, enabled]);

  const close = useCallback(() => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
  }, []);

  return { close };
}

// ─── Training Stream ────────────────────────────────────────────────────────

export function useTrainingStream(
  jobId: string | null,
  onEvent: (event: TrainingEvent) => void,
  onEnd?: () => void,
) {
  return useSSE<TrainingEvent>({
    url: `/api/train/${jobId}/stream`,
    onEvent,
    onEnd,
    enabled: !!jobId,
  });
}

// ─── System Log Stream ──────────────────────────────────────────────────────

export function useLogStream(
  onEvent: (event: LogEvent) => void,
  enabled = true,
) {
  return useSSE<LogEvent>({
    url: '/api/logs/stream',
    onEvent,
    enabled,
  });
}

// ─── Job Log Stream ─────────────────────────────────────────────────────────

export function useJobLogStream(
  jobId: string | null,
  onEvent: (event: LogEvent) => void,
) {
  return useSSE<LogEvent>({
    url: `/api/jobs/${jobId}/logs/stream`,
    onEvent,
    enabled: !!jobId,
  });
}
