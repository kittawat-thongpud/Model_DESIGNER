/**
 * Shared constants used across multiple frontend components.
 * Consolidates duplicated literals from TrainJobsPage, JobDetailPage, useSSE, etc.
 */

/** Tailwind class strings for job status badges. */
export const JOB_STATUS_COLORS: Record<string, string> = {
  running: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  completed: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  failed: 'bg-red-500/10 text-red-400 border-red-500/20',
  stopped: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  pending: 'bg-slate-500/10 text-slate-400 border-slate-500/20',
};

/** Display metadata for training pipeline block roles. */
export const ROLE_META: Record<string, { label: string; color: string }> = {
  setup:   { label: 'Input',   color: '#06b6d4' },
  epoch:   { label: 'Process', color: '#a855f7' },
  monitor: { label: 'Monitor', color: '#ec4899' },
  post:    { label: 'Post',    color: '#14b8a6' },
};

/** SSE event names emitted by the training stream. */
export const SSE_EVENTS = {
  EPOCH: 'epoch',
  ITERATION: 'iteration',
  PROGRESS: 'progress',
  COMPLETE: 'complete',
  STOPPED: 'stopped',
  ERROR: 'error',
  MESSAGE: 'message',
} as const;
