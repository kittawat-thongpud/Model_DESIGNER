/**
 * Jobs Store — shared cache for job list with smart polling.
 * Only polls every 5s when running jobs exist; otherwise on-demand only.
 */
import { create } from 'zustand';
import { api } from '../services/api';
import type { JobRecord } from '../types';

const STALE_MS = 30_000;
const POLL_RUNNING_MS = 5_000;

interface JobsState {
  jobs: JobRecord[];
  loading: boolean;
  lastFetched: number;
  _pollTimer: ReturnType<typeof setInterval> | null;

  load: (force?: boolean) => Promise<void>;
  invalidate: () => void;
  getById: (id: string) => JobRecord | undefined;

  /** Start smart polling — polls fast when running jobs exist, stops otherwise. */
  startPolling: () => void;
  stopPolling: () => void;
}

export const useJobsStore = create<JobsState>((set, get) => ({
  jobs: [],
  loading: false,
  lastFetched: 0,
  _pollTimer: null,

  load: async (force = false) => {
    const now = Date.now();
    const state = get();
    if (!force && state.lastFetched > 0 && now - state.lastFetched < STALE_MS) {
      return;
    }
    if (state.loading) return;
    set({ loading: true });
    try {
      const jobs = await api.listJobs();
      set({ jobs, lastFetched: Date.now(), loading: false });
    } catch {
      set({ loading: false });
    }
  },

  invalidate: () => {
    set({ lastFetched: 0 });
  },

  getById: (id: string) => get().jobs.find((j) => j.job_id === id),

  startPolling: () => {
    const state = get();
    if (state._pollTimer) return; // already polling

    const tick = async () => {
      const { jobs } = get();
      const hasRunning = jobs.some((j) => j.status === 'running' || j.status === 'pending');
      if (hasRunning) {
        // Force re-fetch when there are active jobs
        set({ lastFetched: 0 });
        await get().load(true);
      }
    };

    const timer = setInterval(tick, POLL_RUNNING_MS);
    set({ _pollTimer: timer });
    // Initial load
    get().load(true);
  },

  stopPolling: () => {
    const { _pollTimer } = get();
    if (_pollTimer) {
      clearInterval(_pollTimer);
      set({ _pollTimer: null });
    }
  },
}));
