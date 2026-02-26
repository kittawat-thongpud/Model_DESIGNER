/**
 * Datasets Store — shared cache for dataset list across pages.
 * Replaces redundant api.listDatasets() calls in DatasetsPage, CreateJobModal, etc.
 */
import { create } from 'zustand';
import { api } from '../services/api';
import type { DatasetInfo } from '../types';

const STALE_MS = 60_000; // datasets change rarely

interface DatasetsState {
  datasets: DatasetInfo[];
  loading: boolean;
  lastFetched: number;

  load: (force?: boolean) => Promise<void>;
  invalidate: () => void;
  getByName: (name: string) => DatasetInfo | undefined;
}

export const useDatasetsStore = create<DatasetsState>((set, get) => ({
  datasets: [],
  loading: false,
  lastFetched: 0,

  load: async (force = false) => {
    const now = Date.now();
    const state = get();
    if (!force && state.lastFetched > 0 && now - state.lastFetched < STALE_MS) {
      return;
    }
    if (state.loading) return;
    set({ loading: true });
    try {
      const datasets = await api.listDatasets();
      set({ datasets, lastFetched: Date.now(), loading: false });
    } catch {
      set({ loading: false });
    }
  },

  invalidate: () => {
    set({ lastFetched: 0 });
  },

  getByName: (name: string) => get().datasets.find((d) => d.name === name),
}));
