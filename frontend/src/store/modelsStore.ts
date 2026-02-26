/**
 * Models Store — shared cache for model list across all pages.
 * Replaces redundant api.listModels() calls in Dashboard, Designer,
 * Weights, CreateJobModal, LayerPalette, etc.
 */
import { create } from 'zustand';
import { api } from '../services/api';
import type { ModelSummary } from '../types';

const STALE_MS = 30_000; // 30 seconds

interface ModelsState {
  models: ModelSummary[];
  loading: boolean;
  lastFetched: number;

  /** Load models (stale-while-revalidate: returns cache if fresh, refreshes in background). */
  load: (force?: boolean) => Promise<void>;
  /** Force invalidate — next load() will re-fetch. */
  invalidate: () => void;
  /** Get a single model by ID from the cache. */
  getById: (id: string) => ModelSummary | undefined;
}

export const useModelsStore = create<ModelsState>((set, get) => ({
  models: [],
  loading: false,
  lastFetched: 0,

  load: async (force = false) => {
    const now = Date.now();
    const state = get();
    if (!force && state.lastFetched > 0 && now - state.lastFetched < STALE_MS) {
      return; // still fresh
    }
    if (state.loading) return; // already in flight
    set({ loading: true });
    try {
      const models = await api.listModels();
      set({ models, lastFetched: Date.now(), loading: false });
    } catch {
      set({ loading: false });
    }
  },

  invalidate: () => {
    set({ lastFetched: 0 });
  },

  getById: (id: string) => get().models.find((m) => m.model_id === id),
}));
