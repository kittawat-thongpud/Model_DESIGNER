/**
 * Weights Store — shared cache for weight list across all pages.
 * Replaces redundant api.listWeights() calls in WeightsPage,
 * WeightDetailPage, CreateJobModal, TrainingDesignerPage, etc.
 */
import { create } from 'zustand';
import { api } from '../services/api';
import type { WeightRecord } from '../types';

const STALE_MS = 30_000;

interface WeightsState {
  weights: WeightRecord[];
  loading: boolean;
  lastFetched: number;
  /** Currently active model filter (null = all). */
  currentModelId: string | null;

  /** Load weights, optionally filtered by model. */
  load: (modelId?: string | null, force?: boolean) => Promise<void>;
  invalidate: () => void;
  getById: (id: string) => WeightRecord | undefined;
}

export const useWeightsStore = create<WeightsState>((set, get) => ({
  weights: [],
  loading: false,
  lastFetched: 0,
  currentModelId: null,

  load: async (modelId: string | null = null, force = false) => {
    const now = Date.now();
    const state = get();
    const filterChanged = modelId !== state.currentModelId;
    if (!force && !filterChanged && state.lastFetched > 0 && now - state.lastFetched < STALE_MS) {
      return;
    }
    if (state.loading && !filterChanged) return;
    set({ loading: true, currentModelId: modelId });
    try {
      const weights = await api.listWeights(modelId ?? undefined);
      set({ weights, lastFetched: Date.now(), loading: false });
    } catch {
      set({ loading: false });
    }
  },

  invalidate: () => {
    set({ lastFetched: 0 });
  },

  getById: (id: string) => get().weights.find((w) => w.weight_id === id),
}));
