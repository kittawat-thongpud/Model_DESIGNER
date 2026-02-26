/**
 * Toast Store — lightweight notification system.
 * Usage: useToastStore.getState().show('Copied!', 'success')
 */
import { create } from 'zustand';

export interface ToastItem {
  id: number;
  message: string;
  type: 'info' | 'success' | 'error';
}

interface ToastState {
  toasts: ToastItem[];
  show: (message: string, type?: 'info' | 'success' | 'error', durationMs?: number) => void;
  dismiss: (id: number) => void;
}

let _nextId = 1;

export const useToastStore = create<ToastState>((set, get) => ({
  toasts: [],

  show: (message, type = 'info', durationMs = 2500) => {
    const id = _nextId++;
    set({ toasts: [...get().toasts, { id, message, type }] });
    setTimeout(() => get().dismiss(id), durationMs);
  },

  dismiss: (id) => {
    set({ toasts: get().toasts.filter((t) => t.id !== id) });
  },
}));
