/**
 * Theme Store — manages dark/light theme with localStorage persistence.
 * Applies a `data-theme` attribute on <html> so CSS can switch variable sets.
 */
import { create } from 'zustand';

export type ThemeMode = 'dark' | 'light';

interface ThemeState {
  theme: ThemeMode;
  toggle: () => void;
  set: (mode: ThemeMode) => void;
}

function getInitialTheme(): ThemeMode {
  try {
    const saved = localStorage.getItem('md-theme');
    if (saved === 'light' || saved === 'dark') return saved;
  } catch { /* ignore */ }
  return 'dark';
}

function applyTheme(mode: ThemeMode) {
  document.documentElement.setAttribute('data-theme', mode);
  try { localStorage.setItem('md-theme', mode); } catch { /* ignore */ }
}

// Apply immediately on load (before React mounts)
applyTheme(getInitialTheme());

export const useThemeStore = create<ThemeState>((set, get) => ({
  theme: getInitialTheme(),

  toggle: () => {
    const next = get().theme === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    set({ theme: next });
  },

  set: (mode) => {
    applyTheme(mode);
    set({ theme: mode });
  },
}));
