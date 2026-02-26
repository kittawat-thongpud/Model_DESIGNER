import { useState, useCallback, type Dispatch, type SetStateAction } from 'react';

const PREFIX = 'md-ui-';

/**
 * Drop-in replacement for useState that persists the value in localStorage.
 *
 * Usage:
 *   const [val, setVal] = usePersistedState('dataset.split', 'train');
 *
 * The key is automatically prefixed with "md-ui-" to avoid collisions.
 * Supports primitives, arrays, and plain objects (JSON-serialisable values).
 */
export function usePersistedState<T>(
  key: string,
  defaultValue: T,
): [T, Dispatch<SetStateAction<T>>] {
  const storageKey = PREFIX + key;

  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      if (raw !== null) return JSON.parse(raw) as T;
    } catch { /* ignore parse / access errors */ }
    return defaultValue;
  });

  const setPersisted: Dispatch<SetStateAction<T>> = useCallback(
    (action) => {
      setValue((prev) => {
        const next = action instanceof Function ? action(prev) : action;
        try { localStorage.setItem(storageKey, JSON.stringify(next)); } catch { /* ignore */ }
        return next;
      });
    },
    [storageKey],
  );

  return [value, setPersisted];
}
