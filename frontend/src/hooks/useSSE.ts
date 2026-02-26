/**
 * useSSE — hook for Server-Sent Events subscriptions.
 */
import { useEffect, useRef, useCallback } from 'react';
import { SSE_EVENTS } from '../constants';

export function useSSE(
  url: string | null,
  onEvent: (event: string, data: unknown) => void,
  onError?: () => void,
) {
  const cbRef = useRef(onEvent);
  cbRef.current = onEvent;

  const errRef = useRef(onError);
  errRef.current = onError;

  useEffect(() => {
    if (!url) return;
    console.log('[SSE] Connecting to', url);
    const es = new EventSource(url);

    es.onopen = () => console.log('[SSE] Connected:', url);

    const handler = (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data);
        console.log('[SSE] Event:', e.type, data?.type, data?.phase, 'batch:', data?.batch);
        cbRef.current(e.type, data);
      } catch {
        console.log('[SSE] Raw event:', e.type, e.data?.slice?.(0, 100));
        cbRef.current(e.type, e.data);
      }
    };

    es.addEventListener(SSE_EVENTS.EPOCH, handler);
    es.addEventListener(SSE_EVENTS.PROGRESS, handler);
    es.addEventListener(SSE_EVENTS.ITERATION, handler);
    es.addEventListener(SSE_EVENTS.COMPLETE, handler);
    es.addEventListener(SSE_EVENTS.STOPPED, handler);
    es.addEventListener(SSE_EVENTS.ERROR, (e) => {
      console.log('[SSE] Error event, readyState:', es.readyState);
      if (es.readyState === EventSource.CLOSED) {
        errRef.current?.();
      }
    });
    es.addEventListener(SSE_EVENTS.MESSAGE, handler);
    es.onerror = (e) => console.warn('[SSE] onerror:', e, 'readyState:', es.readyState);

    return () => {
      console.log('[SSE] Closing:', url);
      es.close();
    };
  }, [url]);
}
