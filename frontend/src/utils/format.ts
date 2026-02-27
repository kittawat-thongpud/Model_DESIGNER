/**
 * Shared formatting utilities used across multiple pages.
 * Consolidates duplicated helpers from WeightsPage, WeightDetailPage,
 * WeightEditorPage, JobDetailPage, etc.
 */

/** Format byte count to human-readable size (B / KB / MB / GB). */
export function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1073741824) return `${(bytes / 1048576).toFixed(1)} MB`;
  return `${(bytes / 1073741824).toFixed(2)} GB`;
}

/** Format seconds to human-readable duration (e.g. "1.2s", "3m 12s", "1h 5m"). */
export function fmtTime(sec: number | null | undefined): string {
  if (sec == null) return '—';
  if (sec < 60) return `${sec.toFixed(1)}s`;
  if (sec < 3600) {
    const m = Math.floor(sec / 60);
    const s = Math.round(sec % 60);
    return `${m}m ${s}s`;
  }
  return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
}

/** Format an ISO timestamp to relative "X ago" string. */
export function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

/**
 * Extract a short, human-readable dataset name from a raw value.
 * Handles:
 *   - Full paths: "/home/.../jobs/abc123/data.yaml" → resolved via datasetName fallback
 *   - Dataset dir paths: "/home/.../datasets/coco128/data.yaml" → "coco128"
 *   - Plain names: "coco128" → "coco128"
 *
 * @param raw       raw config.data value (path or name)
 * @param datasetName  optional pre-resolved dataset name from job.dataset_name
 */
export function fmtDataset(raw: string | null | undefined, datasetName?: string | null): string {
  if (!raw && !datasetName) return '—';
  // If we have a pre-resolved human-readable name, prefer it
  if (datasetName) return datasetName;
  if (!raw) return '—';
  // If it's a path ending in data.yaml or *.yaml
  if (raw.includes('/') || raw.includes('\\')) {
    const normalized = raw.replace(/\\/g, '/');
    // datasets dir: .../datasets/<name>/data.yaml or <name>.yaml
    const dsMatch = normalized.match(/\/datasets\/([^/]+)\//);
    if (dsMatch) return dsMatch[1];
    // job dir: .../jobs/<job_id>/data.yaml — can't determine dataset, show placeholder
    const jobMatch = normalized.match(/\/jobs\/([a-f0-9]{8,})\//);
    if (jobMatch) return 'custom dataset';
    // fallback: parent directory name
    const parts = normalized.split('/').filter(Boolean);
    const filename = parts[parts.length - 1];
    if (filename.endsWith('.yaml') || filename.endsWith('.yml')) {
      return parts[parts.length - 2] || filename;
    }
    return parts[parts.length - 1];
  }
  return raw;
}

/** Format megabytes to human-readable memory size (MB / GB). */
export function fmtMem(mb: number | null | undefined): string {
  if (mb == null) return '—';
  if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
  return `${mb.toFixed(0)} MB`;
}
