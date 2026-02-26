"""
BaseJsonStorage — generic file-based JSON persistence with LRU caching.

Eliminates the duplicated save/load/list/delete pattern across
storage.py, build_storage.py, job_storage.py, weight_storage.py.

Each subclass or instance simply specifies a directory and file suffix.

Caching:
  - Per-record LRU cache (configurable max size) for load() calls.
  - An in-memory index (_index.json) for fast list_all() without globbing.
  - Cache is invalidated on save() and delete().
"""
from __future__ import annotations
import json
import shutil
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable


class BaseJsonStorage:
    """Generic JSON-file storage with save/load/list/delete/find operations + LRU caching."""

    def __init__(
        self,
        directory: Path,
        suffix: str = ".json",
        *,
        folder_mode: str | None = None,
        sort_key: Callable[[Path], Any] | None = None,
        cache_max_size: int = 128,
    ):
        """Initialise storage.

        Args:
            directory: Root directory containing records.
            suffix: File suffix for flat-file mode (ignored when *folder_mode* is set).
            folder_mode: If set, each record lives in a subfolder and the JSON
                file is named ``folder_mode`` (e.g. ``"record.json"`` or
                ``"meta.json"``).  The record_id is the subfolder name.
            sort_key: Optional custom sort key for directory listing.
            cache_max_size: Maximum LRU cache entries.
        """
        self.directory = directory
        self.suffix = suffix
        self.folder_mode = folder_mode
        self.sort_key = sort_key or (lambda p: p.stat().st_mtime)
        self.directory.mkdir(parents=True, exist_ok=True)

        # ── LRU cache ──
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_max = cache_max_size
        self._lock = threading.Lock()

        # ── Index ──
        self._index_path = self.directory / "_index.json"
        self._index: dict[str, dict] | None = None  # lazy loaded

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _path(self, record_id: str) -> Path:
        if self.folder_mode:
            d = self.directory / record_id
            d.mkdir(parents=True, exist_ok=True)
            return d / self.folder_mode
        return self.directory / f"{record_id}{self.suffix}"

    def record_dir(self, record_id: str) -> Path:
        """Return the folder for a record (folder_mode only). Creates it if needed."""
        d = self.directory / record_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _cache_put(self, record_id: str, data: dict) -> None:
        with self._lock:
            if record_id in self._cache:
                self._cache.move_to_end(record_id)
            self._cache[record_id] = data
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)

    def _cache_get(self, record_id: str) -> dict | None:
        with self._lock:
            if record_id in self._cache:
                self._cache.move_to_end(record_id)
                return self._cache[record_id]
        return None

    def _cache_remove(self, record_id: str) -> None:
        with self._lock:
            self._cache.pop(record_id, None)

    def _cache_clear(self) -> None:
        with self._lock:
            self._cache.clear()

    # ── Index management ─────────────────────────────────────────────────────

    def _load_index(self) -> dict[str, dict]:
        """Load or rebuild the index. Index maps record_id → summary metadata."""
        if self._index is not None:
            return self._index

        if self._index_path.exists():
            try:
                with open(self._index_path) as f:
                    self._index = json.load(f)
                return self._index
            except Exception:
                pass

        # Rebuild index from disk
        self._rebuild_index()
        return self._index  # type: ignore[return-value]

    def _rebuild_index(self) -> None:
        """Scan all files and rebuild the index."""
        idx: dict[str, dict] = {}
        if self.folder_mode:
            # Folder-per-record: scan subdirectories
            for d in self.directory.iterdir():
                if not d.is_dir() or d.name.startswith("_"):
                    continue
                json_path = d / self.folder_mode
                if not json_path.exists():
                    continue
                try:
                    mtime = json_path.stat().st_mtime
                    idx[d.name] = {"mtime": mtime}
                except Exception:
                    continue
        else:
            # Flat-file mode: scan *.suffix files
            pattern = f"*{self.suffix}"
            for p in self.directory.glob(pattern):
                if p.name == "_index.json":
                    continue
                record_id = p.name.removesuffix(self.suffix)
                try:
                    mtime = p.stat().st_mtime
                    idx[record_id] = {"mtime": mtime}
                except Exception:
                    continue
        self._index = idx
        self._save_index()

    def _save_index(self) -> None:
        """Persist the index to disk."""
        if self._index is None:
            return
        try:
            with open(self._index_path, "w") as f:
                json.dump(self._index, f, indent=1, default=str)
        except Exception:
            pass  # Non-critical — index is rebuilt on next startup

    def _index_add(self, record_id: str) -> None:
        """Add or update an entry in the index."""
        idx = self._load_index()
        path = self._path(record_id)
        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = 0
        idx[record_id] = {"mtime": mtime}
        self._save_index()

    def _index_remove(self, record_id: str) -> None:
        """Remove an entry from the index."""
        idx = self._load_index()
        idx.pop(record_id, None)
        self._save_index()

    # ── Public API ───────────────────────────────────────────────────────────

    def save(self, record_id: str, data: dict) -> str:
        """Save a record to disk. Invalidates cache. Returns the record_id."""
        path = self._path(record_id)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        self._cache_put(record_id, data)
        self._index_add(record_id)
        return record_id

    def load(self, record_id: str) -> dict | None:
        """Load a record — cache hit first, then disk. Returns None if not found."""
        cached = self._cache_get(record_id)
        if cached is not None:
            return cached

        path = self._path(record_id)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        self._cache_put(record_id, data)
        return data

    def list_all(self, **filters: Any) -> list[dict]:
        """
        List all records, optionally filtered by top-level field values.
        Returns most-recent-first by default.
        Uses index for ordering, loads from cache or disk as needed.
        """
        idx = self._load_index()

        # Sort by mtime descending
        sorted_ids = sorted(
            idx.keys(),
            key=lambda rid: idx[rid].get("mtime", 0),
            reverse=True,
        )

        results: list[dict] = []
        for record_id in sorted_ids:
            data = self.load(record_id)
            if data is None:
                # Stale index entry — file was deleted externally
                self._index_remove(record_id)
                continue

            # Apply filters
            skip = False
            for field, value in filters.items():
                if value is not None and data.get(field) != value:
                    skip = True
                    break
            if skip:
                continue

            data["_record_id"] = record_id
            results.append(data)
        return results

    def delete(self, record_id: str) -> bool:
        """Delete a record from disk, cache, and index. Returns True if file existed.

        In folder_mode, removes the entire record subfolder (including
        ancillary files like .pt weights, log files, checkpoints, etc.).
        """
        if self.folder_mode:
            folder = self.directory / record_id
            existed = folder.is_dir()
            if existed:
                shutil.rmtree(folder, ignore_errors=True)
        else:
            path = self._path(record_id)
            existed = path.exists()
            if existed:
                path.unlink()
        self._cache_remove(record_id)
        self._index_remove(record_id)
        return existed

    def find_by_field(self, field: str, value: str, case_insensitive: bool = False) -> dict | None:
        """Find a record where field matches value. Returns first match or None."""
        for data in self.list_all():
            stored = data.get(field, "")
            if case_insensitive:
                if str(stored).lower().strip() == str(value).lower().strip():
                    return data
            elif stored == value:
                return data
        return None

    def exists(self, record_id: str) -> bool:
        """Check if a record file exists on disk."""
        return self._path(record_id).exists()

    def invalidate_cache(self) -> None:
        """Clear all cached data and force index rebuild on next access."""
        self._cache_clear()
        self._index = None
