"""
MCP smoke tests — verify all tools return ok=True against real backend data.

Run with:
    cd backend
    ../venv/bin/python -m pytest tests/test_mcp_smoke.py -v

Or standalone (no pytest):
    cd backend
    ../venv/bin/python tests/test_mcp_smoke.py
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.argv = []

import pytest

from app.mcp.tools import models as _m
from app.mcp.tools import jobs as _j
from app.mcp.tools import weights as _w
from app.mcp.tools import benchmark as _b
from app.mcp.tools import training as _t
from app.mcp.filters import apply_view, apply_list_view, paginate
from app.mcp.serializers import safe_dict, ok, err


# ── Helpers ──────────────────────────────────────────────────────────────────

def _first_model_id() -> str | None:
    r = _m.list_models(view="summary")
    items = r.get("items", [])
    return items[0]["model_id"] if items else None


def _first_job_id() -> str | None:
    r = _j.list_jobs(view="summary", limit=1)
    items = r.get("items", [])
    return items[0]["job_id"] if items else None


def _first_weight_id() -> str | None:
    r = _w.list_weights(view="summary", limit=1)
    items = r.get("items", [])
    return items[0]["weight_id"] if items else None


def _first_benchmark_id() -> str | None:
    r = _b.list_benchmarks(view="summary", limit=1)
    items = r.get("items", [])
    return items[0].get("benchmark_id") if items else None


# ── Unit: filters ────────────────────────────────────────────────────────────

class TestFilters:
    def test_summary_strips_heavy_fields(self):
        record = {"job_id": "abc", "model_id": "xyz", "status": "running", "yaml_def": "BIG", "history": [1]}
        out = apply_view(record, "job", view="summary")
        assert "yaml_def" not in out
        assert "job_id" in out

    def test_detail_keeps_fields(self):
        record = {"job_id": "abc", "status": "done", "config": {"epochs": 10}}
        out = apply_view(record, "job", view="detail")
        assert "job_id" in out
        assert "config" in out

    def test_paginate_offset_limit(self):
        items = list(range(100))
        assert paginate(items, limit=10, offset=5) == list(range(5, 15))
        assert paginate(items, limit=None, offset=90) == list(range(90, 100))
        assert paginate(items, limit=5, offset=0) == list(range(5))


# ── Unit: serializers ────────────────────────────────────────────────────────

class TestSerializers:
    def test_nan_inf_become_none(self):
        import math
        r = {"v": float("nan"), "w": float("inf"), "n": 42}
        s = safe_dict(r)
        assert s["v"] is None
        assert s["w"] is None
        assert s["n"] == 42

    def test_ok_wraps_dict(self):
        result = ok({"model_id": "x", "name": "y"})
        assert result["ok"] is True
        assert result["model_id"] == "x"

    def test_err_wraps_message(self):
        result = err("something failed", "test_error")
        assert result["ok"] is False
        assert result["error"] == "test_error"


# ── Integration: model tools ─────────────────────────────────────────────────

class TestModelTools:
    def test_list_models(self):
        r = _m.list_models(view="summary")
        assert r["ok"] is True
        assert isinstance(r["count"], int)
        assert isinstance(r["items"], list)

    def test_list_models_detail(self):
        r = _m.list_models(view="detail")
        assert r["ok"] is True

    def test_list_models_pagination(self):
        r = _m.list_models(limit=2, offset=0)
        assert r["ok"] is True
        assert len(r["items"]) <= 2

    def test_get_model_summary(self):
        mid = _first_model_id()
        if not mid:
            pytest.skip("No models available")
        r = _m.get_model(mid, view="summary")
        assert r["ok"] is True
        assert r["model_id"] == mid

    def test_get_model_detail(self):
        mid = _first_model_id()
        if not mid:
            pytest.skip("No models available")
        r = _m.get_model(mid, view="detail")
        assert r["ok"] is True

    def test_get_model_yaml(self):
        mid = _first_model_id()
        if not mid:
            pytest.skip("No models available")
        r = _m.get_model_yaml(mid)
        assert r["ok"] is True
        assert isinstance(r.get("yaml"), str)
        assert len(r["yaml"]) > 0

    def test_get_model_not_found(self):
        r = _m.get_model("nonexistent_model_xyz")
        assert r["ok"] is False
        assert r["error"] == "not_found"


# ── Integration: job tools ────────────────────────────────────────────────────

class TestJobTools:
    def test_list_jobs(self):
        r = _j.list_jobs(view="summary", limit=5)
        assert r["ok"] is True

    def test_get_job_summary(self):
        jid = _first_job_id()
        if not jid:
            pytest.skip("No jobs available")
        r = _j.get_job(jid, view="summary")
        assert r["ok"] is True
        assert r["job_id"] == jid

    def test_get_job_detail(self):
        jid = _first_job_id()
        if not jid:
            pytest.skip("No jobs available")
        r = _j.get_job(jid, view="detail")
        assert r["ok"] is True

    def test_get_job_logs(self):
        jid = _first_job_id()
        if not jid:
            pytest.skip("No jobs available")
        r = _j.get_job_logs(jid, limit=10)
        assert r["ok"] is True
        assert isinstance(r.get("logs"), list)

    def test_get_job_metrics(self):
        jid = _first_job_id()
        if not jid:
            pytest.skip("No jobs available")
        r = _j.get_job_metrics(jid)
        assert r["ok"] is True

    def test_get_job_history(self):
        jid = _first_job_id()
        if not jid:
            pytest.skip("No jobs available")
        r = _j.get_job_history(jid, limit=5)
        assert r["ok"] is True

    def test_get_job_not_found(self):
        r = _j.get_job("nonexistent_job_xyz")
        assert r["ok"] is False
        assert r["error"] == "not_found"


# ── Integration: weight tools ─────────────────────────────────────────────────

class TestWeightTools:
    def test_list_weights(self):
        r = _w.list_weights(view="summary", limit=5)
        assert r["ok"] is True

    def test_get_weight_summary(self):
        wid = _first_weight_id()
        if not wid:
            pytest.skip("No weights available")
        r = _w.get_weight(wid, view="summary")
        assert r["ok"] is True
        assert r["weight_id"] == wid

    def test_get_weight_detail(self):
        wid = _first_weight_id()
        if not wid:
            pytest.skip("No weights available")
        r = _w.get_weight(wid, view="detail")
        assert r["ok"] is True

    def test_get_weight_lineage(self):
        wid = _first_weight_id()
        if not wid:
            pytest.skip("No weights available")
        r = _w.get_weight_lineage(wid)
        assert r["ok"] is True
        assert isinstance(r.get("lineage"), list)

    def test_get_weight_not_found(self):
        r = _w.get_weight("nonexistent_weight_xyz")
        assert r["ok"] is False

    def test_list_pretrained(self):
        r = _w.list_pretrained_weights(limit=5)
        assert r["ok"] is True


# ── Integration: benchmark tools ─────────────────────────────────────────────

class TestBenchmarkTools:
    def test_list_benchmarks(self):
        r = _b.list_benchmarks(view="summary", limit=5)
        assert r["ok"] is True

    def test_get_benchmark(self):
        bid = _first_benchmark_id()
        if not bid:
            pytest.skip("No benchmarks available")
        r = _b.get_benchmark(bid, view="detail")
        assert r["ok"] is True

    def test_list_benchmark_datasets(self):
        r = _b.list_benchmark_datasets()
        assert r["ok"] is True

    def test_get_benchmark_not_found(self):
        r = _b.get_benchmark("nonexistent_benchmark_xyz")
        assert r["ok"] is False


# ── Integration: training control ─────────────────────────────────────────────

class TestTrainingControlTools:
    def test_get_training_queue(self):
        r = _t.get_training_queue()
        assert r["ok"] is True

    def test_get_workers_health(self):
        r = _t.get_training_workers_health()
        assert r["ok"] is True

    def test_stop_nonexistent_job(self):
        r = _t.stop_training("nonexistent_job_xyz")
        assert r["ok"] is False

    def test_resume_nonexistent_job(self):
        r = _t.resume_training("nonexistent_job_xyz")
        assert r["ok"] is False


# ── Integration: MCP server wiring ───────────────────────────────────────────

class TestMcpServerWiring:
    def test_tools_count(self):
        from app.mcp.server import mcp
        tools = mcp._tool_manager._tools
        assert len(tools) == 30, f"Expected 30 tools, got {len(tools)}: {sorted(tools.keys())}"

    def test_expected_tools_present(self):
        from app.mcp.server import mcp
        tools = set(mcp._tool_manager._tools.keys())
        required = {
            "list_models", "get_model", "get_model_yaml", "create_model", "validate_model",
            "list_datasets", "get_dataset", "preview_dataset",
            "list_jobs", "get_job", "get_job_logs", "get_job_metrics", "get_job_history",
            "list_weights", "get_weight", "get_weight_info", "get_weight_lineage",
            "create_empty_weight", "list_pretrained_weights", "download_pretrained_weight",
            "list_benchmark_datasets", "run_benchmark", "list_benchmarks", "get_benchmark",
            "start_training", "stop_training", "resume_training", "append_training",
            "get_training_queue", "get_training_workers_health",
        }
        missing = required - tools
        assert not missing, f"Missing tools: {missing}"

    def test_resources_registered(self):
        from app.mcp.server import mcp
        static = mcp._resource_manager._resources
        templates = mcp._resource_manager._templates
        total = len(static) + len(templates)
        assert total == 17, f"Expected 17 resources, got {total}"

    def test_asgi_app_creates(self):
        from app.mcp.server import create_mcp_app
        app = create_mcp_app()
        assert app is not None
        assert type(app).__name__ == "Starlette"

    def test_fastapi_mcp_mount(self):
        from app.main import app
        mcp_mounts = [
            r for r in app.routes
            if hasattr(r, "path") and "/mcp" in str(getattr(r, "path", ""))
        ]
        assert len(mcp_mounts) >= 1, "MCP not mounted in FastAPI app"


# ── Standalone runner ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    suites = [
        TestFilters, TestSerializers,
        TestModelTools, TestJobTools, TestWeightTools,
        TestBenchmarkTools, TestTrainingControlTools,
        TestMcpServerWiring,
    ]

    passed, failed = 0, 0
    for suite_cls in suites:
        suite = suite_cls()
        for attr in [a for a in dir(suite) if a.startswith("test_")]:
            label = f"{suite_cls.__name__}.{attr}"
            try:
                getattr(suite, attr)()
                print(f"  ✓ {label}")
                passed += 1
            except pytest.skip.Exception as e:
                print(f"  ~ {label} (skip: {e})")
            except Exception as e:
                print(f"  ✗ {label}: {e}")
                traceback.print_exc()
                failed += 1

    total = passed + failed
    print(f"\nResults: {passed}/{total} passed", "✓" if not failed else "✗")
    sys.exit(0 if not failed else 1)
