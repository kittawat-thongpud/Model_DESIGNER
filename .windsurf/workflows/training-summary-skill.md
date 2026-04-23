---
description: Regenerate training_summary.md from MCP jobs (per-machine + combined + comparisons)
---

## Goal

Produce a fresh `training_summary.md` that includes **all jobs** (completed/running/stopped/failed) across machines, with:

1. Per-machine summary grouped by `imgsz` and `scale`
2. A combined table (all machines) sorted by `mAP50-95`
3. Grouped tables split by `(scale, imgsz)` that look like the combined table, each sorted by `mAP50-95`
4. Comparison summaries split by `(scale, imgsz)` for **HSG-DET vs YOLOv8**
5. A focused summary for **Mamba-YOLO** as the baseline/compare model

## Inputs

- MCP servers (one per machine):
  - `model-designer-ku4070`
  - `model-designer-ku4070-aj`
  - `model-designer-rase4090`
  - `model-designer-rase4090-2`
- MCP tools you will use:
  - `list_jobs(view="summary")` to get job ids
  - `get_job(view="detail")` for each job id to extract `imgsz`, `optimizer`, etc.

## Output File

- Rewrite (replace) `training_summary.md`

## Definitions (normalize once)

For each job, extract these fields:

- `machine`: which MCP server it came from
- `job_id`
- `model_name`
- `dataset_name`
- `model_scale` (e.g. `n`, `s`, `t`)
- `imgsz` (from `job.config.imgsz`)
- `optimizer` (from `job.config.optimizer`)
- `status` (`running`, `completed`, `stopped`, `failed`)
- `epoch` + `total_epochs`
- Metrics:
  - `best_mAP50_95` (use `job.best_mAP50_95`)
  - `best_mAP50` (use `job.best_mAP50`)

**mAP sorting rule**
- Sort descending by `best_mAP50_95`
- If `best_mAP50_95` is `null`/missing, put the row at the end

**Optimizer display rule**
- Keep the raw string (examples: `auto`, `SGD`, `AdamW`)

## Step 1 — Collect jobs from each machine

For each MCP server:

1. Call `list_jobs(view="summary", limit=200)`.
2. If the call fails (e.g. `no route to host`):
   - Mark that machine as **unreachable** in the report.
   - Continue with the remaining machines.

## Step 2 — Expand each job to detail records

For each `job_id` from Step 1:

1. Call `get_job(job_id, view="detail")`.
2. Parse fields from the job record:
   - `imgsz = job.config.imgsz`
   - `optimizer = job.config.optimizer`
   - `best_mAP50_95 = job.best_mAP50_95`

Store the normalized rows in an in-memory list.

## Step 3 — Write the document header

Include:

- Dataset name (if mixed, show `mixed`)
- Generated timestamp
- Machines list
- Data Sources table:
  - machine, MCP server, reachable/unreachable

## Step 4 — Per-machine summary (Requirement #1)

For each reachable machine:

1. Create a table grouped by `(imgsz, scale)`.
2. For each group, count statuses:
   - running/completed/stopped/failed and total.
3. Create a per-machine job table:
   - Include all jobs for that machine
   - Sort by `best_mAP50_95` desc (missing at bottom)
   - Columns:
     - `#`, `model`, `scale`, `imgsz`, `optimizer`, `status`, `epoch`, `mAP50-95`

## Step 5 — Combined table (Requirement #2)

Create one combined table over all reachable machines:

- Sort by `best_mAP50_95` desc (missing at bottom)
- Columns:
  - `#`, `machine`, `model`, `scale`, `imgsz`, `optimizer`, `status`, `epoch`, `mAP50-95`

## Step 6 — Grouped tables by (scale, imgsz) (Requirement #3)

Create a section: `## Jobs by scale / imgsz (reachable machines)`

1. Bucket all normalized rows by `(model_scale, imgsz)`.
2. For each bucket:
   - Sort rows by `best_mAP50_95` desc (missing at bottom)
   - Render a table **with the same columns as the Combined table**:
     - `#`, `machine`, `model`, `scale`, `imgsz`, `optimizer`, `status`, `epoch`, `mAP50-95`
   - Reset `#` numbering per bucket (start at 1).
3. Order of buckets:
   - Sort by `scale` then `imgsz` (recommended: `imgsz` desc).

## Step 7 — Comparison summary: HSG-DET vs YOLO (Requirement #4)

Create a section: `## Comparisons (HSG-DET vs YOLO)`

1. Filter rows into buckets by `(scale, imgsz)`.
2. For each bucket:
   - Identify **best HSG-DET** row (model_name starts with `HSG-DET`)
   - Identify **best YOLO** row (model_name contains `YOLO` or `yolov8`)
   - If a category is missing, write `—`.
3. Write a compact table:

| scale | imgsz | best HSG-DET (mAP50-95) | best YOLO (mAP50-95) | winner |

Winner rule:
- Compare `best_mAP50_95` if both exist.

## Step 8 — Mamba baseline summary (Requirement #5)

Create a section: `## Mamba-YOLO Baseline`

1. Filter rows where `model_name` contains `Mamba-YOLO`.
2. Sort by `best_mAP50_95` desc.
3. Provide:
   - Best overall Mamba run
   - Best per `(scale, imgsz)` if multiple exist
   - Show also any stopped/running attempts (so you can see failures and early signals)

Recommended table:

| # | machine | model | scale | imgsz | optimizer | status | epoch | mAP50-95 |

## Checklist

- [ ] All reachable machines included
- [ ] Unreachable machines listed in Data Sources
- [ ] Per-machine tables sorted by `mAP50-95`
- [ ] Combined table sorted by `mAP50-95`
- [ ] Grouped `(scale, imgsz)` tables exist and are sorted by `mAP50-95`
- [ ] Missing metrics shown as `—` and placed at bottom
- [ ] Comparison section includes `(scale, imgsz)` buckets present in data
- [ ] Mamba section includes all Mamba jobs, regardless of status
