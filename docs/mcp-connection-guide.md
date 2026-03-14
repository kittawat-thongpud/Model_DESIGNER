# Model DESIGNER — MCP Connection Guide

MCP interface ถูก mount ไว้ใน backend server เดิม ไม่ต้องรัน process แยก

---

## Endpoints

| Transport | URL |
|---|---|
| SSE stream (เชื่อมต่อ) | `http://localhost:8000/mcp/sse` |
| Message (JSON-RPC) | `http://localhost:8000/mcp/messages/` |

> Port 8000 คือ default ของ backend (`run.py`) — เปลี่ยนได้ถ้าสั่ง `--port` อื่น
>
> หมายเหตุ: ตัว MCP app ถูก mount ใต้ `/mcp` ใน FastAPI แต่ภายใน FastMCP ใช้ root mount path (`/`) เพื่อหลีกเลี่ยง bug path ซ้ำแบบ `/mcp/mcp/messages/`

---

## วิธีตรวจสอบว่า MCP ทำงานอยู่

```bash
# 1. ตรวจว่า backend รันอยู่
curl http://localhost:8000/api/health

# 2. ตรวจ SSE endpoint (ต้องได้ text/event-stream กลับมา)
curl -N -H "Accept: text/event-stream" http://localhost:8000/mcp/sse
```

ถ้า SSE ทำงานอยู่จะเห็น:
```
event: endpoint
data: /mcp/messages/?session_id=...
```

---

## เชื่อมต่อกับ Claude Desktop

แก้ไขไฟล์ config ของ Claude Desktop:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`  
**Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "model-designer": {
      "url": "http://localhost:8000/mcp/sse"
    }
  }
}
```

รีสตาร์ท Claude Desktop หลังแก้ config

---

## เชื่อมต่อกับ Cursor

ไปที่ **Settings → MCP** หรือแก้ไขไฟล์ `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "model-designer": {
      "url": "http://localhost:8000/mcp/sse"
    }
  }
}
```

---

## เชื่อมต่อกับ Windsurf

แก้ไขไฟล์ `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "model-designer": {
      "serverUrl": "http://localhost:8000/mcp/sse"
    }
  }
}
```

---

## ทดสอบ/Debug

### วิธีที่ 1 — curl ตรวจ SSE endpoint

```bash
# ตรวจว่า MCP พร้อมรับ connection (Ctrl+C เพื่อหยุด)
curl -N -H "Accept: text/event-stream" http://localhost:8000/mcp/sse
```

ถ้าทำงานปกติจะเห็น:
```
event: endpoint
data: /mcp/messages/?session_id=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

### วิธีที่ 2 — MCP Inspector (browser UI)

```bash
# ต้องมี Node.js
npx @modelcontextprotocol/inspector http://localhost:8000/mcp/sse
```

เปิด browser → `http://localhost:5173` → เรียก tools แบบ interactive

### วิธีที่ 3 — Python client test

```python
import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession

async def test():
    async with sse_client("http://localhost:8000/mcp/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # list tools
            tools = await session.list_tools()
            print(f"Tools: {len(tools.tools)}")
            for t in tools.tools[:5]:
                print(f"  - {t.name}")

            # call a tool
            result = await session.call_tool("list_models", {"view": "summary"})
            print(f"Result: {result.content[0].text[:200]}")

asyncio.run(test())
```

บันทึกเป็น `test_mcp_client.py` แล้วรัน:
```bash
../venv/bin/python test_mcp_client.py
```

---

## Tools ที่ใช้ได้ (30 tools)

### Models
| Tool | หน้าที่ |
|---|---|
| `list_models` | ดูรายการ model ทั้งหมด |
| `get_model` | ดู record ของ model ตาม ID |
| `get_model_yaml` | ดู YAML definition ดิบ |
| `create_model` | สร้าง model ใหม่ |
| `validate_model` | validate YAML + นับ params/layers |

### Datasets
| Tool | หน้าที่ |
|---|---|
| `list_datasets` | ดูรายการ dataset |
| `get_dataset` | ดูข้อมูล dataset ตามชื่อ |
| `preview_dataset` | ดู sample labels (ไม่ส่ง image binary) |

### Training Jobs
| Tool | หน้าที่ |
|---|---|
| `list_jobs` | ดูรายการ training jobs (filter by status/model) |
| `get_job` | ดู record ของ job |
| `get_job_logs` | ดู training logs (paginated) |
| `get_job_metrics` | ดู system metrics (GPU/CPU/RAM) |
| `get_job_history` | ดู epoch-by-epoch history |

### Weights
| Tool | หน้าที่ |
|---|---|
| `list_weights` | ดูรายการ weights |
| `get_weight` | ดู metadata ของ weight |
| `get_weight_info` | ดู params/GFLOPs จากไฟล์ .pt |
| `get_weight_lineage` | ดู training lineage chain |
| `create_empty_weight` | สร้าง weight จาก model architecture |
| `list_pretrained_weights` | ดู pretrained catalog |
| `download_pretrained_weight` | ดาวน์โหลด pretrained model |

### Benchmark
| Tool | หน้าที่ |
|---|---|
| `list_benchmark_datasets` | ดู datasets ที่ benchmark ได้ |
| `run_benchmark` | รัน validation benchmark |
| `list_benchmarks` | ดูประวัติ benchmark |
| `get_benchmark` | ดูผล benchmark แบบละเอียด |

### Training Control
| Tool | หน้าที่ |
|---|---|
| `start_training` | เริ่ม training job |
| `stop_training` | หยุด training |
| `resume_training` | ต่อ training จาก checkpoint |
| `append_training` | เพิ่ม epochs ให้ job ที่เสร็จแล้ว |
| `get_training_queue` | ดูสถานะ queue |
| `get_training_workers_health` | ดูสถานะ worker threads |

---

## Resources URI (อ่าน-อย่างเดียว)

MCP resources เข้าถึงได้ผ่าน URI scheme:

```
models://list
models://{model_id}
models://{model_id}/yaml
datasets://list
datasets://{name}
jobs://list
jobs://{job_id}
jobs://{job_id}/logs
jobs://{job_id}/metrics
jobs://{job_id}/history
weights://list
weights://{weight_id}
weights://{weight_id}/lineage
benchmarks://list
benchmarks://{benchmark_id}
training://queue
training://workers/health
```

---

## Summary vs Detail

ทุก list/get tool รองรับ `view` parameter:

```
view="summary"  (default) — ส่งเฉพาะ field สำคัญ ลด tokens
view="detail"   — ส่งข้อมูลเต็มเหมือน REST API
```

ตัวอย่าง:
```
list_jobs(status="running", view="summary", limit=10)
get_job(job_id="abc123", view="detail", include_history=True)
get_job_logs(job_id="abc123", limit=20, level="ERROR")
```

---

## ตัวอย่าง workflow สำหรับ agent

```
1. list_models()                          → เลือก model_id
2. list_datasets()                        → เลือก dataset name
3. start_training(model_id, config={      → ได้ job_id
       "data": "my_dataset",
       "epochs": 100,
       "batch": 16
   })
4. get_job(job_id)                        → ดู status
5. get_job_logs(job_id, limit=20)         → ดู logs ล่าสุด
6. list_weights(model_id=model_id)        → หา weight ที่ได้
7. run_benchmark(weight_id, dataset)      → ทดสอบผล
8. get_benchmark(benchmark_id)            → ดูผล mAP/precision/recall
```

---

## Smoke test

```bash
cd /home/rase/kittawat_ws/Model_DESIGNER/backend
../venv/bin/python tests/test_mcp_smoke.py
# Expected: 39/39 passed ✓
```
