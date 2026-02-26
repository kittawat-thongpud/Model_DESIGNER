try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, Vertical
    from textual.widgets import Header, Footer, Static, ListView, ListItem, RichLog, Label
    from textual.binding import Binding
    from textual import work
    import asyncio
except ImportError:
    print("❌ 'textual' library is required for the TUI.")
    print("   Please run: pip install textual")
    exit(1)

import os
import signal
import sys
import threading
import subprocess
import time
import atexit
from pathlib import Path

# Config
PROJECT_ROOT = Path(__file__).parent.resolve()
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Global reference for cleanup handler
APP_INSTANCE = None

def cleanup_processes():
    """Force kill all child processes in the process group."""
    try:
        # Kill the entire process group
        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
    except Exception:
        pass

def signal_handler(sig, frame):
    """Handle Ctrl+C / SIGTERM at module level."""
    cleanup_processes()
    sys.exit(0)

# Register cleanup hooks
atexit.register(cleanup_processes)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class ModelDesignerApp(App):
    TITLE = "Model DESIGNER TUI"
    CSS = """
    Screen {
        layout: vertical;
    }
    
    #sidebar {
        width: 25%;
        height: 100%;
        background: $panel;
        border-right: vkey $accent;
        dock: left;
    }
    
    #content-area {
        width: 75%;
        height: 100%;
    }

    RichLog {
        height: 100%;
        background: $surface;
        color: $text;
        border: solid $secondary;
        overflow-y: scroll;
    }
    
    .hidden {
        display: none;
    }
    
    Label {
        padding: 1;
        width: 100%;
    }
    
    ListItem {
        padding: 1 2;
        background: $panel;
    }
    
    ListItem:hover {
        background: $block-hover-background;
    }
    
    ListItem.--highlight {
        background: $accent;
        color: $text;
    }
    """
    
    BINDINGS = [
        Binding("q", "action_quit", "Quit"),
        Binding("ctrl+c", "action_quit", "Quit"),
        Binding("c", "clear_log", "Clear Log"),
    ]

    def __init__(self):
        super().__init__()
        global APP_INSTANCE
        APP_INSTANCE = self
        
        self.processes = {}
        self.should_stop = threading.Event()
        self.log_buffers = {"backend": [], "frontend": []}
        self.buffer_lock = threading.Lock()
        self.shutting_down = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label("🚀 Services", classes="header")
                self.list_view = ListView(
                    ListItem(Label("Backend (FastAPI)"), id="item_backend"),
                    ListItem(Label("Frontend (Vite)"), id="item_frontend"),
                    initial_index=0
                )
                yield self.list_view
                
                yield Label("ℹ️ Guide", classes="header")
                yield Label("• Click to select session\n• 'c': Clear Log\n• 'q' or Ctrl+C: Quit", classes="guide")
                
            with Container(id="content-area"):
                yield RichLog(id="log_backend", markup=True, wrap=True)
                yield RichLog(id="log_frontend", markup=True, wrap=True, classes="hidden")
        yield Footer()

    async def on_mount(self) -> None:
        self.logs = {
            "backend": self.query_one("#log_backend", RichLog),
            "frontend": self.query_one("#log_frontend", RichLog)
        }
        self.logs["backend"].write("[bold green]Starting services...[/]")
        
        self.set_interval(0.1, self.flush_logs)
        
        self.run_process("backend", self.start_backend_cmd)
        self.run_process("frontend", self.start_frontend_cmd)

    def flush_logs(self):
        if self.shutting_down: return
        for name in ["backend", "frontend"]:
            with self.buffer_lock:
                if self.log_buffers[name]:
                    text = "\n".join(self.log_buffers[name])
                    self.log_buffers[name].clear()
                    try:
                        self.logs[name].write(text)
                    except: pass

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        view_name = event.item.id.replace("item_", "")
        self.switch_to_view(view_name)

    def switch_to_view(self, view_name: str):
        self.current_view = view_name
        self.query("#log_backend").add_class("hidden")
        self.query("#log_frontend").add_class("hidden")
        self.query(f"#log_{view_name}").remove_class("hidden")
        self.title = f"Model DESIGNER - {view_name.capitalize()}"

    def action_clear_log(self):
        if "hidden" not in self.query_one("#log_backend").classes:
            self.logs["backend"].clear()
        elif "hidden" not in self.query_one("#log_frontend").classes:
            self.logs["frontend"].clear()

    def action_quit(self):
        """Explicit quit action."""
        self.exit()

    @work(thread=True)
    def run_process(self, name, cmd_builder):
        cmd, cwd = cmd_builder()
        
        with self.buffer_lock:
            self.log_buffers[name].append(f"[blue]Starting {name}...[/]")
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid
            )
            self.processes[name] = process
            
            for line in process.stdout:
                if self.should_stop.is_set():
                    break
                stripped = line.rstrip()
                if stripped:
                    with self.buffer_lock:
                        self.log_buffers[name].append(stripped)
            process.wait()
                    
        except Exception as e:
            with self.buffer_lock:
                self.log_buffers[name].append(f"[bold red]Error: {e}[/]")

    def start_backend_cmd(self):
        venv_python = PROJECT_ROOT / "backend" / "venv" / "bin" / "python"
        if not venv_python.exists():
             venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
             
        if venv_python.exists():
            python_exec = str(venv_python)
        else:
            python_exec = sys.executable

        cmd = [
            python_exec, "-m", "uvicorn", "app.main:app", 
            "--reload", "--host", "0.0.0.0", "--port", "8000",
            "--no-access-log"
        ]
        return cmd, BACKEND_DIR

    def start_frontend_cmd(self):
        cmd = ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
        return cmd, FRONTEND_DIR

    def on_unmount(self):
        self.should_stop.set()
        cleanup_processes()

if __name__ == "__main__":
    app = ModelDesignerApp()
    app.run()