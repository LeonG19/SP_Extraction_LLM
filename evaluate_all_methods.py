#!/usr/bin/env python3
"""
Comprehensive evaluation script for all baseline methods across multiple models.
Tests: PromptFuzz (fuzz), ReAct-Leak (re), PLeak (pleak), and LeakAgent (leakagent)
Models: Llama 3.1-8B, Llama 3.1-70B, Mistral-7B, GPT-OSS 20B, Qwen 3.5-27B

How this works:
    evaluate_task.py talks to an OpenAI-compatible server (via --server_url),
    not a local HF model. So the efficient pattern is:

      for each MODEL:
          start a vLLM server hosting that model        # loaded once
          for each METHOD:
              run evaluate_task.py against the server   # model stays hot
          stop the server and free the GPU
"""

import os
import sys
import time
import atexit
import socket
import signal
import logging
import argparse
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Iterable
from urllib.request import urlopen
from urllib.error import URLError

# ---------------------------------------------------------------------------
# Global process registry - ensures vLLM subprocesses are ALWAYS cleaned up,
# no matter how this script exits (Ctrl-C, SIGTERM, unhandled exception, etc).
# ---------------------------------------------------------------------------
_TRACKED_PROCS: List[subprocess.Popen] = []
_CLEANUP_LOCK = threading.Lock()
_CLEANUP_DONE = False


def _register_proc(proc: subprocess.Popen) -> None:
    """Track a child process so it gets killed on exit."""
    with _CLEANUP_LOCK:
        _TRACKED_PROCS.append(proc)


def _unregister_proc(proc: subprocess.Popen) -> None:
    """Stop tracking a process (after it has been cleanly stopped)."""
    with _CLEANUP_LOCK:
        try:
            _TRACKED_PROCS.remove(proc)
        except ValueError:
            pass


def _kill_proc_group(proc: subprocess.Popen, timeout: int = 15) -> None:
    """Kill a process group with SIGTERM then SIGKILL. Safe to call multiple times."""
    if proc.poll() is not None:
        return  # already dead
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    except (ProcessLookupError, PermissionError):
        pass  # process already gone or not ours


def _cleanup_all_procs() -> None:
    """
    Kill every tracked child process. Called from atexit and signal handlers.
    Idempotent: safe to invoke multiple times.
    """
    global _CLEANUP_DONE
    with _CLEANUP_LOCK:
        if _CLEANUP_DONE:
            return
        _CLEANUP_DONE = True
        procs = list(_TRACKED_PROCS)
        _TRACKED_PROCS.clear()

    if not procs:
        return

    # Use print() here because the logger may already be torn down during exit.
    print(f"\n[cleanup] Killing {len(procs)} tracked subprocess(es)...", flush=True)
    for proc in procs:
        if proc.poll() is None:
            print(f"[cleanup]   pid={proc.pid} (killing process group)", flush=True)
            _kill_proc_group(proc, timeout=10)
    print("[cleanup] Done.", flush=True)


def _signal_handler(signum, frame):
    """SIGINT/SIGTERM handler: clean up children, then exit with matching code."""
    print(f"\n[signal] Received signal {signum}, cleaning up subprocesses...",
          flush=True)
    _cleanup_all_procs()
    # Use the conventional exit code: 128 + signal number
    sys.exit(128 + signum)


# Register cleanup hooks as early as possible so they cover every code path.
atexit.register(_cleanup_all_procs)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGHUP, _signal_handler)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = Path("evaluation_logs")
RESULTS_DIR = Path("evaluation_results")
LOG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"evaluation_{TIMESTAMP}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Methods -> the folder their good_prompts.csv lives in.
# Each method's good_prompts.csv is at: <folder>/good_prompts.csv
METHODS: Dict[str, str] = {
    "fuzz":      "llama3_fuzz_finetune",
    "fuzz-ft":   "llama3_fuzz_finetune",
    "re":        "llama3_re_finetune",
    "leakagent": "llama3_rl_finetune",
    # "pleak":   <path>,   # uncomment and fill in if you have a prompts folder for PLeak
}

# Short name -> HF model id
MODELS: Dict[str, str] = {
    "llama3.1-8b":  "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1-70b": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    "mistral-7b":   "mistralai/Mistral-7B-Instruct-v0.3",
    "gpt-oss-20b":  "openai/gpt-oss-20b",
    "qwen3.5-27b":  "Qwen/Qwen3.5-27B",
}

# vLLM server settings
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1"
API_KEY = "EMPTY"                 # vLLM accepts any non-empty key
SERVER_STARTUP_TIMEOUT = 900      # 15 min - big models take a while to load
DATASET_PATH = "test_data_pleak.csv"
N_SAMPLES = 5


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

class EvaluationTracker:
    def __init__(self):
        self.total = 0
        self.successful = 0
        self.failed = 0
        self.skipped = 0
        self.results: List[Dict] = []

    def add_result(self, method: str, model: str, status: str, message: str = ""):
        self.total += 1
        label = {"success": "✓ SUCCESS", "failed": "✗ FAILED", "skipped": "⊘ SKIPPED"}[status]
        attr = {"success": "successful", "failed": "failed", "skipped": "skipped"}[status]
        setattr(self, attr, getattr(self, attr) + 1)
        self.results.append({"method": method, "model": model, "status": label, "message": message})

    def print_summary(self):
        logger.info("=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total experiments: {self.total}")
        logger.info(f"Successful: {self.successful} ✓")
        logger.info(f"Failed:     {self.failed} ✗")
        logger.info(f"Skipped:    {self.skipped} ⊘")
        if self.total > 0:
            logger.info(f"Success rate: {(self.successful / self.total * 100):.1f}%")
        logger.info("=" * 80)
        logger.info("\nDetailed Results:")
        logger.info("-" * 80)
        for r in self.results:
            logger.info(f"{r['method']:12} | {r['model']:15} | {r['status']:15} | {r['message']}")
        logger.info("-" * 80)


# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------

def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def _tail_file(path: Path, n: int = 5) -> List[str]:
    """Return the last n non-empty lines of a file, or [] if unavailable."""
    if not path.exists():
        return []
    try:
        with open(path, "r", errors="replace") as f:
            lines = [ln.rstrip() for ln in f.readlines() if ln.strip()]
        return lines[-n:]
    except Exception:
        return []


# Signatures we scan the vLLM log for, in order of lifecycle.
# Each entry: (substring to match, short stage label shown to user)
_VLLM_STAGE_MARKERS = [
    ("Downloading",                  "downloading weights from HuggingFace"),
    ("Fetching ",                    "downloading weights from HuggingFace"),
    ("resolve/main",                 "downloading weights from HuggingFace"),
    ("Loading safetensors checkpoint", "loading weights into memory"),
    ("Loading checkpoint shards",    "loading weights into memory"),
    ("Loading pt checkpoint",        "loading weights into memory"),
    ("model weights took",           "weights loaded - initializing engine"),
    ("Starting to load model",       "starting model load"),
    ("Model loading took",           "weights loaded - initializing engine"),
    ("Initializing a V1 LLM engine", "initializing vLLM engine"),
    ("Initializing an LLM engine",   "initializing vLLM engine"),
    ("Graph capturing",              "capturing CUDA graphs"),
    ("Memory profiling",             "profiling GPU memory"),
    ("Capturing the model",          "capturing CUDA graphs"),
    ("Available KV cache memory",    "allocating KV cache"),
    ("Application startup complete", "HTTP server starting"),
    ("Uvicorn running",              "server ready (not yet responsive)"),
]


def _detect_vllm_stage(log_path: Path) -> Optional[str]:
    """
    Scan the vLLM log and return the label for the most recent lifecycle stage
    we can identify. Returns None if nothing recognizable has been logged yet.
    """
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return None
    current_stage = None
    # We iterate the whole log so the *latest* matching stage wins.
    for line in text.splitlines():
        for needle, label in _VLLM_STAGE_MARKERS:
            if needle in line:
                current_stage = label
    return current_stage


def _gpu_status() -> Optional[str]:
    """Return a one-line GPU status via nvidia-smi, or None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            return None
        lines = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
        if not lines:
            return None
        parts = []
        for i, line in enumerate(lines):
            try:
                used, total, util = [x.strip() for x in line.split(",")]
                parts.append(f"GPU{i}: {used}/{total} MiB, {util}% util")
            except Exception:
                continue
        return " | ".join(parts) if parts else None
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return None


def _wait_for_server(
    url: str,
    timeout: int,
    proc: Optional[subprocess.Popen] = None,
    log_path: Optional[Path] = None,
) -> bool:
    """
    Poll /v1/models until the server answers. Prints detailed progress while
    waiting: current vLLM lifecycle stage, last lines of the vLLM log, and
    GPU memory usage. Bails early if the vLLM process dies.
    """
    health_url = url.rstrip("/") + "/models"
    start = time.time()
    last_heartbeat = start
    heartbeat_interval = 20  # seconds - more frequent than before
    last_stage_reported: Optional[str] = None

    logger.info("Startup in progress. Progress updates every 20s below.")

    while time.time() - start < timeout:
        # Process died?
        if proc is not None and proc.poll() is not None:
            elapsed = time.time() - start
            logger.error(
                f"vLLM process exited after {elapsed:.1f}s "
                f"with code {proc.returncode} (see log tail below)"
            )
            return False

        # Probe the HTTP endpoint
        try:
            with urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    elapsed = time.time() - start
                    logger.info(f"[OK] Server answered /v1/models after {elapsed:.1f}s")
                    return True
        except (URLError, ConnectionError, OSError):
            pass

        # If we detect a stage change, announce it immediately regardless
        # of the heartbeat timer, so progress feels responsive.
        if log_path is not None:
            stage = _detect_vllm_stage(log_path)
            if stage is not None and stage != last_stage_reported:
                elapsed = int(time.time() - start)
                logger.info(f"[{elapsed}s] Stage: {stage}")
                last_stage_reported = stage

        # Periodic heartbeat: stage + log tail + GPU
        now = time.time()
        if now - last_heartbeat >= heartbeat_interval:
            elapsed = int(now - start)
            logger.info(f"--- Status at {elapsed}s / {timeout}s ---")

            stage = _detect_vllm_stage(log_path) if log_path else None
            logger.info(f"  Stage: {stage if stage else '(no recognizable stage yet)'}")

            gpu = _gpu_status()
            if gpu:
                logger.info(f"  {gpu}")

            if log_path is not None:
                tail = _tail_file(log_path, n=5)
                if tail:
                    logger.info("  Last lines of vLLM log:")
                    for ln in tail:
                        # Truncate overly long lines so the console stays readable
                        logger.info(f"    | {ln[:200]}")
                else:
                    logger.info("  (vLLM log empty - process may not have started writing yet)")

            logger.info("  ---")
            last_heartbeat = now

        time.sleep(2)

    return False


def _dump_server_log_tail(log_path: Path, n_lines: int = 40):
    """
    Print the last n_lines of the vLLM log to the main evaluation log,
    so the real error is visible without opening a second file.
    """
    if not log_path.exists():
        logger.error(f"(no vLLM log at {log_path} - the process may not have started)")
        return
    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Could not read {log_path}: {e}")
        return

    if not lines:
        logger.error(f"(vLLM log at {log_path} is empty - the process may have been killed before writing)")
        return

    tail = lines[-n_lines:]
    logger.error("--- vLLM server log tail ---")
    for line in tail:
        logger.error(line.rstrip())
    logger.error("--- end of log tail ---")
    logger.error(f"(full log: {log_path})")


def _classify_startup_error(log_path: Path) -> str:
    """
    Scan the vLLM log for common failure signatures and return a short label.
    Used in the summary so skipped models are easy to scan.
    """
    if not log_path.exists():
        return "Process did not start (log missing)"
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return "Unknown error (log unreadable)"

    # Order matters: check more specific patterns first.
    signatures = [
        ("out of memory",                        "OOM loading model"),
        ("CUDA out of memory",                   "OOM loading model"),
        ("No space left on device",              "Disk full"),
        ("HfHubHTTPError",                       "HuggingFace Hub download failed"),
        ("401 Client Error",                     "HF auth error (gated model? login required)"),
        ("403 Client Error",                     "HF access denied (gated model?)"),
        ("404 Client Error",                     "Model id not found on HF"),
        ("GatedRepoError",                       "HF gated model - request access"),
        ("RepositoryNotFoundError",              "Model id not found on HF"),
        ("does not appear to have a file",       "Model config missing expected file"),
        ("ConnectionError",                      "Network/connection error"),
        ("ReadTimeoutError",                     "Network timeout downloading weights"),
        ("ImportError",                          "Missing Python dependency"),
        ("ModuleNotFoundError",                  "Missing Python dependency"),
        ("Address already in use",               f"Port {SERVER_PORT} already in use"),
        ("The model's max seq len",              "max-model-len exceeds model capacity"),
        ("unsupported dtype",                    "Dtype not supported by this GPU"),
        ("Traceback (most recent call last)",    "Python exception (see log tail)"),
    ]
    for needle, label in signatures:
        if needle in text:
            return label
    return "Startup failed (see log)"


def start_vllm_server(model_hf: str, log_path: Path, gpu_memory_utilization: float = 0.50) -> Optional[subprocess.Popen]:
    """
    Launch a vLLM OpenAI-compatible server hosting `model_hf`.
    Returns the Popen handle, or None on failure. On failure, prints the
    classified error and the tail of the vLLM log to the main log.
    """
    if _port_in_use(SERVER_HOST, SERVER_PORT):
        logger.error(f"Port {SERVER_PORT} is already in use - aborting")
        logger.error(
            "Something else is listening on this port. Find and kill it with:  "
            f"lsof -i :{SERVER_PORT}  (or)  fuser -k {SERVER_PORT}/tcp"
        )
        return None

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_hf,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--dtype", "bfloat16",
        "--trust-remote-code",
        # Single-H200 defaults. Adjust if you add more GPUs.
        "--tensor-parallel-size", "1",
        # GPU memory cap. This is a *fraction* of total GPU memory (144GB on
        # this server), not absolute. 0.50 -> ~72GB, which covers our largest
        # model (Qwen 27B in bf16: ~54GB weights + KV cache + overhead ~= 65GB).
        # The remaining ~72GB stays free for other users on the shared GPU.
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        # Cap context window: evaluate_task.py uses max_tokens=128 and short
        # system+user messages. 8192 leaves comfortable headroom for long
        # system prompts in test_data_pleak.csv and shrinks the KV cache a lot.
        "--max-model-len", "8192",
        # --- Startup-speed flags ---
        # CUDA graph capture adds 30-90s per startup for a marginal inference
        # speedup that doesn't matter for batch evaluation.
        "--enforce-eager",
        # Skip the startup profiling run that vLLM does to pick a KV block size.
        "--disable-log-stats",
    ]
    logger.info(f"Launching vLLM server: {' '.join(cmd)}")
    logger.info(f"vLLM server log: {log_path}")
    logger.info("(You can also watch it live in another terminal with:  "
                f"tail -f {log_path} )")

    # Force unbuffered output from the vLLM child, otherwise its logs land in
    # big delayed chunks and the stage-detection here lags by tens of seconds.
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,   # own process group, so we can kill children
        env=env,
    )
    # Register IMMEDIATELY so even an interrupt during startup cleans it up.
    _register_proc(proc)

    logger.info(f"Waiting up to {SERVER_STARTUP_TIMEOUT}s for server to be ready...")
    if not _wait_for_server(SERVER_URL, SERVER_STARTUP_TIMEOUT, proc, log_path):
        # Give the log a moment to flush any final traceback before we read it
        time.sleep(2)
        reason = _classify_startup_error(log_path)
        logger.error(f"vLLM server failed to start: {reason}")
        _dump_server_log_tail(log_path, n_lines=40)
        stop_vllm_server(proc)
        return None

    logger.info(f"vLLM server ready at {SERVER_URL}")
    return proc


def stop_vllm_server(proc: Optional[subprocess.Popen]):
    """Terminate the vLLM server, free the GPU, and remove it from the registry."""
    if proc is None:
        return
    logger.info("Stopping vLLM server...")
    _kill_proc_group(proc, timeout=30)
    _unregister_proc(proc)
    # Give the GPU a moment to actually free memory
    time.sleep(5)
    logger.info("vLLM server stopped")


# ---------------------------------------------------------------------------
# Running evaluate_task.py
# ---------------------------------------------------------------------------

def run_evaluation(
    method: str,
    model_name: str,
    model_hf: str,
    prompts_file: str,
    tracker: EvaluationTracker,
) -> bool:
    if not Path(prompts_file).exists():
        logger.warning(f"Prompts file not found: {prompts_file}")
        tracker.add_result(method, model_name, "skipped", "Results file not found")
        return False

    logger.info(f"  \u2192 {method.upper()} vs {model_name}")
    cmd = [
        "python", "evaluate_task.py",
        "--prompts_data_path", prompts_file,
        "--model_name",        model_hf,
        "--n_samples",         str(N_SAMPLES),
        "--dataset_path",      DATASET_PATH,
        "--server_url",        SERVER_URL,
        "--api_key",           API_KEY,
        "--disable_tqdm",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            logger.info(f"  \u2713 {method} vs {model_name}")
            tracker.add_result(method, model_name, "success")
            return True
        # Grab the tail of whatever output we got - Python scripts often print
        # tracebacks to stderr but some go to stdout.
        stderr_tail = (result.stderr or "").strip().splitlines()[-15:]
        stdout_tail = (result.stdout or "").strip().splitlines()[-5:]
        logger.error(f"  \u2717 {method} vs {model_name} (exit code {result.returncode})")
        if stderr_tail:
            logger.error("  --- evaluate_task.py stderr tail ---")
            for line in stderr_tail:
                logger.error(f"  {line}")
        if stdout_tail:
            logger.error("  --- evaluate_task.py stdout tail ---")
            for line in stdout_tail:
                logger.error(f"  {line}")
        # Short summary for the tracker table
        err_summary = (stderr_tail[-1] if stderr_tail else
                       stdout_tail[-1] if stdout_tail else
                       f"exit {result.returncode}")
        tracker.add_result(method, model_name, "failed", err_summary[:200])
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"  \u2717 {method} vs {model_name}: timeout")
        tracker.add_result(method, model_name, "failed", "Timeout (>1 hour)")
        return False
    except Exception as e:
        logger.error(f"  \u2717 {method} vs {model_name}: {e}")
        tracker.add_result(method, model_name, "failed", str(e)[:200])
        return False


# ---------------------------------------------------------------------------
# CLI / filtering
# ---------------------------------------------------------------------------

def _parse_csv_list(s: Optional[str]) -> List[str]:
    """Turn 'a,b, c' into ['a', 'b', 'c'] (lowercase, stripped, no empties)."""
    if not s:
        return []
    return [tok.strip().lower() for tok in s.split(",") if tok.strip()]


def _matches_any(needle: str, patterns: Iterable[str]) -> bool:
    """
    Case-insensitive substring match. A model is excluded if its short name
    OR its HF id contains any of the user-provided patterns. So --exclude-models
    accepts: short names ('llama3.1-70b'), HF ids ('meta-llama/Llama-3.1-70B-Instruct'),
    families ('llama', 'qwen'), sizes ('70b'), or providers ('openai').
    """
    needle = needle.lower()
    return any(p in needle for p in patterns)


def filter_models(models: Dict[str, str], include: List[str], exclude: List[str]) -> Dict[str, str]:
    if include:
        models = {
            name: hf for name, hf in models.items()
            if _matches_any(name, include) or _matches_any(hf, include)
        }
        logger.info(f"Included models (allowlist): {sorted(models)}")
    if exclude:
        kept = {
            name: hf for name, hf in models.items()
            if not (_matches_any(name, exclude) or _matches_any(hf, exclude))
        }
        dropped = set(models) - set(kept)
        if dropped:
            logger.info(f"Excluded models: {sorted(dropped)}")
        models = kept
    return models


def filter_methods(methods: Dict[str, str], include: List[str], exclude: List[str]) -> Dict[str, str]:
    if include:
        methods = {m: d for m, d in methods.items() if m.lower() in include}
        logger.info(f"Included methods (allowlist): {sorted(methods)}")
    if exclude:
        kept = {m: d for m, d in methods.items() if m.lower() not in exclude}
        dropped = set(methods) - set(kept)
        if dropped:
            logger.info(f"Excluded methods: {sorted(dropped)}")
        methods = kept
    return methods


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run baseline evaluations across methods and models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Allowlist examples (only run these):
  --models llama3.1-8b                     # run only the 8B model
  --models llama3.1-8b,mistral-7b          # run two specific models
  --models llama                           # run anything matching 'llama'
  --methods fuzz                           # run only the fuzz method
  --methods fuzz,re                        # run two methods

Exclusion examples (skip these):
  --exclude-models llama3.1-70b            # drop one specific model
  --exclude-models 70b                     # drop anything with '70b' in its name/id
  --exclude-models llama,qwen              # drop whole families
  --exclude-models openai/gpt-oss-20b      # drop by full HF id
  --exclude-methods leakagent              # skip a method
  --exclude-methods fuzz,re                # skip multiple methods

Note: --models / --methods are applied before --exclude-models / --exclude-methods.
""",
    )
    p.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated allowlist of models to run. Matches substring against "
             "both short names (e.g. 'llama3.1-8b') and HF ids. "
             "If omitted, all models are included.",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma-separated allowlist of methods to run. "
             "Choices: fuzz, re, pleak, leakagent. "
             "If omitted, all methods are included.",
    )
    p.add_argument(
        "--exclude-models",
        type=str,
        default="",
        help="Comma-separated list of models to skip. Matches substring against "
             "both short names (e.g. 'llama3.1-70b') and HF ids (e.g. 'Llama-3.1-70B'). "
             "Useful patterns: family ('llama'), size ('70b'), full id.",
    )
    p.add_argument(
        "--exclude-methods",
        type=str,
        default="",
        help="Comma-separated list of methods to skip. "
             "Choices: fuzz, re, pleak, leakagent.",
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.50,
        metavar="FRAC",
        help="Fraction of GPU memory vLLM may use (0.0–1.0). Default: 0.50.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print the resolved model/method plan and exit without running.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    include_models  = _parse_csv_list(args.models)
    include_methods = _parse_csv_list(args.methods)
    exclude_models  = _parse_csv_list(args.exclude_models)
    exclude_methods = _parse_csv_list(args.exclude_methods)

    models  = filter_models(MODELS,  include_models,  exclude_models)
    methods = filter_methods(METHODS, include_methods, exclude_methods)

    if not models:
        logger.error("No models remaining after exclusions - nothing to do.")
        sys.exit(1)
    if not methods:
        logger.error("No methods remaining after exclusions - nothing to do.")
        sys.exit(1)

    tracker = EvaluationTracker()

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE METHOD EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {TIMESTAMP}")
    logger.info(f"Log dir:   {LOG_DIR}")
    logger.info(f"Results:   {RESULTS_DIR}")
    logger.info(f"Server:    {SERVER_URL}")
    logger.info("")
    logger.info("Models (outer loop, each loaded once):")
    for n, hf in models.items():
        logger.info(f"  - {n:15} ({hf})")
    logger.info("Methods (inner loop, reusing the hot model):")
    for m, d in methods.items():
        logger.info(f"  - {m.upper():10} ({d})")
    logger.info("")

    if args.list:
        logger.info("--list given, exiting without running.")
        return

    total = len(models) * len(methods)
    exp = 0

    for model_name, model_hf in models.items():
        logger.info("=" * 80)
        logger.info(f"MODEL: {model_name} ({model_hf})")
        logger.info("=" * 80)

        server_log = LOG_DIR / f"vllm_{model_name}_{TIMESTAMP}.log"
        server = start_vllm_server(model_hf, server_log, args.gpu_memory_utilization)

        if server is None:
            reason = _classify_startup_error(server_log)
            logger.error(
                f"Could not start server for {model_name}: {reason}. "
                f"Marking all methods failed."
            )
            for method in methods:
                exp += 1
                logger.info(f"[{exp}/{total}]")
                tracker.add_result(method, model_name, "failed", f"Server: {reason}")
            continue

        try:
            for method, result_dir in methods.items():
                exp += 1
                logger.info(f"[{exp}/{total}]")
                prompts_file = str(Path(result_dir) / "good_prompts.csv")
                try:
                    run_evaluation(method, model_name, model_hf, prompts_file, tracker)
                except KeyboardInterrupt:
                    logger.warning("Interrupted by user")
                    stop_vllm_server(server)
                    tracker.print_summary()
                    sys.exit(0)
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    tracker.add_result(method, model_name, "failed", str(e)[:200])
        finally:
            stop_vllm_server(server)

    logger.info("")
    tracker.print_summary()
    logger.info(f"\nFull log: {log_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)