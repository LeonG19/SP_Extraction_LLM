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
import socket
import signal
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlopen
from urllib.error import URLError

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

# Methods -> the folder their good_prompts.csv lives in
METHODS: Dict[str, str] = {
    "fuzz":      "fuzz_results",
    "re":        "re_results",
    "pleak":     "pleak_results",
    "leakagent": "leakagent_results",
}

# Short name -> HF model id
MODELS: Dict[str, str] = {
    "llama3.1-8b":  "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
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


def _wait_for_server(url: str, timeout: int) -> bool:
    """Poll /v1/models until the server is answering."""
    health_url = url.rstrip("/") + "/models"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urlopen(health_url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (URLError, ConnectionError, OSError):
            pass
        time.sleep(5)
    return False


def start_vllm_server(model_hf: str, log_path: Path) -> Optional[subprocess.Popen]:
    """
    Launch a vLLM OpenAI-compatible server hosting `model_hf`.
    Returns the Popen handle, or None on failure.
    """
    if _port_in_use(SERVER_HOST, SERVER_PORT):
        logger.error(f"Port {SERVER_PORT} is already in use - aborting")
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
        "--gpu-memory-utilization", "0.90",
        # Cap context window: evaluate_task.py uses max_tokens=128 and short
        # system+user messages. 8192 leaves comfortable headroom for long
        # system prompts in test_data_pleak.csv and shrinks the KV cache a lot.
        "--max-model-len", "8192",
    ]
    logger.info(f"Launching vLLM server: {' '.join(cmd)}")
    logger.info(f"Server log: {log_path}")

    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,   # own process group, so we can kill children
    )

    logger.info(f"Waiting up to {SERVER_STARTUP_TIMEOUT}s for server to be ready...")
    if not _wait_for_server(SERVER_URL, SERVER_STARTUP_TIMEOUT):
        logger.error("Server did not become ready in time")
        stop_vllm_server(proc)
        return None

    logger.info(f"vLLM server ready at {SERVER_URL}")
    return proc


def stop_vllm_server(proc: Optional[subprocess.Popen]):
    """Terminate the vLLM server and free the GPU."""
    if proc is None:
        return
    logger.info("Stopping vLLM server...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not exit in 30s - sending SIGKILL")
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
    except ProcessLookupError:
        pass
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
        err = (result.stderr or "Unknown error")[:200]
        logger.error(f"  \u2717 {method} vs {model_name}: {err}")
        tracker.add_result(method, model_name, "failed", err)
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
# Main
# ---------------------------------------------------------------------------

def main():
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
    for n, hf in MODELS.items():
        logger.info(f"  - {n:15} ({hf})")
    logger.info("Methods (inner loop, reusing the hot model):")
    for m, d in METHODS.items():
        logger.info(f"  - {m.upper():10} ({d})")
    logger.info("")

    total = len(MODELS) * len(METHODS)
    exp = 0

    for model_name, model_hf in MODELS.items():
        logger.info("=" * 80)
        logger.info(f"MODEL: {model_name} ({model_hf})")
        logger.info("=" * 80)

        server_log = LOG_DIR / f"vllm_{model_name}_{TIMESTAMP}.log"
        server = start_vllm_server(model_hf, server_log)

        if server is None:
            logger.error(f"Could not start server for {model_name} - marking all methods failed")
            for method in METHODS:
                exp += 1
                logger.info(f"[{exp}/{total}]")
                tracker.add_result(method, model_name, "failed", "Server failed to start")
            continue

        try:
            for method, result_dir in METHODS.items():
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
