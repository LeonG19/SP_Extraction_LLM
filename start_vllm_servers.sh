#!/bin/bash

###############################################################################
# Start vLLM Servers with 2-GPU Configuration
#
# This script starts vLLM servers configured to use 2 GPUs with 0.95 memory
# utilization for maximum throughput and performance.
#
# Configuration:
# - GPU Memory Utilization: 0.95 (95%)
# - Tensor Parallel Size: 2 (use 2 GPUs)
# - CUDA_VISIBLE_DEVICES: 0,1 (GPUs 0 and 1)
###############################################################################

set -e

# GPU Configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_MEMORY_UTILIZATION=0.95
TENSOR_PARALLEL_SIZE=2

# Server configuration
PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-vllm_logs}"
mkdir -p "$LOG_DIR"

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}vLLM Server Startup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}GPU Configuration:${NC}"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  GPU Memory Utilization: $GPU_MEMORY_UTILIZATION (95%)"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo ""
echo -e "${GREEN}Server Configuration:${NC}"
echo "  Port: $PORT"
echo "  Server URL: http://localhost:$PORT/v1"
echo "  Log File: $LOG_DIR/vllm_server_$PORT.log"
echo ""

# Function to start server
start_server() {
    local model=$1
    local port=$2
    local log_file="$LOG_DIR/vllm_server_${port}.log"

    echo -e "${BLUE}Starting vLLM server for $model on port $port...${NC}"

    python -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --port "$port" \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --dtype auto \
        --max-model-len 4096 \
        >> "$log_file" 2>&1 &

    local pid=$!
    echo -e "${GREEN}Server started with PID: $pid${NC}"
    echo "$pid" > "$LOG_DIR/server_${port}.pid"

    # Wait for server to be ready
    echo -e "${YELLOW}Waiting for server to be ready...${NC}"
    for i in {1..60}; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo -e "${GREEN}Server is ready!${NC}"
            echo "  Model: $model"
            echo "  URL: http://localhost:$port/v1"
            echo ""
            return 0
        fi
        echo -n "."
        sleep 2
    done

    echo -e "${YELLOW}Timeout waiting for server${NC}"
    return 1
}

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [COMMAND] [MODEL]

Commands:
  start-8b              - Start Llama 3.1-8B on port 8000 (recommended for quick tests)
  start-70b             - Start Llama 3.1-70B on port 8000
  start-mistral         - Start Mistral-7B on port 8000
  start-qwen            - Start Qwen 3.5-27B on port 8000
  start-multi           - Start 8B on port 8000, 70B on port 8001 (requires GPU memory)
  stop-all              - Stop all running vLLM servers
  status                - Show running servers
  help                  - Show this message

Examples:
  # Start Llama 3.1-8B (recommended to start)
  $0 start-8b

  # Start and keep running in background
  nohup $0 start-8b > vllm.log 2>&1 &

  # Check server status
  curl http://localhost:8000/health

  # Kill server
  kill \$(cat vllm_logs/server_8000.pid)

Configuration:
  GPU_MEMORY_UTILIZATION: 0.95 (set to lower if OOM)
  TENSOR_PARALLEL_SIZE: 2 (use 2 GPUs)
  CUDA_VISIBLE_DEVICES: 0,1 (GPUs 0 and 1)

  To use different GPUs or settings:
  export CUDA_VISIBLE_DEVICES=0,2  # Use GPUs 0 and 2
  $0 start-8b

EOF
}

# Main execution
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}No command specified. Starting Llama 3.1-8B (default)...${NC}"
    echo ""
    start_server "meta-llama/Llama-3.1-8B-Instruct" 8000
    echo ""
    echo -e "${GREEN}To use this server with evaluations:${NC}"
    echo "  export OPENAI_BASE_URL=http://localhost:8000/v1"
    echo "  ./run_full_evaluation.sh base"
    exit 0
fi

case "$1" in
    start-8b)
        start_server "meta-llama/Llama-3.1-8B-Instruct" 8000
        ;;
    start-70b)
        echo -e "${YELLOW}Note: Llama 3.1-70B requires significant GPU memory${NC}"
        echo "      Make sure you have enough memory on both GPUs"
        echo ""
        start_server "meta-llama/Llama-3.1-70B-Instruct" 8000
        ;;
    start-mistral)
        start_server "mistralai/Mistral-7B-Instruct-v0.1" 8000
        ;;
    start-qwen)
        echo -e "${YELLOW}Note: Qwen 3.5-27B requires significant GPU memory${NC}"
        start_server "Qwen/Qwen3.5-27B" 8000
        ;;
    start-multi)
        echo -e "${YELLOW}Starting multiple servers (requires enough GPU memory)${NC}"
        echo ""
        start_server "meta-llama/Llama-3.1-8B-Instruct" 8000
        sleep 5
        echo ""
        start_server "meta-llama/Llama-3.1-70B-Instruct" 8001
        echo ""
        echo -e "${GREEN}Both servers are running:${NC}"
        echo "  8B:  http://localhost:8000/v1"
        echo "  70B: http://localhost:8001/v1"
        ;;
    stop-all)
        echo -e "${YELLOW}Stopping all vLLM servers...${NC}"
        for pid_file in "$LOG_DIR"/server_*.pid; do
            if [ -f "$pid_file" ]; then
                pid=$(cat "$pid_file")
                echo "Killing PID $pid..."
                kill "$pid" 2>/dev/null || true
                rm "$pid_file"
            fi
        done
        echo -e "${GREEN}All servers stopped${NC}"
        ;;
    status)
        echo -e "${BLUE}Running vLLM servers:${NC}"
        if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
            ps aux | grep "vllm.entrypoints.openai.api_server" | grep -v grep
        else
            echo "No vLLM servers running"
        fi
        echo ""
        echo -e "${BLUE}Server PIDs:${NC}"
        ls -la "$LOG_DIR"/server_*.pid 2>/dev/null || echo "No PID files found"
        ;;
    help|-h|--help)
        show_usage
        ;;
    *)
        echo -e "${YELLOW}Unknown command: $1${NC}"
        echo ""
        show_usage
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}For evaluation, run:${NC}"
echo "  export OPENAI_BASE_URL=http://localhost:8000/v1"
echo "  ./run_full_evaluation.sh base"
