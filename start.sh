#!/bin/bash

# LLM Council - Start script

echo "Starting LLM Council..."
echo ""

# Start backend
echo "Starting backend on http://localhost:8001..."

# Force uv to use public PyPI instead of corp index that needs auth
export UV_NO_CONFIG=1
export UV_DEFAULT_INDEX="https://pypi.org/simple"
export UV_INDEX_URL="https://pypi.org/simple"
export UV_EXTRA_INDEX_URL=""

# Respect an already-activated virtualenv so uv doesn't try (and fail) to create .venv
UV_RUN_CMD=("uv" "run" "--index-url" "https://pypi.org/simple")
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "Detected active virtualenv at ${VIRTUAL_ENV}; using it."
  UV_RUN_CMD+=("--active")
fi

"${UV_RUN_CMD[@]}" python -m backend.main &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 2

# Start frontend
echo "Starting frontend on http://localhost:5173..."
cd frontend
if [[ ! -d node_modules ]]; then
  echo "Installing frontend dependencies..."
  npm install
fi
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✓ LLM Council is running!"
echo "  Backend:  http://localhost:8001"
echo "  Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
