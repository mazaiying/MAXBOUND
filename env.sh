#!/usr/bin/env bash
# Example environment. Adapt CUDA_HOME and the Python environment to your machine.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAXBOUND_HOME="${MAXBOUND_HOME:-$HOME/maxbound}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-48}"
export PYTHONPATH="$MAXBOUND_HOME:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
