#!/usr/bin/env bash
set -euo pipefail
uvicorn asg_transformer.api.main:app --host 0.0.0.0 --port "${PORT:-8000}" --reload
