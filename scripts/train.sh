#!/usr/bin/env bash
set -euo pipefail
python -m asg_transformer.training.train_encoder --epochs "${EPOCHS:-10}" --batch-size "${BATCH_SIZE:-16}"
