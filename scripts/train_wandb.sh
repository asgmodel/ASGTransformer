#!/usr/bin/env bash
set -euo pipefail

python -m asg_transformer.training.train_encoder \
  --wandb \
  --wandb-project "${WANDB_PROJECT:-asg-transformer}" \
  --wandb-entity "${WANDB_ENTITY:-}" \
  --wandb-mode "${WANDB_MODE:-online}" \
  --epochs "${ASG_EPOCHS:-10}" \
  --batch-size "${ASG_BATCH_SIZE:-16}"
