
# ASGTransformer Multitask Training and Evaluation

## Overview

ASGTransformer is a transformer-based architecture for defensive cybersecurity scenario generation.

The supplied model contains four coupled components:

1. A causal language generator.
2. A semantic encoder.
3. A duration prediction head.
4. A scenario classification head.

The task is to complete, stabilize, and validate the training path so that all four objectives are optimized through the shared ASGTransformer representation.

This is a multitask training problem. Improvements to one objective must not be achieved by bypassing, replacing, or disabling another required objective.

## Objective

Starting from the supplied ASGTransformer checkpoint, implement a reliable joint training pipeline that produces a single trained Hugging Face-compatible checkpoint.

The final model must preserve the existing ASGTransformer architecture and inference interfaces while improving scenario classification and duration prediction without materially degrading semantic representation or language generation.

## Required Model Components

The following components must remain part of the trained ASGTransformer:

- Generator
- Semantic Encoder
- Duration Head
- Scenario Head

All required trainable components must participate in the real shared forward and backward optimization path.

Do not replace these components with independent external classifiers, synthetic predictions, hard-coded outputs, or evaluator-specific logic.

## Training Requirements

The implementation must:

- Start from the supplied checkpoint.
- Use the supplied training and validation data.
- Jointly optimize the required model objectives.
- Preserve the shared ASGTransformer representation.
- Compute and expose individual task losses.
- Maintain numerically stable training.
- Ensure required trainable components receive real gradient signals.
- Produce actual parameter updates for required trainable components.
- Support deterministic execution when deterministic mode is enabled.
- Keep training within the provided compute and runtime limits.

The required objectives are:

- causal language modeling;
- semantic representation learning;
- duration prediction;
- scenario classification.

Loss weighting, optimization strategy, learning-rate configuration, calibration, and checkpoint selection may be modified as necessary.

## Evaluation

The supplied initial checkpoint is the baseline.

The baseline and trained checkpoints must be evaluated using matched preprocessing, data partitions, metric implementations, and evaluation settings.

### Scenario Classification

Primary metric:

- F1

Secondary metric:

- Accuracy

### Duration Prediction

Primary metric:

- Mean Absolute Error (MAE)

Secondary metric:

- Root Mean Squared Error (RMSE)

### Semantic Representation

Evaluate semantic representation quality using the semantic metric supported by the supplied repository and evaluation data.

### Language Generation

Evaluate the generative objective using validation language-model loss together with functional generation checks.

The trained checkpoint must demonstrate measurable improvement in the required predictive objectives while preserving the semantic and generative objectives within the evaluation criteria.

Metric definitions, evaluation settings, thresholds, and allowable regression margins must be fixed before held-out evaluation and must not be selected based on hidden evaluator results.

## Gradient and Parameter Verification

The implementation must allow verification that the required trainable components participate in optimization.

The evaluator may verify:

- non-zero gradient signals;
- finite gradient values;
- parameter changes from initialization;
- individual objective losses;
- shared forward/backward participation.

A solution that bypasses a required component or produces artificial outputs without genuine model optimization does not satisfy the task.

## Checkpoint Requirements

The final result must be a single Hugging Face-compatible checkpoint containing the required ASGTransformer components.

The checkpoint must:

- save successfully;
- reload successfully;
- preserve the required model components;
- preserve trained parameters;
- remain compatible with the repository's normal loading path.

Saving and reloading the model must not materially change its evaluated behavior.

## Interface Compatibility

The existing repository interfaces must remain functional.

This includes the applicable interfaces for:

- `forward()`
- `generate()`
- `encode()`
- duration prediction
- scenario prediction
- `generate_scenario()`

Do not remove or bypass required public model functionality to improve evaluation metrics.

## Held-Out Evaluation

Evaluation may include held-out variations in:

- scenario-class distribution;
- duration distribution;
- sequence length;
- class balance;
- random seed;
- multitask example composition.

The implementation should therefore learn generalizable behavior rather than depend on fixed evaluation distributions or example-specific rules.

Hidden evaluator labels or results must not be used for:

- training;
- calibration;
- model selection;
- threshold selection;
- checkpoint selection.

## Offline Requirement

The complete task must operate without external network access during evaluation.

Use only the model assets, datasets, tokenizer resources, configuration files, and dependencies provided in the evaluation environment.

The solution must not depend on external APIs, hosted inference services, remote datasets, or network downloads.

## Compute Environment

The target evaluation environment provides:

- 1 NVIDIA L4 GPU
- 8 CPU cores
- 32 GiB host memory

The implementation must remain within the available resources.

Memory-efficient techniques such as mixed precision, bounded batch sizes, bounded sequence lengths, and gradient accumulation may be used while preserving genuine model training.

## Required Outputs

The completed implementation must produce sufficient information to evaluate:

- language-model loss;
- semantic objective performance;
- scenario classification performance;
- duration prediction performance;
- gradient participation;
- parameter updates;
- checkpoint integrity;
- checkpoint reloadability;
- required inference interfaces;
- runtime behavior.

The final trained model must be saved as a Hugging Face-compatible checkpoint.

## Constraints

The solution must not:

- replace required ASGTransformer heads with external classifiers;
- bypass the shared model representation;
- disable a required objective to improve another metric;
- hard-code evaluator-specific predictions;
- modify evaluator-controlled labels;
- use hidden evaluation labels;
- fabricate training or evaluation metrics;
- depend on external network access;
- remove required checkpoint components;
- break required inference interfaces.

Training must remain numerically valid. NaN or Infinity losses are considered invalid behavior.

## Success Criteria

A successful solution:

1. Executes genuine joint ASGTransformer training.
2. Preserves all four required objectives.
3. Produces real gradient and parameter updates.
4. Improves the required predictive objectives relative to the supplied baseline under the predefined evaluation criteria.
5. Preserves semantic representation and functional generation within the allowed evaluation margins.
6. Produces a complete reloadable Hugging Face-compatible checkpoint.
7. Preserves the required repository inference interfaces.
8. Operates completely offline.
9. Remains within the provided compute and runtime constraints.
10. Generalizes to the evaluator's held-out conditions.
