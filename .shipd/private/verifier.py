import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / ".shipd" / "artifacts"
REPORT_PATH = ARTIFACTS_DIR / "evaluation-report.json"


def fail(message: str, score: float = 0.0):
    result = {
        "status": "fail",
        "score": score,
        "message": message,
    }
    print(json.dumps(result, indent=2))
    sys.exit(1)


def require(condition: bool, message: str):
    if not condition:
        fail(message)


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def load_report():
    require(REPORT_PATH.exists(), f"Missing evaluation report: {REPORT_PATH}")

    try:
        with REPORT_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        fail(f"Unable to read evaluation report: {exc}")


def check_required_structure(report):
    required = [
        "baseline",
        "trained",
        "training",
        "checkpoint",
        "interfaces",
        "resources",
    ]

    for key in required:
        require(key in report, f"Missing report section: {key}")


def check_training(report):
    training = report["training"]

    require(
        training.get("completed_steps", 0) > 0,
        "No real training steps were completed.",
    )

    losses = training.get("losses", {})

    required_losses = [
        "language",
        "semantic",
        "duration",
        "scenario",
    ]

    for name in required_losses:
        require(name in losses, f"Missing {name} loss.")

        value = losses[name]

        require(
            finite(value),
            f"{name} loss is not finite.",
        )


def check_gradients(report):
    gradients = report["training"].get("gradients", {})

    required_components = [
        "generator",
        "semantic_encoder",
        "duration_head",
        "scenario_head",
    ]

    for component in required_components:
        require(
            component in gradients,
            f"Missing gradient statistics for {component}.",
        )

        grad = gradients[component]

        require(
            finite(grad),
            f"Gradient value for {component} is invalid.",
        )

        require(
            grad > 0,
            f"{component} did not receive a non-zero gradient signal.",
        )


def check_parameter_changes(report):
    changes = report["training"].get("parameter_changes", {})

    required_components = [
        "generator",
        "semantic_encoder",
        "duration_head",
        "scenario_head",
    ]

    for component in required_components:
        require(
            component in changes,
            f"Missing parameter-change evidence for {component}.",
        )

        delta = changes[component]

        require(
            finite(delta),
            f"Invalid parameter change for {component}.",
        )

        require(
            delta > 0,
            f"No trainable parameter change detected for {component}.",
        )


def check_metrics(report):
    baseline = report["baseline"]
    trained = report["trained"]

    required_metrics = [
        "classification_f1",
        "classification_accuracy",
        "duration_mae",
        "duration_rmse",
        "semantic_metric",
        "lm_loss",
    ]

    for metric in required_metrics:
        require(
            metric in baseline,
            f"Baseline is missing metric: {metric}",
        )

        require(
            metric in trained,
            f"Trained model is missing metric: {metric}",
        )

        require(
            finite(baseline[metric]),
            f"Invalid baseline value for {metric}",
        )

        require(
            finite(trained[metric]),
            f"Invalid trained value for {metric}",
        )


def check_primary_improvements(report):
    baseline = report["baseline"]
    trained = report["trained"]

    require(
        trained["classification_f1"] > baseline["classification_f1"],
        "Scenario classification F1 did not improve over baseline.",
    )

    require(
        trained["duration_mae"] < baseline["duration_mae"],
        "Duration MAE did not improve over baseline.",
    )


def check_objective_preservation(report):
    baseline = report["baseline"]
    trained = report["trained"]

    semantic_margin = report.get(
        "acceptance",
        {},
    ).get("semantic_regression_margin")

    lm_margin = report.get(
        "acceptance",
        {},
    ).get("lm_loss_regression_margin")

    if semantic_margin is not None:
        require(
            finite(semantic_margin) and semantic_margin >= 0,
            "Invalid semantic regression margin.",
        )

        minimum_semantic = baseline["semantic_metric"] - semantic_margin

        require(
            trained["semantic_metric"] >= minimum_semantic,
            "Semantic representation degraded beyond the allowed margin.",
        )

    if lm_margin is not None:
        require(
            finite(lm_margin) and lm_margin >= 0,
            "Invalid language-model regression margin.",
        )

        maximum_lm_loss = baseline["lm_loss"] + lm_margin

        require(
            trained["lm_loss"] <= maximum_lm_loss,
            "Language-model loss degraded beyond the allowed margin.",
        )


def check_checkpoint(report):
    checkpoint = report["checkpoint"]

    checkpoint_path = checkpoint.get("path")

    require(
        isinstance(checkpoint_path, str) and checkpoint_path,
        "Checkpoint path is missing.",
    )

    path = Path(checkpoint_path)

    if not path.is_absolute():
        path = ROOT / path

    require(
        path.exists(),
        f"Checkpoint does not exist: {path}",
    )

    require(
        checkpoint.get("save_success") is True,
        "Checkpoint save verification failed.",
    )

    require(
        checkpoint.get("reload_success") is True,
        "Checkpoint reload verification failed.",
    )

    components = checkpoint.get("components", [])

    required_components = {
        "generator",
        "semantic_encoder",
        "duration_head",
        "scenario_head",
    }

    require(
        required_components.issubset(set(components)),
        "Final checkpoint is missing required ASGTransformer components.",
    )


def check_interfaces(report):
    interfaces = report["interfaces"]

    required = [
        "forward",
        "generate",
        "encode",
        "duration_prediction",
        "scenario_prediction",
        "generate_scenario",
    ]

    for name in required:
        require(
            interfaces.get(name) is True,
            f"Required interface failed: {name}",
        )


def check_resources(report):
    resources = report["resources"]

    peak_host_gib = resources.get("peak_host_memory_gib")
    peak_gpu_gib = resources.get("peak_gpu_memory_gib")

    require(
        finite(peak_host_gib),
        "Missing or invalid host-memory measurement.",
    )

    require(
        peak_host_gib <= 32,
        f"Host-memory limit exceeded: {peak_host_gib:.2f} GiB > 32 GiB",
    )

    if peak_gpu_gib is not None:
        require(
            finite(peak_gpu_gib),
            "Invalid GPU-memory measurement.",
        )


def check_offline_and_determinism(report):
    runtime = report.get("runtime", {})

    require(
        runtime.get("network_used") is False,
        "Evaluation used external network access.",
    )

    if runtime.get("deterministic_mode") is True:
        require(
            runtime.get("deterministic_check_passed") is True,
            "Deterministic execution check failed.",
        )


def calculate_score(report):
    baseline = report["baseline"]
    trained = report["trained"]

    score = 0.0

    # 20 points: real joint training
    score += 20.0

    # 15 points: gradients and parameter participation
    score += 15.0

    # 15 points: classification improvement
    if trained["classification_f1"] > baseline["classification_f1"]:
        score += 15.0

    # 15 points: duration improvement
    if trained["duration_mae"] < baseline["duration_mae"]:
        score += 15.0

    # 10 points: semantic preservation
    score += 10.0

    # 10 points: generation preservation
    score += 10.0

    # 10 points: checkpoint and interfaces
    score += 10.0

    # 5 points: resource / offline / determinism integrity
    score += 5.0

    return min(score, 100.0)


def main():
    report = load_report()

    check_required_structure(report)
    check_training(report)
    check_gradients(report)
    check_parameter_changes(report)
    check_metrics(report)
    check_primary_improvements(report)
    check_objective_preservation(report)
    check_checkpoint(report)
    check_interfaces(report)
    check_resources(report)
    check_offline_and_determinism(report)

    score = calculate_score(report)

    result = {
        "status": "pass",
        "score": score,
        "message": "ASGTransformer evaluation completed successfully.",
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
