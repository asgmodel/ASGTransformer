export default {
  spec: {
    profile: {
      resources: {
        cpu: 8,
        memory: "32GiB",
        gpu: {
          type: "L4",
          count: 1
        }
      },

      evaluation: {
        run: [
          "python3",
          ".shipd/private/verifier.py"
        ]
      }
    },

    rubric: [
      {
        id: "joint_multitask_training",
        title: "Joint multitask training",
        weight: 20,
        description:
          "All four ASGTransformer objectives must participate in a real shared forward and backward optimization path."
      },
      {
        id: "gradient_parameter_updates",
        title: "Gradient and parameter participation",
        weight: 15,
        description:
          "Required trainable components receive finite non-zero gradients and their parameters change from the supplied initialization."
      },
      {
        id: "scenario_classification",
        title: "Scenario classification",
        weight: 15,
        description:
          "Evaluate scenario classification primarily using F1 and secondarily using accuracy against the supplied baseline."
      },
      {
        id: "duration_prediction",
        title: "Duration prediction",
        weight: 15,
        description:
          "Evaluate duration prediction primarily using MAE and secondarily using RMSE against the supplied baseline."
      },
      {
        id: "semantic_preservation",
        title: "Semantic representation",
        weight: 10,
        description:
          "Preserve semantic representation quality within the predefined evaluation margin."
      },
      {
        id: "generation_preservation",
        title: "Generation quality",
        weight: 10,
        description:
          "Preserve validation language-model behavior and functional generation while optimizing the predictive objectives."
      },
      {
        id: "checkpoint_interfaces",
        title: "Checkpoint and interfaces",
        weight: 10,
        description:
          "The final Hugging Face-compatible checkpoint must save, reload, and preserve required ASGTransformer inference interfaces."
      },
      {
        id: "runtime_integrity",
        title: "Runtime and integrity",
        weight: 5,
        description:
          "The implementation remains offline, deterministic when requested, numerically valid, and within the configured runtime and resource limits."
      }
    ]
  }
};
