from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "asgmodel/ASGTransformer"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )

    result = model.generate_scenario(
        tokenizer,
        (
            "Create an authorized defensive enterprise scenario focused on "
            "phishing awareness, credential protection, and response readiness."
        ),
        language="en",
        max_new_tokens=384,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    print(result["text"])
    print("Estimated duration:", result["estimated_duration_minutes"], "minutes")
    print("Scenario type:", result["scenario_type"])


if __name__ == "__main__":
    main()
