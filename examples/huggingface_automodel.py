from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "asgmodel/ASGTransformer"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)

user_text = (
    "Create an authorized defensive enterprise cybersecurity scenario "
    "focused on phishing awareness, credential protection, and response readiness."
)
prompt = model.build_grounded_prompt(user_text)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    semantic_embedding = model.encode(**inputs)
    duration_logits, duration_minutes = model.predict_duration(semantic_embedding)
    generated = model.generate(
        **inputs,
        max_new_tokens=384,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

sequences = generated.sequences if hasattr(generated, "sequences") else generated
print(tokenizer.decode(sequences[0], skip_special_tokens=True))
print("Estimated duration:", int(duration_minutes[0]), "minutes")
