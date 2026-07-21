from asg_transformer import ASGUnifiedModel

model = ASGUnifiedModel.from_pretrained("asgmodel/ASG-Unified-Scenario-Model")
result = model.generate(
    "Build a defensive enterprise training scenario focused on phishing, credential access, and lateral movement.",
    max_steps=6,
    total_duration_minutes=240,
    language="en",
)
print(result.generated_text)
