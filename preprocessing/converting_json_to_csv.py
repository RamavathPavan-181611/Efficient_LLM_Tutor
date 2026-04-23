import json
import csv

# Load JSON file
with open("/Users/ramavathramesh/Desktop/Projects/RL_Project/data/CoMTA_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

rows = []

for item in dataset:
    test_id = item.get("test_id")
    math_level = item.get("math_level")
    expected_result = item.get("expected_result")

    for turn_id, message in enumerate(item.get("data", []), start=1):
        rows.append({
            "test_id": test_id,
            "math_level": math_level,
            "expected_result": expected_result,
            "turn_id": turn_id,
            "role": message.get("role"),
            "content": message.get("content")
        })

# Save CSV
with open("CoMTA_dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "test_id",
            "math_level",
            "expected_result",
            "turn_id",
            "role",
            "content"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print("CSV file created successfully!")