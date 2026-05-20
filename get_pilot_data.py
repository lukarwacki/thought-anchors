from datasets import load_dataset
import json
import os

ds = load_dataset("uzaymacar/math-rollouts", streaming=True)

target_model = "deepseek-r1-distill-llama-8b"
base_dir = "/workspace/volume/thought-anchors/whitebox-analyses/math-rollouts"

saved = {}

for sample in ds["default"]:
    if target_model not in sample["path"].lower():
        continue
    if sample["filename"] not in ["base_solution.json", "chunks.json"]:
        continue
    if "chunk_" in sample["path"]:
        continue

    parts = sample["path"].split("/")
    problem_id = next((p for p in parts if p.startswith("problem_")), None)
    condition = next((p for p in parts if p in ["correct_base_solution", "incorrect_base_solution"]), None)
    if not problem_id or not condition:
        continue

    # for base_solution.json verify full_cot exists
    data = json.loads(sample["content"])
    if sample["filename"] == "base_solution.json" and "full_cot" not in data:
        continue

    out_dir = f"{base_dir}/{target_model}/temperature_0.6_top_p_0.95/{condition}/{problem_id}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{sample['filename']}"

    # skip if already saved
    if os.path.exists(out_path):
        continue

    with open(out_path, "w") as f:
        json.dump(data, f)

    if problem_id not in saved:
        saved[problem_id] = []
    saved[problem_id].append(f"{condition}/{sample['filename']}")
    print(f"Saved {problem_id}/{condition}/{sample['filename']}")

    complete = sum(
        1 for p in saved.values()
        if any("correct_base_solution/base_solution.json" in x for x in p)
        and any("correct_base_solution/chunks.json" in x for x in p)
        and any("incorrect_base_solution/base_solution.json" in x for x in p)
        and any("incorrect_base_solution/chunks.json" in x for x in p)
    )
    if complete >= 20:
        break

print(f"\nDone. Complete problems: {len(saved)}")