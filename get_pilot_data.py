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
    if sample["filename"] != "base_solution.json":
        continue
    # only correct and incorrect base solutions, not chunk subfolders
    if "chunk_" in sample["path"]:
        continue

    parts = sample["path"].split("/")
    problem_id = next((p for p in parts if p.startswith("problem_")), None)
    condition = next((p for p in parts if p in ["correct_base_solution", "incorrect_base_solution"]), None)
    if not problem_id or not condition:
        continue

    # verify it has full_cot
    data = json.loads(sample["content"])
    if "full_cot" not in data:
        continue

    # save to exact path the code expects
    out_dir = f"{base_dir}/{target_model}/temperature_0.6_top_p_0.95/{condition}/{problem_id}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/base_solution.json", "w") as f:
        json.dump(data, f)

    if problem_id not in saved:
        saved[problem_id] = []
    saved[problem_id].append(condition)
    print(f"Saved {problem_id}/{condition}")

    # stop after 5 complete problems (both conditions)
    complete = sum(
        1 for p in saved.values()
        if "correct_base_solution" in p and "incorrect_base_solution" in p
    )
    if complete >= 5:
        break

print(f"\nDone. {len(saved)} problems saved.")
for pid, conditions in saved.items():
    print(f"  {pid}: {conditions}")