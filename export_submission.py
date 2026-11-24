import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
TASKS = {f"{track}_{i:02d}" for track in (
    "reliability", "disease", "sleep", "emotion", "stimulus", "motor", "ssvep"
) for i in (1, 2)}
MODEL_FILES = ["model.py", "cbramod/cbramod.py", "cbramod/criss_cross_transformer.py", "cbramod/LICENSE"]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, default=ROOT.parent / "public_evaluation")
    args = parser.parse_args()

    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    selected = {}
    for task in sorted(TASKS):
        run = args.runs / task / str(args.seed)
        result = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        chosen = result["candidates"][result["C"]]
        if (result["task"]["dataset_name"] != task or result["seed"] != args.seed
                or result["config_sha256"] != config_hash):
            raise ValueError(f"{task}: saved run and supplied training configuration differ")
        weight = run / chosen["checkpoint"]
        if not weight.is_file():
            raise FileNotFoundError(weight)
        selected[task] = (weight, result)

    args.out.mkdir(parents=True, exist_ok=False)
    for name in ["predict.py", "README.md", "requirements.txt", "training/train.py", *MODEL_FILES]:
        dest = args.out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    shutil.copy2(args.config, args.out / "training/config.json")
    (args.out / "checkpoints").mkdir()
    tasks = {}
    for task, (weight, _) in selected.items():
        relative = f"checkpoints/{task}.pt"
        shutil.copy2(weight, args.out / relative)
        tasks[task] = {"checkpoint": relative}
    write_json(args.out / "model_config.json", {task: result for task, (_, result) in selected.items()})
    write_json(args.out / "manifest.json", {
        "manifest_version": "1.0", "predictor_api_version": "1.0", "entrypoint": "predict.py",
        "tasks": tasks, "artifacts": {
            "model_code": MODEL_FILES, "model_config": ["model_config.json"],
            "training_code": ["training/train.py"], "training_config": ["training/config.json"],
            "requirements": ["requirements.txt"], "documentation": ["README.md"],
        },
    })
    subprocess.run([sys.executable, str(args.evaluator / "refresh_submission_hashes.py"),
                    "--submission-dir", str(args.out)], check=True)


if __name__ == "__main__":
    main()
