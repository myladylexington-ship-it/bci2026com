import json
from pathlib import Path

import torch

from model import TaskHead, Window, encode, new_backbone, normalized_inputs


def load_model(model_dir: str, task_info: dict, device: str):
    directory = Path(model_dir)
    name = task_info["dataset_name"]
    manifest = json.loads((directory.parent / "manifest.json").read_text(encoding="utf-8"))
    path = directory.parent / manifest["tasks"][name]["checkpoint"]
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    task = checkpoint["task"]
    for key in ("dataset_name", "label_values", "channels", "timepoints"):
        if task[key] != task_info[key]:
            raise ValueError(f"{name}: checkpoint {key}={task[key]!r}, evaluator {task_info[key]!r}")
    backbone = new_backbone()
    backbone.load_state_dict(checkpoint["backbone"])
    head = TaskHead(task, checkpoint["path"])
    head.load_state_dict(checkpoint["head"])
    return (
        Window(task).to(device), backbone.to(device).eval(), head.to(device).eval(),
        checkpoint["stats"], torch.device(device),
    )


@torch.inference_mode()
def predict(model, batch: dict):
    window, backbone, head, stats, device = model
    eeg = torch.as_tensor(batch["eeg"], dtype=torch.float32)
    early = head.scale_projection if head.path == "E" else None
    scores = []
    # The evaluator controls the outer batch size.
    for chunk in eeg.split(4):
        mapped, rho, scale, marks = normalized_inputs(window(chunk.to(device)), stats, raw=head.path == "R")
        h = encode(backbone, mapped, rho, scale, early)
        scores.append(head(h, rho, scale, marks).float().cpu())
    return torch.cat(scores).numpy()
