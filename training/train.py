import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import lance
import numpy as np
import pyarrow.compute as pc
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import C_ORDER, SCALE_PATHS, TaskHead, Window, encode, load_public_backbone, normalized_inputs


def batches(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


class Fold:
    def __init__(self, config_dir, task, split, profile):
        self.dataset = lance.dataset(str(config_dir / task["lance"]), version=task["lance_version"])
        self.shape = (task["channels"], task["timepoints"])
        self.labels = {label: col for col, label in enumerate(task["label_values"])}
        self.rows = {
            part: np.load(config_dir / split[f"{part}_rows"], allow_pickle=False)
            for part in (("train",) if profile else ("train", "val"))
        }
        self.class_counts = {}
        subjects = {}
        for part, rows in self.rows.items():
            if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or not len(rows):
                raise ValueError(f"{part}: expected a nonempty integer row index array")
            if rows.min() < 0 or rows.max() >= self.dataset.count_rows() or len(np.unique(rows)) != len(rows):
                raise ValueError(f"{part}: invalid or duplicate Lance row indices")
            sid = task["subject_column"]
            counts = np.zeros(len(self.labels), dtype=np.int64)
            subjects[part] = set()
            for chunk in batches(rows, 8192):
                records = self.dataset.take(chunk.tolist(), columns=[sid, "label"])
                subjects[part].update(records[sid].to_pylist())
                counts += np.bincount(self.targets(records), minlength=len(self.labels))
            if None in subjects[part] or "" in subjects[part]:
                raise ValueError(f"{part}: missing subject identifier in {sid}")
            if (counts == 0).any():
                raise ValueError(f"{part}: at least one official class has zero support: {counts.tolist()}")
            self.class_counts[part] = counts
        if not profile and subjects["train"] & subjects["val"]:
            raise ValueError("train and validation contain overlapping subjects")

    def targets(self, records):
        return np.array([self.labels[label] for label in records["label"].to_pylist()], dtype=np.int64)

    def eeg(self, rows):
        records = self.dataset.take(rows.tolist(), columns=["data", "shape", "label"])
        shapes = records["shape"].to_pylist()
        data = records["data"].combine_chunks()
        lengths = pc.list_value_length(data).to_numpy(zero_copy_only=False)
        if any(shape is None or tuple(shape) != self.shape for shape in shapes) or not np.all(lengths == math.prod(self.shape)):
            raise ValueError(f"Lance rows do not match configured EEG shape {self.shape}")
        values = pc.list_flatten(data).to_numpy(zero_copy_only=False)
        eeg = np.array(values, dtype=np.float32, copy=True).reshape(-1, *self.shape)
        return torch.from_numpy(eeg), torch.from_numpy(self.targets(records))


def fit_scale(fold, window, task, device):
    rms = []
    with torch.no_grad():
        for rows in batches(fold.rows["train"], 32):
            eeg, _ = fold.eeg(rows)
            rms.append(window(eeg.to(device))[2].cpu().numpy())
    positive = np.concatenate(rms).astype(np.float64)
    positive = positive[positive > 0]
    median = float(np.median(positive)) if len(positive) else 1.0
    log = np.log(positive)
    std = float(log.std()) if len(log) >= 2 and log.min() < log.max() else 0.0
    physical = task["scale_state"] == "physical_uv"
    return {
        "raw_divisor": 100.0 if physical else median,
        "z_multiplier": median / 100.0 if physical else 1.0,
        # Standardizing log(RMS) gives the same result after subtraction of a fixed log reference.
        "log_mean": float(log.mean()) if len(log) else 0.0,
        "log_std": std if std > 0 else 1.0,
        "nonzero_windows": len(positive),
        "scale_eligible": task["scale_state"] != "unknown" and std > 0,
    }


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def run_group(paths, fit, fold, window, task, stats, source, out, seed, device, profile):
    frozen = paths[0].endswith(("_F", "_P"))
    backbone = load_public_backbone(source).to(device)
    if frozen:
        backbone.requires_grad_(False)
    heads = {}
    for name in paths:
        torch.manual_seed(seed)
        heads[name] = TaskHead(task, name).to(device)
    torch.manual_seed(seed)
    optimizers = {}
    for name, head in heads.items():
        param_groups = [{"params": list(head.parameters()), "lr": 1e-3}]
        if not frozen:
            param_groups.append({"params": [p for p in backbone.parameters() if p.requires_grad], "lr": 1e-5})
        optimizers[name] = torch.optim.AdamW(param_groups, weight_decay=1e-2)

    weight = torch.as_tensor(1.0 / fold.class_counts["train"], dtype=torch.float32, device=device)
    effective_batch = 32
    updates = math.ceil(len(fold.rows["train"]) / effective_batch) * fit["epochs"]
    warmup = max(1, math.ceil(updates * 0.1))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step + 1 - warmup) / (updates - warmup)))

    def forward(eeg, mixed_precision):
        mapped, rho, scale, marks = normalized_inputs(window(eeg), stats, raw=paths[0] == "R")
        early = heads["E"].scale_projection if "E" in heads else None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=mixed_precision):
            h = encode(backbone, mapped, rho, scale, early)
            return {name: head(h, rho, scale, marks).float() for name, head in heads.items()}

    best = {}
    step = 0
    for epoch in range(1, fit["epochs"] + 1):
        backbone.train(not frozen)
        for head in heads.values():
            head.train()
        order = np.random.default_rng(seed + epoch).permutation(fold.rows["train"])
        start = time.perf_counter()
        if profile and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        seen = 0
        batch_count = 0
        for rows in batches(order, effective_batch):
            eeg, targets = (item.to(device) for item in fold.eeg(rows))
            denominator = weight[targets].sum()
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            for lo in range(0, len(rows), fit["micro_batch"]):
                hi = lo + fit["micro_batch"]
                logits = forward(eeg[lo:hi], mixed_precision=device.type == "cuda")
                for scores in logits.values():
                    loss = F.cross_entropy(scores, targets[lo:hi], weight=weight, reduction="sum")
                    (loss / denominator).backward()
            for opt in optimizers.values():
                params = [p for group in opt.param_groups for p in group["params"]]
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                for group, base_lr in zip(opt.param_groups, (1e-3, 1e-5)):
                    group["lr"] = base_lr * lr_factor(step)
                opt.step()
            step += 1
            seen += len(rows)
            batch_count += 1
            if profile and batch_count == profile:
                break
        if profile:
            summary = {"paths": paths, "updates": batch_count, "windows": seen}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                summary["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            summary["seconds"] = time.perf_counter() - start
            print(json.dumps(summary), flush=True)
            return {}

        backbone.eval()
        for head in heads.values():
            head.eval()
        hits = {name: np.zeros(len(task["label_values"]), dtype=np.int64) for name in paths}
        with torch.inference_mode():
            for rows in batches(fold.rows["val"], fit["micro_batch"]):
                eeg, targets = fold.eeg(rows)
                for name, scores in forward(eeg.to(device), mixed_precision=False).items():
                    match = scores.argmax(dim=1).cpu() == targets
                    hits[name] += np.bincount(targets[match].numpy(), minlength=len(hits[name]))
        for name, correct in hits.items():
            ba = float(np.mean(correct / fold.class_counts["val"]))
            print(json.dumps({"path": name, "epoch": epoch, "val_ba": ba}), flush=True)
            if name not in best or ba > best[name]["val_ba"]:
                path = out / f"{name}.pt"
                save_checkpoint(path, {
                    "task": task, "seed": seed, "path": name, "stats": stats,
                    "epoch": epoch, "val_ba": ba,
                    "backbone": backbone.state_dict(), "head": heads[name].state_dict(),
                })
                best[name] = {"checkpoint": path.name, "epoch": epoch, "val_ba": ba}
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile-updates", type=int, default=0, help="training throughput only, without validation or checkpoints")
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    spec = config["tasks"][args.task]
    fit = config["training"]
    if fit["epochs"] not in (10, 20, 50) or not 1 <= fit["micro_batch"] <= 32 or args.profile_updates < 0:
        raise ValueError("use 10, 20 or 50 epochs, a micro-batch from 1 to 32 and nonnegative profile updates")
    task = {key: spec.get(key) for key in (
        "channels", "timepoints", "sample_rate", "label_values", "scale_state", "uv_per_value", "saturation_abs"
    )}
    task["dataset_name"] = args.task
    validate_task(task)
    source_hash = hashlib.sha256(args.backbone.read_bytes()).hexdigest()
    if source_hash != config["backbone_sha256"]:
        raise ValueError(f"CBraMod weight hash differs from configured source: {source_hash}")
    out = args.output / args.task / str(args.seed)
    fold = Fold(args.config.resolve().parent, spec, spec["folds"][str(args.seed)], args.profile_updates)
    device = torch.device(args.device)
    window = Window(task).to(device)
    stats = fit_scale(fold, window, task, device)
    if not args.profile_updates:
        out.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"task": args.task, "seed": args.seed, "scale_stats": stats,
                      "class_counts": {p: c.tolist() for p, c in fold.class_counts.items()}}), flush=True)
    groups = [("R",), ("N",), ("L",), ("E",), ("N_F", "L_F", "N_P", "L_P")]
    best = {}
    for group in groups:
        paths = [p for p in group if stats["scale_eligible"] or p not in SCALE_PATHS]
        if paths:
            best.update(run_group(paths, fit, fold, window, task, stats, args.backbone, out,
                                  args.seed, device, args.profile_updates))
    if args.profile_updates:
        return
    chosen = max((p for p in C_ORDER if p in best), key=lambda p: best[p]["val_ba"])
    research = max((p for p in ("N", "L") if p in best), key=lambda p: best[p]["val_ba"])
    result = {"task": task, "seed": args.seed, "backbone_sha256": source_hash, "training": fit,
              "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
              "scale_stats": stats, "candidates": best, "C": chosen, "S": research}
    (out / "selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


def validate_task(task):
    labels = task["label_values"]
    if len(labels) < 2 or any(type(value) is not int for value in labels) or len(set(labels)) != len(labels):
        raise ValueError("label_values must give a fixed, unique integer class order")
    hz = task["sample_rate"]
    expected_hz = 1000 if task["dataset_name"] == "reliability_02" else 200
    if hz != expected_hz or task["channels"] < 1 or task["timepoints"] < hz or task["timepoints"] % hz:
        raise ValueError("configure the official sampling rate and a whole-second EEG shape")
    state = task["scale_state"]
    uv = task["uv_per_value"]
    if state not in ("unknown", "physical_uv", "fixed_multiplier"):
        raise ValueError("scale_state must be unknown, physical_uv, or fixed_multiplier")
    if (state == "physical_uv" and (uv is None or uv <= 0 or not math.isfinite(uv))) or (state != "physical_uv" and uv is not None):
        raise ValueError("only physical_uv uses a positive uv_per_value conversion factor")
    limit = task["saturation_abs"]
    if limit is not None and (limit <= 0 or not math.isfinite(limit)):
        raise ValueError("saturation_abs must be positive and expressed in input units")


if __name__ == "__main__":
    main()
