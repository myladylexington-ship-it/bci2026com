import torch
from torch import nn
from torch.nn import functional as F

from cbramod.cbramod import CBraMod


C_ORDER = ("N_P", "N_F", "N", "L_P", "L_F", "L", "E", "R")
SCALE_PATHS = {"E", "L", "L_F", "L_P"}


class Window(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.hz = task["sample_rate"]
        self.uv = task.get("uv_per_value")
        self.saturation = task.get("saturation_abs")
        if self.hz == 1000:
            taps = torch.arange(-50, 51, dtype=torch.float32)
            kernel = torch.sinc(0.16 * taps) * torch.hann_window(101, periodic=False)
            self.register_buffer("kernel", (kernel / kernel.sum()).view(1, 1, -1))

    def forward(self, eeg):
        valid = eeg.isfinite()
        eeg = eeg.masked_fill(~valid, 0)
        n_valid = valid.sum(dim=(1, 2))
        zero = (n_valid > 0) & ~(eeg != 0).any(dim=(1, 2))
        saturated = (eeg.abs() >= self.saturation).any(dim=(1, 2)) if self.saturation else torch.zeros_like(zero)
        marks = torch.stack(
            (zero, n_valid == 0, (n_valid > 0) & (n_valid < eeg[0].numel()),
             (~valid).any(dim=(1, 2)), saturated), dim=1
        ).float()
        if self.uv is not None:
            eeg = eeg * self.uv
        if self.hz == 1000:
            batch, channels, _ = eeg.shape
            padded = F.pad(eeg.flatten(0, 1).unsqueeze(1), (50, 50), mode="reflect")
            eeg = F.conv1d(padded, self.kernel, stride=5).reshape(batch, channels, -1)
            valid = valid.unflatten(-1, (-1, 5)).all(dim=-1)
            eeg = eeg.masked_fill(~valid, 0)
        blocks = eeg.unflatten(-1, (-1, 200))
        rho = valid.unflatten(-1, (-1, 200)).float().mean(dim=-1)
        rms = (eeg.square().sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1)).sqrt()
        return blocks, rho, rms, marks


def normalized_inputs(parts, stats, raw):
    blocks, rho, rms, marks = parts
    nonzero = rms > 0
    safe_rms = torch.where(nonzero, rms, 1.0)
    if raw:
        mapped = blocks / stats["raw_divisor"]
    else:
        mapped = blocks / safe_rms[:, None, None, None] * stats["z_multiplier"]
    scale = (safe_rms.log() - stats["log_mean"]) / stats["log_std"]
    return mapped, rho, torch.where(nonzero, scale, 0.0), marks


def new_backbone():
    backbone = CBraMod()
    backbone.proj_out = nn.Identity()
    return backbone


def load_public_backbone(path):
    backbone = CBraMod()
    backbone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    backbone.proj_out = nn.Identity()
    return backbone


def encode(backbone, mapped, rho, scale, early_projection=None):
    observed = (rho > 0).flatten(1).any(dim=1).nonzero(as_tuple=True)[0]
    if observed.numel() == 0:
        return torch.zeros_like(mapped)
    embedded = backbone.patch_embedding(mapped[observed])
    if early_projection is not None:
        embedded = embedded + (
            (rho[observed] > 0)[..., None] * scale[observed, None, None, None] * early_projection
        )
    h = backbone.encoder(embedded)
    if observed.numel() == mapped.shape[0]:
        return h
    return h.new_zeros(mapped.shape).index_copy(0, observed, h)


class TaskHead(nn.Module):
    def __init__(self, task, path):
        super().__init__()
        self.path = path
        self.pool = path.endswith("_P")
        tokens = task["channels"] * (task["timepoints"] // task["sample_rate"])
        self.compress = nn.Linear(200, 32)
        self.readout = nn.Sequential(
            nn.Linear(32 if self.pool else 32 * tokens, 256),
            nn.ELU(), nn.Dropout(0.1), nn.Linear(256, 200),
        )
        self.activation = nn.Sequential(nn.ELU(), nn.Dropout(0.1))
        self.classifier = nn.Linear(200 + 5, len(task["label_values"]))
        self.scale_projection = nn.Parameter(torch.zeros(200)) if path in SCALE_PATHS else None

    def forward(self, h, rho, scale, marks):
        compressed = self.compress(h) * rho[..., None]
        if self.pool:
            denom = rho.sum(dim=(1, 2)).unsqueeze(1)
            readout_input = compressed.sum(dim=(1, 2)) / torch.where(denom > 0, denom, 1.0)
        else:
            readout_input = compressed.flatten(1)
        readout = self.readout(readout_input)
        if self.path.startswith("L"):
            readout = readout + scale[:, None] * self.scale_projection
        return self.classifier(torch.cat((self.activation(readout), marks), dim=1))
