"""Real inference over the five-fold Track B C1 ensemble.

Model construction, checkpoint verification and study preparation follow the
packaged Kaggle inference script so the weights served here are the same ones
that produced the recorded out-of-fold metrics. Folds are loaded once and kept
resident, because each checkpoint is roughly 89 MB.
"""

from __future__ import annotations

import hashlib
import math
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from study_ingest import SLOT_NAMES, IngestedSeries

LABELS: tuple[str, ...] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

EXPECTED_CHECKPOINT_HASHES: tuple[str, ...] = (
    "95f5eb633a3edc7c8029605f05513334821ba7d18e98c23002f00f14a82e1309",
    "dff3c815df2d72fca7d06b7140a271f9abeffd18bbf3c5c6e22e73ab10293182",
    "e58fc4fb32cd7f450ec00537cce9cea919c6cf094c080358c499422c046f60eb",
    "6e8a01af45742a00a3ff1289ea477f875320c9fd11c049f1268308be03d6bb02",
    "dc670537a5426eed25c3b2f35ce846e5faea2dd1bc2fb0c642e590a3945bf96a",
)

ASSET_ROOT = (
    Path(__file__).resolve().parents[2]
    / "rsna-knee-abnormality-detection"
    / "kaggle_c1_hidden_inference"
    / "assets"
)


@dataclass(frozen=True)
class Config:
    image_size: int = 336
    feature_dim: int = 768
    hidden_dim: int = 256
    dropout: float = 0.2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_modules():
    """Define the served architecture against the installed torch."""
    import torch
    from torch import Tensor, nn

    class SparseDinoCLSMeanBackbone(nn.Module):
        def __init__(self, backbone: nn.Module) -> None:
            super().__init__()
            self.backbone = backbone
            self.output_dim = int(backbone.embed_dim) * 2
            # Spatial patch tokens are pooled away by the trained pathway. They
            # are retained only when explanation is requested, so the served
            # forward pass stays byte-identical to the evaluated one.
            self.capture_patches = False
            self.last_patch_tokens: Tensor | None = None

        def forward(self, images: Tensor) -> Tensor:
            tokens = self.backbone.prepare_tokens_with_masks(images)
            for block in self.backbone.blocks:
                tokens = block(tokens)
            tokens = self.backbone.norm(tokens)
            if self.capture_patches:
                self.last_patch_tokens = tokens[:, 1:]
            return torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1)

    class MaskedPerLabelSlotAttention(nn.Module):
        """Per-label attention over the six slots, which is what we explain."""

        def __init__(self, feature_dim: int, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.slot_count = len(SLOT_NAMES)
            self.projection = nn.Sequential(
                nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU()
            )
            self.slot_embedding = nn.Parameter(torch.randn(self.slot_count, hidden_dim) * 0.02)
            self.label_queries = nn.Parameter(torch.randn(len(LABELS), hidden_dim) * 0.02)
            self.dropout = nn.Dropout(dropout)
            self.output = nn.Linear(hidden_dim, len(LABELS))
            self.last_attention: Tensor | None = None

        def forward(self, features: Tensor, valid_mask: Tensor) -> Tensor:
            mask = valid_mask.bool()
            hidden = self.projection(features)
            hidden = hidden + self.slot_embedding.unsqueeze(0).to(hidden.dtype)
            scores = torch.einsum(
                "bsh,lh->bls", hidden, self.label_queries.to(hidden.dtype)
            ) / math.sqrt(self.hidden_dim)
            scores = scores.masked_fill(~mask.unsqueeze(1), -1e4)
            attention = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
            attention = attention * mask.unsqueeze(1).to(attention.dtype)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            # Retained so the API can report which slot drove each label.
            self.last_attention = attention.detach().float().cpu()
            context = self.dropout(torch.einsum("bls,bsh->blh", attention, hidden))
            return (
                context * self.output.weight.unsqueeze(0).to(context.dtype)
            ).sum(dim=-1) + self.output.bias

    class SixSlotStudyEncoder(nn.Module):
        def __init__(self, slice_backbone: nn.Module, config: Config) -> None:
            super().__init__()
            self.slice_backbone = slice_backbone
            self.feature_dim = config.feature_dim
            self.head = MaskedPerLabelSlotAttention(
                config.feature_dim, config.hidden_dim, config.dropout
            )

        def forward(self, images: Tensor, valid_mask: Tensor) -> Tensor:
            if images.ndim == 6:
                group_images = images.transpose(1, 2)
                group_mask = valid_mask.transpose(1, 2).bool()
                view_mask = group_mask.any(dim=2)
                flat_images = group_images.reshape(-1, len(SLOT_NAMES), *images.shape[3:])
                flat_mask = group_mask.reshape(-1, len(SLOT_NAMES))
                valid_logits = self.forward(
                    flat_images[view_mask.reshape(-1)], flat_mask[view_mask.reshape(-1)]
                )
                logits = valid_logits.new_zeros(
                    images.shape[0] * images.shape[2], valid_logits.shape[-1]
                )
                logits[view_mask.reshape(-1)] = valid_logits
                logits = logits.view(images.shape[0], images.shape[2], -1)
                weights = view_mask.unsqueeze(-1).to(logits.dtype)
                return (logits * weights).sum(dim=1) / weights.sum(dim=1)

            mask = valid_mask.bool()
            flattened = images.reshape(-1, *images.shape[2:])
            flat_mask = mask.reshape(-1)
            valid_features = self.slice_backbone(flattened[flat_mask])
            features = valid_features.new_zeros(
                images.shape[0] * images.shape[1], self.feature_dim
            )
            features[flat_mask] = valid_features
            return self.head(features.view(images.shape[0], images.shape[1], -1), mask)

    return SparseDinoCLSMeanBackbone, SixSlotStudyEncoder


def _create_model(config: Config):
    source = ASSET_ROOT / "dinov2_source"
    if not (source / "dinov2").is_dir():
        raise FileNotFoundError("pinned DINOv2 source is missing")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from dinov2.hub.backbones import dinov2_vits14

    adapter_cls, encoder_cls = _build_modules()
    backbone = dinov2_vits14(pretrained=False)
    adapter = adapter_cls(backbone)
    if adapter.output_dim != config.feature_dim:
        raise RuntimeError("DINOv2 feature dimension is incompatible")
    return encoder_cls(adapter, config)


class EnsembleRegistry:
    """Loads folds once and keeps them resident across requests."""

    def __init__(self) -> None:
        self._models: dict[int, Any] = {}
        self._lock = threading.Lock()
        self._config = Config()

    @property
    def available_folds(self) -> list[int]:
        root = ASSET_ROOT / "checkpoints"
        return [
            fold for fold in range(5) if (root / f"fold-{fold}.pt").is_file()
        ]

    def is_ready(self) -> bool:
        return (ASSET_ROOT / "dinov2_source" / "dinov2").is_dir() and bool(
            self.available_folds
        )

    @property
    def loaded_folds(self) -> list[int]:
        with self._lock:
            return sorted(self._models)

    def load(self, fold: int):
        if fold not in range(5):
            raise ValueError("Invalid fold.")
        with self._lock:
            if fold in self._models:
                return self._models[fold]
            import torch

            path = ASSET_ROOT / "checkpoints" / f"fold-{fold}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"fold {fold} checkpoint is missing")
            if sha256_file(path) != EXPECTED_CHECKPOINT_HASHES[fold]:
                raise RuntimeError(f"fold {fold} checkpoint hash mismatch")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if (
                checkpoint.get("schema_version") != 1
                or checkpoint.get("stage") != "TB-P0"
                or checkpoint.get("fold") != fold
                or checkpoint.get("coverage_arm") != "c1"
            ):
                raise RuntimeError(f"fold {fold} checkpoint metadata mismatch")
            raw = checkpoint["config"]
            config = Config(image_size=int(raw["image_size"]), feature_dim=int(raw["feature_dim"]),
                            hidden_dim=int(raw["hidden_dim"]), dropout=float(raw["dropout"]))
            if config.image_size != 336:
                raise RuntimeError("Unsupported checkpoint image size.")
            model = _create_model(config)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self._models[fold] = model
            return model


REGISTRY = EnsembleRegistry()


def prepare_images(series: list[IngestedSeries], slot_resolution: dict[str, object]):
    """Build the six-slot tensor the encoder expects, as at training time."""
    import torch

    from study_ingest import _ensure_rsna_on_path

    _ensure_rsna_on_path()
    from rsna_knee.track_b_data import (
        TBCoverageConfig,
        TBP0DataConfig,
        build_grouped_six_slot_images,
        LateralityDecision,
    )

    by_uid = {item.series_uid: item for item in series}
    slot_series = {}
    for entry in slot_resolution["slots"]:
        if entry["present"] and entry["seriesUid"] in by_uid:
            slot_series[entry["slot"]] = by_uid[entry["seriesUid"]].files
    if not slot_series:
        raise ValueError("no usable series were assigned to protocol slots")

    side = slot_resolution.get("laterality")
    if side not in {"L", "R"}:
        raise ValueError("Laterality unresolved; cannot safely canonicalize input.")
    laterality = LateralityDecision(side, str(slot_resolution["lateralitySource"]),
                                    slot_resolution.get("lateralityCenterMm"), None)
    images, mask, _, _ = build_grouped_six_slot_images(
        slot_series,
        laterality,
        config=TBP0DataConfig(image_size=336, crop_mm=130.0),
        coverage=TBCoverageConfig(group_count=4),
    )
    images = images.float().div_(255)
    mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
    std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
    images = (images - mean) / std
    images[~mask] = 0
    return images, mask


# Attention is captured on the shared head: lock the entire forward + read.
INFERENCE_LOCK = threading.Lock()


def run_ensemble(images, mask, folds: list[int], progress=None):
    """Average sigmoid probabilities; report actual extraction/forward progress."""
    import numpy as np
    import torch

    if not folds or len(set(folds)) != len(folds) or any(f not in range(5) for f in folds):
        raise ValueError("Invalid fold selection.")
    if not mask.any():
        raise ValueError("No usable input; predictions are unavailable.")
    probabilities, attentions = [], []
    with INFERENCE_LOCK:
        for fold in folds:
            if progress:
                progress(fold, "loading")
            model = REGISTRY.load(fold)
            if progress:
                progress(fold, "forward")
            with torch.inference_mode():
                logits = model(images.unsqueeze(0), mask.unsqueeze(0))
            probabilities.append(torch.sigmoid(logits)[0].cpu().numpy())
            captured = model.head.last_attention
            if captured is None:
                raise RuntimeError("Trained attention was not captured.")
            attentions.append(captured.mean(dim=0).numpy().copy())
            if progress:
                progress(fold, "done")
        probability, attention = np.mean(probabilities, axis=0), np.mean(attentions, axis=0)
    if (probability.shape != (12,) or not np.isfinite(probability).all()
            or not ((probability >= 0) & (probability <= 1)).all()
            or attention.shape != (12, 6) or not np.isfinite(attention).all()):
        raise RuntimeError("Invalid model output; predictions are unavailable.")
    return probability, attention


class UnsupportedSaliency(RuntimeError):
    """Raised when a backbone cannot expose patch tokens to explain."""


def compute_saliency(images, mask, fold: int):
    """Per-label spatial saliency from real patch tokens and trained weights.

    The served model pools DINOv2 patch tokens into one vector per slot, so the
    deployed forward pass carries no spatial signal to report. Here each patch
    token is pushed through the same trained head (projection, slot embedding,
    label query, output row) that produces the logit, giving each patch's
    contribution to that label. This is a faithful attribution of the trained
    weights, not a saliency heuristic layered on top, but it describes where
    evidence was pooled from, not a lesion boundary.

    Returns (grids, slots, side) where grids is [label][slot] -> HxW floats in
    0..1, and slots lists the slot index each grid column belongs to.
    """
    import numpy as np
    import torch

    model = REGISTRY.load(fold)
    backbone = model.slice_backbone
    head = model.head
    if not hasattr(backbone, "capture_patches"):
        raise UnsupportedSaliency("backbone does not expose patch tokens")
    # One representative view per slot: the first coverage group that is valid.
    slot_views, slot_indices = [], []
    for slot in range(images.shape[0]):
        groups = torch.nonzero(mask[slot], as_tuple=False).flatten()
        if groups.numel():
            slot_views.append(images[slot, int(groups[0])])
            slot_indices.append(slot)
    if not slot_views:
        raise ValueError("No usable input; saliency is unavailable.")
    batch = torch.stack(slot_views)

    backbone.capture_patches = True
    try:
        with torch.inference_mode():
            backbone(batch)
            tokens = backbone.last_patch_tokens
            if tokens is None or tokens.ndim != 3 or tokens.shape[1] < 4:
                raise UnsupportedSaliency("patch tokens were not captured")
            tokens = tokens.clone()
    finally:
        backbone.capture_patches = False
        backbone.last_patch_tokens = None

    with torch.inference_mode():
        # Mirror the pooled feature layout the head was trained on: the CLS
        # half is shared by every patch, the mean half becomes the patch itself.
        cls_half = tokens.mean(dim=1, keepdim=True).expand_as(tokens)
        features = torch.cat((cls_half, tokens), dim=-1)
        hidden = head.projection(features)
        embedding = head.slot_embedding.to(hidden.dtype)
        hidden = hidden + embedding[slot_indices].unsqueeze(1)
        # Each patch's contribution to each label logit under the trained
        # output row, which is exactly the quantity the logit sums over.
        contribution = torch.einsum(
            "sph,lh->lsp", hidden, head.output.weight.to(hidden.dtype)
        ).float().cpu().numpy()

    patches = contribution.shape[-1]
    size = int(round(math.sqrt(patches)))
    if size * size != patches:
        raise UnsupportedSaliency("patch grid is not square")
    grids = contribution.reshape(len(LABELS), len(slot_indices), size, size)
    # Normalize per label+slot so the map shows relative emphasis within a view.
    # Percentile bounds rather than min/max: a single extreme patch would
    # otherwise define the top of the scale and flatten everything else.
    flat = grids.reshape(len(LABELS), len(slot_indices), -1)
    low = np.percentile(flat, 2.0, axis=-1, keepdims=True)
    high = np.percentile(flat, 98.0, axis=-1, keepdims=True)
    scaled = np.clip((flat - low) / np.clip(high - low, 1e-8, None), 0.0, 1.0)
    grids = scaled.reshape(len(LABELS), len(slot_indices), size, size)
    if not np.isfinite(grids).all():
        raise RuntimeError("Invalid saliency output.")
    return grids, slot_indices, size


def slot_preview(images, mask, slot: int, group: int, size: int | None = None):
    """Decoded, de-identified slot image as base64 PNG for the live preview.

    Served at the model's own 336 px resolution: downscaling here and letting
    the browser stretch the result back up is what makes a preview look blurry.
    Intensities are window-levelled to the 1st-99th percentile of the slice, the
    same display convention a PACS viewer applies, because normalized network
    input occupies only a narrow part of the 0-255 range and renders flat grey.
    """
    import base64
    import io

    import numpy as np
    from PIL import Image

    view = images[slot, group]
    mean = np.array((0.485, 0.456, 0.406)).reshape(3, 1, 1)
    std = np.array((0.229, 0.224, 0.225)).reshape(3, 1, 1)
    array = view.float().cpu().numpy() * std + mean
    array = np.clip(array, 0, 1)[0]
    low, high = np.percentile(array, 1.0), np.percentile(array, 99.0)
    if high - low > 1e-6:
        array = np.clip((array - low) / (high - low), 0, 1)
    image = Image.fromarray((array * 255).astype("uint8"), mode="L")
    if size is not None and size != image.width:
        image = image.resize((size, size), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
