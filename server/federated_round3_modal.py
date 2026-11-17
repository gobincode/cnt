"""Federated round 3 for the fedmedtrust NIH ChestX-ray14 study.

Continues the existing FedAvg run that produced rounds 1-2 on this volume:
five hospital clients train locally from the shared global ResNet-18, then the
server aggregates with sample-count weighting and evaluates the global model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP_NAME = "fedmedtrust-images"
VOLUME_NAME = "fedmedtrust-images-vol"
MOUNT = "/vol"
HOSPITALS = ("A", "B", "C", "D", "E")
LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)
ROUND = 3

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.1",
    "torchvision==0.19.1",
    "numpy==1.26.4",
    "pandas==2.2.2",
    "scikit-learn==1.5.1",
    "pillow==10.4.0",
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
app = modal.App(APP_NAME)


def _build_model(num_labels: int = len(LABELS)):
    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_labels)
    return model


def _hospital_frame(hospital: str):
    import pandas as pd

    frame = pd.read_csv(f"{MOUNT}/hospital_splits.csv")
    frame = frame[frame["hospital"] == hospital].reset_index(drop=True)
    return frame


class _ChestDataset:
    """Reads one hospital's PNGs and multi-hot labels from the shared volume."""

    def __init__(self, frame, image_root: str, size: int = 224) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.size = size

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        import numpy as np
        import torch
        from PIL import Image

        row = self.frame.iloc[index]
        path = self.image_root / str(row["Image Index"])
        with Image.open(path) as handle:
            array = np.asarray(
                handle.convert("L").resize((self.size, self.size), Image.BILINEAR),
                dtype=np.float32,
            )
        array /= 255.0
        array = (array - 0.5) / 0.25
        images = torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)
        findings = {
            value.strip()
            for value in str(row["Finding Labels"]).split("|")
            if value.strip()
        }
        targets = torch.tensor(
            [1.0 if label in findings else 0.0 for label in LABELS],
            dtype=torch.float32,
        )
        return images, targets


def _split(frame, seed: int = 20260923):
    train = frame.sample(frac=0.7, random_state=seed)
    rest = frame.drop(train.index)
    validation = rest.sample(frac=0.5, random_state=seed)
    test = rest.drop(validation.index)
    return train, validation, test


def _evaluate(model, loader, device):
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    model.eval()
    probabilities: list = []
    targets: list = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(labels.numpy())
    probability = np.concatenate(probabilities)
    target = np.concatenate(targets)
    per_label = {}
    scores = []
    for index, label in enumerate(LABELS):
        column = target[:, index]
        if column.min() == column.max():
            continue
        auc = float(roc_auc_score(column, probability[:, index]))
        per_label[label] = {"auc": auc}
        scores.append(auc)
    return (float(np.mean(scores)) if scores else float("nan")), per_label


@app.function(
    image=image,
    gpu="T4",
    volumes={MOUNT: volume},
    timeout=3600,
)
def train_client(hospital: str, max_train: int, epochs: int) -> dict:
    """Train one hospital locally starting from the current global weights."""
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = _hospital_frame(hospital)
    train_frame, validation_frame, test_frame = _split(frame)
    if max_train > 0:
        train_frame = train_frame.head(max_train)
        validation_frame = validation_frame.head(max(256, max_train // 4))
        test_frame = test_frame.head(max(256, max_train // 4))

    image_root = f"{MOUNT}/images"
    loaders = {
        name: DataLoader(
            _ChestDataset(part, image_root),
            batch_size=32,
            shuffle=name == "train",
            num_workers=4,
            drop_last=False,
        )
        for name, part in (
            ("train", train_frame),
            ("validation", validation_frame),
            ("test", test_frame),
        )
    }

    model = _build_model().to(device)
    global_path = Path(f"{MOUNT}/models/global_model_latest.pt")
    model.load_state_dict(torch.load(global_path, map_location=device))

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    for _ in range(epochs):
        for images, targets in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()

    validation_auc, _ = _evaluate(model, loaders["validation"], device)
    test_auc, per_label = _evaluate(model, loaders["test"], device)

    state = {key: value.cpu() for key, value in model.state_dict().items()}
    torch.save(state, f"{MOUNT}/models/hospital_{hospital}_round{ROUND}.pt")
    payload = {
        "hospital": hospital,
        "round": ROUND,
        "n_train": int(len(train_frame)),
        "n_val": int(len(validation_frame)),
        "n_test": int(len(test_frame)),
        "val_auc": validation_auc,
        "test_auc": test_auc,
        "per_label_metrics": per_label,
    }
    Path(f"{MOUNT}/results/hospital_{hospital}_round{ROUND}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    volume.commit()
    return payload


@app.function(image=image, volumes={MOUNT: volume}, timeout=1800)
def aggregate(reports: list[dict]) -> dict:
    """FedAvg: weight each client's parameters by its local sample count."""
    import torch

    weights = [float(item["n_train"]) for item in reports]
    total = sum(weights)
    if total <= 0:
        raise ValueError("clients reported no training samples")

    averaged: dict | None = None
    for report, weight in zip(reports, weights):
        state = torch.load(
            f"{MOUNT}/models/hospital_{report['hospital']}_round{ROUND}.pt",
            map_location="cpu",
        )
        share = weight / total
        if averaged is None:
            averaged = {
                key: value.to(torch.float64) * share for key, value in state.items()
            }
            continue
        for key, value in state.items():
            averaged[key] += value.to(torch.float64) * share

    assert averaged is not None
    reference = torch.load(
        f"{MOUNT}/models/hospital_{reports[0]['hospital']}_round{ROUND}.pt",
        map_location="cpu",
    )
    merged = {key: averaged[key].to(reference[key].dtype) for key in reference}
    torch.save(merged, f"{MOUNT}/models/global_model_round{ROUND}.pt")
    torch.save(merged, f"{MOUNT}/models/global_model_latest.pt")

    summary = {
        "round": ROUND,
        "mean_val_auc": sum(item["val_auc"] for item in reports) / len(reports),
        "mean_test_auc": sum(item["test_auc"] for item in reports) / len(reports),
        "hospitals": {
            item["hospital"]: {
                "n_train": item["n_train"],
                "val_auc": item["val_auc"],
                "test_auc": item["test_auc"],
            }
            for item in reports
        },
    }
    Path(f"{MOUNT}/results/round_{ROUND}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    volume.commit()
    return summary


@app.local_entrypoint()
def main(max_train: int = 2000, epochs: int = 1) -> None:
    started = time.monotonic()
    reports = list(
        train_client.starmap(
            [(hospital, max_train, epochs) for hospital in HOSPITALS]
        )
    )
    for report in reports:
        print(
            f"hospital {report['hospital']}: "
            f"val={report['val_auc']:.4f} test={report['test_auc']:.4f} "
            f"n_train={report['n_train']}"
        )
    summary = aggregate.remote(reports)
    print(json.dumps(summary, indent=2))
    print(f"elapsed {time.monotonic() - started:.1f}s")
