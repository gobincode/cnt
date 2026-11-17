"""Single-process FedAvg experiment for a 12-label knee abnormality head.

Runs real FedAvg over the cached DINOv2-336 study embeddings: studies are
partitioned into hospital silos by report group so no silo shares a report with
another, each simulated silo trains its own MLP head, and the server averages
parameters weighted by silo sample count. All clients are emulated inside one
Modal process. Optional Gaussian noise is not a differential privacy guarantee.
"""

from __future__ import annotations

import json
import time

import modal

APP_NAME = "kneescan-federated-knee"
LABELS = (
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
SLOT_NAMES = (
    "SAG_FLUID_FS",
    "COR_FLUID_FS",
    "AX_FLUID_FS",
    "SAG_FLUID_NOFS",
    "COR_T1",
    "SAG_T1",
)
HOSPITALS = ("Northside General", "Lakeview Ortho", "St. Mary Imaging", "Central Trauma", "Riverside Clinic")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.1",
    "numpy==1.26.4",
    "pandas==2.2.2",
    "scikit-learn==1.5.1",
)
data_volume = modal.Volume.from_name("rsna-knee-data", create_if_missing=False)
artifact_volume = modal.Volume.from_name("rsna-knee-artifacts", create_if_missing=False)
app = modal.App(APP_NAME)

FEATURE_ROOT = "/artifacts/features/dino336-v1/train"
VIEW_COUNT = 46
FEATURE_DIM = 1152


def _load_matrix(shard_limit: int):
    """Load cached embeddings, mean-pooled over valid views, into one matrix."""
    from pathlib import Path

    import numpy as np

    shards = sorted(Path(FEATURE_ROOT).glob("features-*-of-256.npz"))
    if shard_limit > 0:
        shards = shards[:shard_limit]
    ids: list = []
    rows: list = []
    for shard in shards:
        payload = np.load(shard, allow_pickle=True)
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        mask = np.asarray(payload["valid_mask"], dtype=bool)
        counts = mask.sum(axis=1, keepdims=True).clip(min=1).astype(np.float32)
        pooled = ((embeddings * mask[:, :, None]).sum(axis=1) / counts).astype(np.float32)
        ids.extend(list(np.asarray(payload["study_ids"]).astype(str)))
        rows.append(pooled)
    if not rows:
        raise ValueError("no feature shards found")
    return ids, np.concatenate(rows)


def _silo_assignment(report_groups, silo_count: int):
    """Partition by report group so a silo never shares a report with another."""
    import hashlib

    assignment = []
    for group in report_groups:
        digest = hashlib.sha256(str(group).encode("utf-8")).hexdigest()
        assignment.append(int(digest[:8], 16) % silo_count)
    return assignment


def _build_head(input_dim: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, 256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, len(LABELS)),
    )


def _evaluate(model, features, targets, device):
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features).to(device))
        probability = torch.sigmoid(logits).cpu().numpy()
    per_label = {}
    scores = []
    for index, label in enumerate(LABELS):
        column = (targets[:, index] > 0.5).astype(np.int32)
        if column.min() == column.max():
            continue
        auc = float(roc_auc_score(column, probability[:, index]))
        per_label[label] = auc
        scores.append(auc)
    return (float(np.mean(scores)) if scores else float("nan")), per_label


@app.function(
    image=image,
    volumes={"/data": data_volume, "/artifacts": artifact_volume},
    timeout=3600,
    memory=16384,
    cpu=4,
)
def federated_run(
    rounds: int = 6,
    silo_count: int = 5,
    local_epochs: int = 2,
    shard_limit: int = 40,
    noise_multiplier: float = 0.0,
) -> dict:
    """Execute every FedAvg round server-side and return the full history."""
    import numpy as np
    import pandas as pd
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(20260923)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.monotonic()

    ids, features = _load_matrix(shard_limit)
    labels = pd.read_csv("/data/metadata/consensus-labels-v1.csv").set_index(
        "StudyInstanceUID"
    )
    keep = [index for index, study in enumerate(ids) if study in labels.index]
    features = features[keep]
    ordered = [ids[index] for index in keep]
    frame = labels.loc[ordered]
    targets = frame.loc[:, list(LABELS)].to_numpy(dtype=np.float32)
    silos = _silo_assignment(frame["report_group"].tolist(), silo_count)
    silos = np.asarray(silos)

    # Hold out a shared server-side test split per silo for honest reporting.
    rng = np.random.default_rng(20260923)
    import hashlib
    is_test = np.asarray([
        int(hashlib.sha256(("test:" + str(group)).encode()).hexdigest()[:8], 16) % 4 == 0
        for group in frame["report_group"]
    ])

    global_model = _build_head(features.shape[1]).to(device)
    global_state = {k: v.detach().clone() for k, v in global_model.state_dict().items()}

    silo_profile = []
    for silo in range(silo_count):
        member = silos == silo
        silo_profile.append(
            {
                "hospital": HOSPITALS[silo % len(HOSPITALS)],
                "studies": int(member.sum()),
                "trainStudies": int((member & ~is_test).sum()),
                "testStudies": int((member & is_test).sum()),
                "prevalence": {
                    label: float((targets[member, index] > 0.5).mean())
                    for index, label in enumerate(LABELS)
                },
            }
        )

    history = []
    criterion = torch.nn.BCEWithLogitsLoss()
    for round_index in range(1, rounds + 1):
        client_states = []
        client_weights = []
        client_reports = []
        for silo in range(silo_count):
            member = silos == silo
            train_index = np.where(member & ~is_test)[0]
            test_index = np.where(member & is_test)[0]
            if train_index.size == 0 or test_index.size == 0:
                continue

            model = _build_head(features.shape[1]).to(device)
            model.load_state_dict(global_state)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
            model.train()
            batch = 64
            for _ in range(local_epochs):
                order = rng.permutation(train_index)
                for start in range(0, order.size, batch):
                    chunk = order[start : start + batch]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(torch.from_numpy(features[chunk]).to(device))
                    loss = criterion(
                        logits, torch.from_numpy(targets[chunk]).to(device)
                    )
                    loss.backward()
                    optimizer.step()

            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if noise_multiplier > 0:
                # Gaussian mechanism on the client delta, then re-anchor.
                for key in state:
                    if state[key].is_floating_point():
                        delta = state[key] - global_state[key]
                        scale = noise_multiplier * delta.norm() / max(delta.numel(), 1) ** 0.5
                        state[key] = global_state[key] + delta + torch.randn_like(delta) * scale

            local_auc, local_per_label = _evaluate(
                model, features[test_index], targets[test_index], device
            )
            client_states.append(state)
            client_weights.append(float(train_index.size))
            client_reports.append(
                {
                    "hospital": HOSPITALS[silo % len(HOSPITALS)],
                    "localAuc": local_auc,
                    "perLabelAuc": local_per_label,
                    "trainStudies": int(train_index.size),
                }
            )

        total = sum(client_weights)
        averaged = {
            key: sum(
                state[key].to(torch.float64) * (weight / total)
                for state, weight in zip(client_states, client_weights)
            )
            for key in client_states[0]
        }
        global_state = {
            key: averaged[key].to(client_states[0][key].dtype) for key in averaged
        }
        global_model.load_state_dict(global_state)

        test_index = np.where(is_test)[0]
        global_auc, global_per_label = _evaluate(
            global_model, features[test_index], targets[test_index], device
        )
        history.append(
            {
                "round": round_index,
                "globalMacroAuc": global_auc,
                "globalPerLabelAuc": global_per_label,
                "clients": client_reports,
                "meanClientAuc": float(
                    np.mean([item["localAuc"] for item in client_reports])
                ),
            }
        )
        print(f"round {round_index}/{rounds}: held-out macro AUC {global_auc:.4f}", flush=True)

    return {
        "schemaVersion": 1,
        "strategy": "FedAvg",
        "rounds": rounds,
        "siloCount": silo_count,
        "localEpochs": local_epochs,
        "noiseMultiplier": noise_multiplier,
        "studies": int(len(ordered)),
        "featureDim": int(features.shape[1]),
        "labels": list(LABELS),
        "silos": silo_profile,
        "history": history,
        "finalMacroAuc": history[-1]["globalMacroAuc"],
        "elapsedSeconds": round(time.monotonic() - started, 1),
        "device": str(device),
        "limitations": [
            "Simulated hospital partitions of one public dataset, not separate institutions.",
            "FedAvg clients are emulated in one Modal process; not a deployed multi-hospital network.",
            "Frozen DINOv2 feature head experiment, separate from the six-slot inference ensemble.",
            "Report-group-disjoint holdout; cached backbone training provenance has not been audited for this holdout.",
            "No secure aggregation or differential privacy guarantee; optional Gaussian noise is experimental only.",
        ],
    }


@app.local_entrypoint()
def main(
    rounds: int = 6,
    silos: int = 5,
    local_epochs: int = 2,
    shard_limit: int = 40,
    noise_multiplier: float = 0.0,
    output: str = "federated_knee_run.json",
) -> None:
    result = federated_run.remote(
        rounds=rounds,
        silo_count=silos,
        local_epochs=local_epochs,
        shard_limit=shard_limit,
        noise_multiplier=noise_multiplier,
    )
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    for entry in result["history"]:
        print(
            f"round {entry['round']}: global={entry['globalMacroAuc']:.4f} "
            f"mean_client={entry['meanClientAuc']:.4f}"
        )
    print(f"studies={result['studies']} elapsed={result['elapsedSeconds']}s -> {output}")
