from __future__ import annotations

import io
import json
import os
import pathlib
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pydicom
import pytest
import torch
from fastapi.testclient import TestClient
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

sys.path.insert(0, str(Path(__file__).parent))
import main
import inference_engine as engine
from deidentify import deidentify_dataset, pseudonymise
from study_ingest import SLOT_NAMES, _infer_fat_suppression, read_study, resolve_slots, sanitize_upload


def dicom(study="1.2.3", series="1.2.3.4", instance=None, side="R", description="SAG PD FS", rows=32):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = instance or generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = study, series
    ds.PatientName, ds.PatientID = "SECRET^PATIENT", "SECRET-ID"
    ds.InstitutionName, ds.StudyDate = "SECRET-HOSPITAL", "20200101"
    ds.SeriesDescription = description + " SECRET-PATIENT"
    ds.Modality = "MR"
    ds.ImageLaterality = side
    ds.Rows, ds.Columns = rows, 32
    ds.SamplesPerPixel, ds.BitsAllocated, ds.BitsStored, ds.HighBit = 1, 16, 16, 15
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelSpacing = [1, 1]
    ds.ImageOrientationPatient = [0, 1, 0, 0, 0, 1]
    ds.ImagePositionPatient = [-40, 0, 0] if side else [0, 0, 0]
    ds.InstanceNumber = 1
    ds.PixelData = np.arange(max(rows, 1) * 32, dtype=np.uint16).tobytes()
    child = Dataset()
    child.PatientName = "SECRET^NESTED"
    child.ReferencedSOPInstanceUID = ds.SOPInstanceUID
    child.add_new((0x0011, 0x1010), "LO", "SECRET-PRIVATE")
    ds.ReferencedImageSequence = Sequence([child])
    ds.add_new((0x0011, 0x1010), "LO", "SECRET-PRIVATE")
    return ds


def encoded(ds):
    output = io.BytesIO()
    ds.save_as(output, enforce_file_format=True)
    return output.getvalue()


@pytest.fixture
def client():
    with TestClient(main.app) as client:
        yield client
    for job_id, job in list(main.JOBS.items()):
        main.discard(job_id, job)


def upload(client, datasets=None):
    datasets = datasets or [dicom()]
    return client.post("/api/studies", files=[("files", (f"folder/{i}.dcm", encoded(ds), "application/dicom"))
                                               for i, ds in enumerate(datasets)])


def events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


@pytest.fixture
def fake_model(monkeypatch):
    _, encoder = engine._build_modules()

    class Backbone(torch.nn.Module):
        def forward(self, images):
            return images.mean(dim=(-2, -1)).repeat(1, 2)

    model = encoder(Backbone(), engine.Config(feature_dim=6, hidden_dim=8)).eval()
    monkeypatch.setattr(engine.REGISTRY, "load", lambda fold: model)
    return model


def test_recursive_sanitization_and_uid_consistency(tmp_path):
    ds = dicom()
    original = ds.SOPInstanceUID
    report = sanitize_upload(encoded(ds), tmp_path, "salt", 0)
    raw = (tmp_path / "000000.dcm").read_bytes()
    assert b"SECRET" not in raw
    clean = pydicom.dcmread(tmp_path / "000000.dcm")
    assert "PatientName" not in clean
    assert clean.StudyDate == ""
    assert clean.ReferencedImageSequence[0].ReferencedSOPInstanceUID == clean.SOPInstanceUID
    assert clean.file_meta.MediaStorageSOPInstanceUID == clean.SOPInstanceUID
    assert clean.SOPInstanceUID == pseudonymise(original, "salt")
    assert report.private_tags_removed == 2
    assert len(report.as_dict()["warnings"]) >= 2
    assert np.array_equal(clean.pixel_array, ds.pixel_array)


def test_partial_slots_are_masked_not_priors(tmp_path):
    sanitize_upload(encoded(dicom()), tmp_path, "salt", 0)
    series, _ = read_study(tmp_path, "salt")
    resolution = resolve_slots(series)
    assert resolution["presentCount"] == 1
    assert resolution["slots"][0]["present"]
    assert resolution["usingFallbackPriors"] is False
    images, mask = engine.prepare_images(series, resolution)
    assert images.shape == (6, 4, 3, 336, 336)
    assert mask.shape == (6, 4)
    assert mask[0].any() and not mask[1:].any()
    assert torch.isfinite(images).all() and (images[~mask] == 0).all()


def test_fat_suppression_no_substring_guess():
    ds = Dataset()
    ds.SeriesDescription = "SAG PD NOFS"
    assert _infer_fat_suppression(ds) == 0
    ds.SeriesDescription = "SAG PD"
    assert _infer_fat_suppression(ds) is None


def test_generated_upload_studies_fill_every_declared_slot(tmp_path):
    """The selectable course folders must resolve all slots they claim.

    T1 series carry no fat-suppression signal unless stated, so without an
    explicit marker they fall out of the six slots and silently reduce
    coverage. Regression guard for the generated studies shipped to students.
    """
    import json

    import generate_upload_studies

    root = tmp_path / "upload-studies"
    records = generate_upload_studies.generate(root)
    assert json.loads((root / "manifest.json").read_text()) == records

    for record in records:
        staged = tmp_path / f"staged-{record['folder']}"
        staged.mkdir()
        sources = sorted((root / record["folder"]).rglob("*.dcm"))
        assert len(sources) == record["files"]
        for index, source in enumerate(sources):
            sanitize_upload(source.read_bytes(), staged, "salt", index)
        series, _ = read_study(staged, "salt")
        resolution = resolve_slots(series)
        assert resolution["presentCount"] == record["slots"], (
            record["folder"], resolution["unassignedSeries"]
        )
        assert resolution["laterality"] == record["laterality"]
        assert resolution["usingFallbackPriors"] is False
        images, mask = engine.prepare_images(series, resolution)
        assert images.shape == (6, 4, 3, 336, 336)
        assert int(mask.any(dim=1).sum()) == record["slots"]


def test_health_calibration_and_federated(client, monkeypatch, tmp_path):
    response = client.get("/api/health").json()
    assert response["modelReady"] and response["availableFolds"] == list(range(5))
    payload = client.get("/api/calibration").json()
    assert len(payload["labels"]) == 12
    assert payload["operatingPoints"] is None
    assert all("sensitivity" not in label and "specificity" not in label for label in payload["labels"])
    monkeypatch.setattr(main, "FEDERATED_PATH", tmp_path / "run.json")
    assert client.get("/api/federated").status_code == 404
    main.FEDERATED_PATH.write_text('{"status":"complete"}')
    assert client.get("/api/federated").json() == {"status": "complete"}
    assert client.get("/openapi.json").status_code == 200


def test_stream_real_preprocess_small_model(client, fake_model):
    response = upload(client, [dicom(), dicom()])
    assert response.status_code == 202, response.text
    job_id = response.json()["jobId"]
    root = main.JOBS[job_id].root
    streamed = client.get(f"/api/studies/{job_id}/stream?folds=0,1,2,3,4")
    assert streamed.status_code == 200
    received = events(streamed)
    assert received[-1]["status"] == "done", received
    result = received[-1]["result"]
    assert len(result["predictions"]) == 12
    assert result["folds"] == [0, 1, 2, 3, 4]
    assert result["slotResolution"]["presentCount"] == 1
    assert result["deidentification"]["filesProcessed"] == 2
    assert {e["stage"] for e in received} == {"intake", "deidentify", "slots", "preprocess", "features", "inference", "report"}
    forward = next(i for i, e in enumerate(received) if "extracting DINO" in e.get("detail", ""))
    features_done = next(i for i, e in enumerate(received) if e["stage"] == "features" and e["status"] == "done")
    assert forward < features_done
    for prediction in result["predictions"]:
        assert 0 <= prediction["probability"] <= 100 and prediction["usingPrior"] is False
        weights = prediction["slotAttention"]
        assert len(weights) == 6
        assert sum(w["weight"] for w in weights) == pytest.approx(1)
        assert weights[0]["weight"] == pytest.approx(1)
        assert all(w["weight"] == 0 for w in weights[1:])
    assert not root.exists() and job_id not in main.JOBS
    assert "SECRET" not in streamed.text
    assert client.get(f"/api/studies/{job_id}/stream").status_code == 404


@pytest.mark.parametrize("datasets", [
    lambda: [dicom(), dicom(study="2.3.4")],
    lambda: [dicom(), dicom(side="L")],
    lambda: [dicom(instance="1.2.3.99"), dicom(instance="1.2.3.99")],
])
def test_reject_mixed_and_duplicate_studies(client, datasets):
    response = upload(client, datasets())
    assert response.status_code == 422 and "SECRET" not in response.text
    assert not main.JOBS


@pytest.mark.parametrize("folds", ["", "5", "0,0", "-1", "0,x", "0,1,2,3,4,0"])
def test_invalid_folds(client, folds):
    assert client.get(f"/api/studies/missing/stream?folds={folds}").status_code == 422


def test_upload_bounds_and_no_raw_spool(client, monkeypatch):
    assert client.post("/api/studies", content=b"x").status_code == 415
    assert client.post("/api/studies", files={"files": ("secret.dcm", b"SECRET")}).status_code == 422
    assert upload(client, [dicom(rows=2049)]).status_code == 422
    monkeypatch.setattr(main, "MAX_FILE_BYTES", 128)
    assert upload(client).status_code == 413
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 128)
    assert upload(client).status_code == 413
    assert not main.JOBS


def test_file_and_queue_limits(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_FILES", 1)
    assert upload(client, [dicom(), dicom()]).status_code == 413
    monkeypatch.setattr(main, "MAX_JOBS", 1)
    job_id = upload(client).json()["jobId"]
    assert upload(client).status_code == 429
    job = main.JOBS[job_id]
    main.expire(job_id, job)
    assert not job.root.exists() and job_id not in main.JOBS


def test_unknown_laterality_is_not_assumed_right(client):
    response = upload(client, [dicom(side="")])
    job_id = response.json()["jobId"]
    received = events(client.get(f"/api/studies/{job_id}/stream"))
    assert received[-1]["status"] == "error"
    assert "Laterality unresolved" in received[-1]["message"]
    assert all("result" not in e for e in received)


def test_no_slots_and_failure_no_fabricated_predictions(client, monkeypatch):
    job_id = upload(client, [dicom(description="SCOUT")]).json()["jobId"]
    received = events(client.get(f"/api/studies/{job_id}/stream"))
    assert received[-1]["status"] == "error" and all("result" not in e for e in received)
    job_id = upload(client).json()["jobId"]
    root = main.JOBS[job_id].root

    def fail(*args):
        raise RuntimeError("SECRET-PATIENT")

    monkeypatch.setattr(main, "prepare_images", fail)
    response = client.get(f"/api/studies/{job_id}/stream")
    assert events(response)[-1]["status"] == "error"
    assert "SECRET" not in response.text and not root.exists()


def test_claim_conflict_and_unavailable_models(client, monkeypatch):
    job_id = upload(client).json()["jobId"]
    main.JOBS[job_id].claimed = True
    assert client.get(f"/api/studies/{job_id}/stream").status_code == 409
    monkeypatch.setattr(main.REGISTRY, "is_ready", lambda: False)
    assert client.get(f"/api/studies/{job_id}/stream").status_code == 503


def test_cancelled_worker_cleans_files(client):
    job_id = upload(client).json()["jobId"]
    job = main.JOBS[job_id]
    job.cancel.set()
    output = queue.Queue()
    main.produce(job_id, job, [0], output)
    assert not job.root.exists()
    assert output.get_nowait() is None


def test_disconnect_during_work_defers_safe_cleanup(client, monkeypatch):
    import asyncio
    from starlette.requests import Request

    entered, release = threading.Event(), threading.Event()
    original = main.prepare_images

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(main, "prepare_images", blocked)
    job_id = upload(client).json()["jobId"]
    job = main.JOBS[job_id]

    async def exercise():
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}
        request = Request({"type": "http", "method": "GET", "path": "/", "headers": []}, receive)
        response = await main.stream(job_id, request)
        iterator = response.body_iterator
        await anext(iterator)
        await iterator.aclose()

    asyncio.run(exercise())
    assert job.cancel.is_set()
    release.set()
    deadline = time.monotonic() + 5
    while job_id in main.JOBS and time.monotonic() < deadline:
        time.sleep(0.01)
    assert job_id not in main.JOBS and not job.root.exists()


def test_forward_and_captured_attention_are_serialized(monkeypatch, fake_model):
    active, max_active = 0, 0
    state_lock = threading.Lock()
    original = fake_model.forward

    def forward(*args):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        try:
            return original(*args)
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(fake_model, "forward", forward)
    # Ungrouped path avoids counting recursive coverage-group forward calls
    # as if they were two concurrent requests.
    images = torch.zeros(6, 3, 4, 4)
    mask = torch.ones(6, dtype=torch.bool)
    results, failures = [], []

    def run():
        try:
            results.append(engine.run_ensemble(images, mask, [0]))
        except Exception as exc:
            failures.append(type(exc).__name__)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert not failures and len(results) == 2 and max_active == 1


def test_hash_cannot_be_disabled(monkeypatch):
    registry = engine.EnsembleRegistry()
    monkeypatch.setattr(engine, "sha256_file", lambda path: "invalid")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        registry.load(0)
    with pytest.raises(TypeError):
        registry.load(0, verify_hash=False)
    with pytest.raises(ValueError):
        registry.load(5)


@pytest.mark.skipif(os.environ.get("KNEESCAN_REAL_SMOKE") != "1", reason="Opt-in local 89MB real checkpoint smoke")
def test_real_checkpoint_load_and_forward():
    torch.set_num_threads(2)
    registry = engine.EnsembleRegistry()
    model = registry.load(0)
    assert registry.loaded_folds == [0] and model is registry.load(0)
    # Real deployment image size and four groups, one present slot. This
    # validates trained DINO/head wiring, not clinical accuracy.
    images = torch.zeros(1, 6, 4, 3, 336, 336)
    mask = torch.zeros(1, 6, 4, dtype=torch.bool)
    mask[:, 0] = True
    with torch.inference_mode():
        logits = model(images, mask)
    assert logits.shape == (1, 12) and torch.isfinite(logits).all()
    assert model.head.last_attention.shape == (4, 12, 6)
    assert (model.head.last_attention[:, :, 1:] == 0).all()


def test_saliency_failure_never_discards_predictions(client, fake_model, monkeypatch):
    """A visualization problem must not turn a valid study into an error.

    The fake backbone exposes no patch tokens, so compute_saliency raises.
    Predictions must still stream through with an empty saliency list.
    """
    job_id = upload(client).json()["jobId"]
    streamed = client.get(f"/api/studies/{job_id}/stream?folds=0")
    received = events(streamed)
    assert received[-1]["status"] == "done", received
    result = received[-1]["result"]
    assert len(result["predictions"]) == 12
    assert all(p["saliency"] == [] for p in result["predictions"])
    assert result["saliencyGridSize"] == 0
    # Decoded slot previews come from pixels, not the model, so they survive.
    assert result["previews"] and result["previews"][0]["png"]


@pytest.mark.skipif(not os.environ.get("KNEESCAN_REAL_SMOKE"), reason="opt-in real model run")
def test_real_saliency_is_spatial_and_label_specific():
    """Saliency must be real patch attribution, not a constant or a copy."""
    import numpy as np

    from generate_upload_studies import generate
    from study_ingest import read_study, resolve_slots, sanitize_upload

    with tempfile.TemporaryDirectory() as temp:
        root = pathlib.Path(temp)
        generate(root / "studies")
        staged = root / "staged"
        staged.mkdir()
        sources = sorted((root / "studies" / "01-complete-right-synthetic").rglob("*.dcm"))
        for index, source in enumerate(sources):
            sanitize_upload(source.read_bytes(), staged, "salt", index)
        series, _ = read_study(staged, "salt")
        resolution = resolve_slots(series)
        images, mask = engine.prepare_images(series, resolution)
        grids, slots, size = engine.compute_saliency(images, mask, 0)

    assert size == 24 and grids.shape == (12, 6, 24, 24) and slots == list(range(6))
    assert np.isfinite(grids).all() and grids.min() >= 0 and grids.max() <= 1
    # Spatially varying, not a flat map.
    assert all(grids[label, 0].std() > 0.01 for label in range(12))
    # Different labels must not produce identical maps.
    assert not np.allclose(grids[0, 0], grids[11, 0])
