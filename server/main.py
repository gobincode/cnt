"""Local-only research inference API. Run: uvicorn main:app --host 127.0.0.1.

Jobs are bearer capabilities: keep their random IDs private. No authentication
or persistent clinical storage is provided; do not expose this demo publicly.
"""
from __future__ import annotations

import asyncio
import json
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from multipart import MultipartParser
from multipart.multipart import parse_options_header

from calibration import LABELS, MACRO_AUC, MEASURED_AUC, calibration_payload, load_federated_summary
from deidentify import DeidentificationReport
from inference_engine import (REGISTRY, UnsupportedSaliency, compute_saliency, prepare_images,
                              run_ensemble, slot_preview)
from study_ingest import SLOT_NAMES, read_study, resolve_slots, sanitize_upload

MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_UPLOAD_BYTES = 128 * 1024 * 1024
MAX_FILES = 1024
MAX_JOBS = 4
JOB_TTL_SECONDS = 600
FEDERATED_PATH = Path(__file__).with_name("federated_knee_run.json")
JOBS_LOCK = threading.Lock()
WORK_LOCK = threading.Lock()
JOBS: dict[str, "Job"] = {}
RESERVATIONS = 0


@dataclass
class Job:
    root: Path
    report: DeidentificationReport
    series: list
    created: float = field(default_factory=time.monotonic)
    claimed: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)
    timer: threading.Timer | None = None


def discard(job_id: str, job: Job) -> None:
    if job.timer:
        job.timer.cancel()
    shutil.rmtree(job.root, ignore_errors=True)
    with JOBS_LOCK:
        if JOBS.get(job_id) is job:
            JOBS.pop(job_id, None)


def expire(job_id: str, job: Job) -> None:
    with JOBS_LOCK:
        if job.claimed or JOBS.get(job_id) is not job:
            return
        job.claimed = True
    discard(job_id, job)


@asynccontextmanager
async def lifespan(app):
    yield
    with JOBS_LOCK:
        jobs = list(JOBS.items())
    for job_id, job in jobs:
        job.cancel.set()
        if not job.claimed:
            discard(job_id, job)


app = FastAPI(title="KneeScan local research inference", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"], allow_credentials=False,
)


class UploadProblem(Exception):
    def __init__(self, status: int, message: str):
        self.status, self.message = status, message


class UploadReceiver:
    """Streaming multipart receiver: never uses UploadFile's raw-disk spool."""
    def __init__(self, root: Path, boundary: bytes):
        self.root, self.salt = root, uuid.uuid4().hex
        self.count, self.part_count, self.header_bytes = 0, 0, 0
        self.data = bytearray()
        self.field, self.value = bytearray(), bytearray()
        self.disposition = b""
        self.report = DeidentificationReport()
        self.finished = False
        self.parser = MultipartParser(boundary, {
            "on_part_begin": self.part_begin, "on_header_field": self.header_field,
            "on_header_value": self.header_value, "on_header_end": self.header_end,
            "on_headers_finished": self.headers_finished, "on_part_data": self.part_data,
            "on_part_end": self.part_end, "on_end": self.end,
        })

    def part_begin(self):
        self.part_count += 1
        if self.part_count > MAX_FILES:
            raise UploadProblem(413, "Too many files.")
        self.data.clear()
        self.disposition = b""
        self.header_bytes = 0

    def header_field(self, data, start, end):
        self.header_bytes += end - start
        if self.header_bytes > 8192:
            raise UploadProblem(413, "Multipart headers too large.")
        self.field.extend(data[start:end])

    def header_value(self, data, start, end):
        self.header_bytes += end - start
        if self.header_bytes > 8192:
            raise UploadProblem(413, "Multipart headers too large.")
        self.value.extend(data[start:end])

    def header_end(self):
        if bytes(self.field).lower() == b"content-disposition":
            self.disposition = bytes(self.value)
        self.field.clear()
        self.value.clear()

    def headers_finished(self):
        kind, options = parse_options_header(self.disposition)
        if kind != b"form-data" or b"filename" not in options or options.get(b"name") not in {b"files", b"files[]"}:
            raise UploadProblem(422, "Provide DICOM files in the multipart files field.")

    def part_data(self, data, start, end):
        if len(self.data) + end - start > MAX_FILE_BYTES:
            raise UploadProblem(413, "DICOM file exceeds upload limit.")
        self.data.extend(data[start:end])

    def part_end(self):
        try:
            report = sanitize_upload(bytes(self.data), self.root, self.salt, self.count)
        except ValueError:
            raise UploadProblem(422, "Invalid or unsupported DICOM; only single-frame monochrome MR slices are accepted.") from None
        finally:
            self.data.clear()
        self.report.merge(report)
        self.count += 1

    def end(self):
        self.finished = True


@app.get("/api/health")
def health():
    return {"modelReady": REGISTRY.is_ready(), "availableFolds": REGISTRY.available_folds,
            "loadedFolds": REGISTRY.loaded_folds}


@app.get("/api/calibration")
def calibration():
    return calibration_payload()


@app.get("/api/federated")
def federated():
    payload = load_federated_summary(FEDERATED_PATH)
    if payload is None:
        raise HTTPException(404, "No completed federated run is available.")
    return payload


@app.post("/api/studies", status_code=202, openapi_extra={"requestBody": {
    "required": True, "content": {"multipart/form-data": {"schema": {"type": "object", "required": ["files"],
    "properties": {"files": {"type": "array", "items": {"type": "string", "format": "binary"}}}}}}}})
async def upload(request: Request):
    global RESERVATIONS
    kind, options = parse_options_header(request.headers.get("content-type", ""))
    boundary = options.get(b"boundary", b"")
    if kind != b"multipart/form-data" or not (0 < len(boundary) <= 200):
        raise HTTPException(415, "Expected multipart/form-data with DICOM files.")
    length = request.headers.get("content-length")
    if length and (not length.isdigit() or int(length) > MAX_UPLOAD_BYTES):
        raise HTTPException(413, "Upload exceeds byte limit.")
    with JOBS_LOCK:
        if len(JOBS) + RESERVATIONS >= MAX_JOBS:
            raise HTTPException(429, "Local study queue is full; retry after a job finishes.")
        RESERVATIONS += 1
    root = None
    accepted = False
    try:
        root = Path(tempfile.mkdtemp(prefix="kneescan-sanitized-"))
        receiver = UploadReceiver(root, boundary)
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise UploadProblem(413, "Upload exceeds byte limit.")
            await anyio.to_thread.run_sync(receiver.parser.write, chunk)
        receiver.parser.finalize()
        if not receiver.finished or receiver.count == 0:
            raise UploadProblem(422, "Incomplete or empty DICOM upload.")
        try:
            series, _ = await anyio.to_thread.run_sync(read_study, root, receiver.salt)
        except Exception:
            raise UploadProblem(422, "Upload must contain unique DICOM instances from exactly one study and one knee.") from None
        job_id = uuid.uuid4().hex
        job = Job(root, receiver.report, series)
        job.timer = threading.Timer(JOB_TTL_SECONDS, expire, args=(job_id, job))
        job.timer.daemon = True
        with JOBS_LOCK:
            JOBS[job_id] = job
        job.timer.start()
        accepted = True
        return {"jobId": job_id}
    except UploadProblem as exc:
        raise HTTPException(exc.status, exc.message) from None
    except Exception:
        raise HTTPException(422, "DICOM upload could not be processed.") from None
    finally:
        with JOBS_LOCK:
            RESERVATIONS -= 1
        if not accepted and root is not None:
            shutil.rmtree(root, ignore_errors=True)


def event(stage, status, **extra):
    return {"stage": stage, "label": stage.capitalize(), "status": status, **extra}


def produce(job_id: str, job: Job, folds: list[int], output: queue.Queue):
    started = time.monotonic()
    stage, locked = "intake", False

    def emit(stage, status, **extra):
        if job.cancel.is_set():
            raise InterruptedError()
        output.put(event(stage, status, **extra))

    try:
        emit("intake", "done", detail=f"Accepted {job.report.files_processed} DICOM slices from one study.")
        emit("deidentify", "done", detail="Sanitized metadata persisted before pixel decoding; original bytes were not saved.", deidentification=job.report.as_dict())
        stage = "slots"
        emit(stage, "running")
        resolution = resolve_slots(job.series)
        emit(stage, "done", slotResolution=resolution)
        if not resolution["presentCount"]:
            emit(stage, "error", message="No usable protocol slots; no patient predictions were produced.")
            return
        if resolution["laterality"] not in {"L", "R"}:
            emit(stage, "error", message="Laterality unresolved; no patient predictions were produced.")
            return
        stage = "preprocess"
        emit(stage, "running", detail="Waiting for serialized local processing, then 130 mm crop and four coverage groups.")
        while not job.cancel.is_set():
            if WORK_LOCK.acquire(timeout=0.2):
                locked = True
                break
        if not locked:
            return
        if job.cancel.is_set():
            return
        images, mask = prepare_images(job.series, resolution)
        # Stream the decoded slots so the operator sees the actual pixels that
        # were cropped, not a placeholder, while the model is still running.
        previews = []
        for slot_index, slot_name in enumerate(SLOT_NAMES):
            groups = [g for g in range(mask.shape[1]) if bool(mask[slot_index, g])]
            if not groups:
                continue
            previews.append({"slot": slot_name, "slotIndex": slot_index,
                             "png": slot_preview(images, mask, slot_index, groups[0])})
            emit(stage, "running", detail=f"Decoded {slot_name} at 336 pixels.",
                 preview=previews[-1])
        emit(stage, "done", detail="Prepared six masked slots × four coverage groups at 336 pixels.",
             previews=previews)
        stage = "features"
        emit("features", "running", detail="DINO feature extraction occurs inside each fold forward pass; extraction and head inference are combined.")
        emit("inference", "running", detail="Real fold forward passes; no prior substitution for missing slots.")

        def progress(fold, state):
            emit("features" if state == "forward" else "inference", "running",
                 detail=f"Fold {fold}: " + {"loading": "verifying/loading local checkpoint", "forward": "extracting DINO features and evaluating trained attention head", "done": "forward pass complete"}[state])

        probabilities, attention = run_ensemble(images, mask, folds, progress=progress)
        emit("features", "done", detail="Actual per-fold feature extraction completed.")
        stage = "inference"
        emit(stage, "done", detail=f"Averaged {len(folds)} real fold probabilities.")
        stage = "report"
        emit(stage, "running", detail="Attributing patch tokens through the trained head.")
        # Spatial attribution runs on the lowest selected fold: the pooled
        # production forward pass keeps no patch tokens to explain. An
        # explanation failure must never discard valid predictions.
        grids, saliency_slots, grid_size = None, [], 0
        try:
            grids, saliency_slots, grid_size = compute_saliency(images, mask, min(folds))
            emit(stage, "running",
                 detail=f"Built {grid_size}×{grid_size} saliency grids for {len(LABELS)} labels.")
        except (UnsupportedSaliency, RuntimeError, ValueError):
            emit(stage, "running", detail="Spatial attribution unavailable for this model; predictions are unaffected.")
        predictions = [{"label": label, "probability": float(probabilities[i]) * 100,
                        "auc": MEASURED_AUC[label], "status": "model-output", "usingPrior": False,
                        "slotAttention": [{"slot": slot, "weight": float(attention[i, j])}
                                          for j, slot in enumerate(SLOT_NAMES)],
                        "saliency": [] if grids is None else
                                    [{"slot": SLOT_NAMES[slot_index],
                                      "slotIndex": slot_index,
                                      "grid": [round(float(v), 4) for v in grids[i, j].ravel()]}
                                     for j, slot_index in enumerate(saliency_slots)]}
                       for i, label in enumerate(LABELS)]
        result = {"predictions": predictions, "slotResolution": resolution,
                  "deidentification": job.report.as_dict(), "folds": folds, "macroAuc": MACRO_AUC,
                  "elapsedSeconds": round(time.monotonic() - started, 3),
                  "previews": previews,
                  "saliencyGridSize": grid_size,
                  "attentionNote": "Trained head slot weights, not spatial localization or clinical confidence.",
                  "saliencyNote": "Patch-token attribution through the trained output head. Shows where evidence was pooled from, not a lesion boundary or segmentation.",
                  "validationNote": "AUC is five-fold out-of-fold validation, not measured performance of this fold subset or patient confidence."}
        emit(stage, "done", result=result)
    except InterruptedError:
        pass
    except Exception:
        # Never stringify decoder/model exceptions: they can contain identifiers.
        if not job.cancel.is_set():
            output.put(event(stage, "error", message="Study processing failed safely; no patient predictions were produced."))
    finally:
        if locked:
            WORK_LOCK.release()
        discard(job_id, job)
        output.put(None)


@app.get("/api/studies/{job_id}/stream")
async def stream(job_id: str, request: Request, folds: str = "0"):
    if not re.fullmatch(r"[0-4](?:,[0-4])*", folds) or len(folds) > 9:
        raise HTTPException(422, "Folds must be distinct comma-separated indices from 0 through 4.")
    selected = [int(fold) for fold in folds.split(",")]
    if len(set(selected)) != len(selected):
        raise HTTPException(422, "Duplicate folds are not allowed.")
    if not REGISTRY.is_ready() or any(fold not in REGISTRY.available_folds for fold in selected):
        raise HTTPException(503, "Requested local model assets are unavailable.")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Study job not found or expired.")
        if job.claimed:
            raise HTTPException(409, "Study stream has already been claimed.")
        if time.monotonic() - job.created >= JOB_TTL_SECONDS:
            raise HTTPException(410, "Study job has expired.")
        job.claimed = True
        if job.timer:
            job.timer.cancel()

    async def events():
        output = queue.Queue()
        worker = threading.Thread(target=produce, args=(job_id, job, selected, output), daemon=True)
        worker.start()
        heartbeat = time.monotonic()
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = output.get_nowait()
                except queue.Empty:
                    if time.monotonic() - heartbeat > 10:
                        yield ": keep-alive\n\n"
                        heartbeat = time.monotonic()
                    await asyncio.sleep(0.05)
                    continue
                if payload is None:
                    break
                yield "data: " + json.dumps(payload, allow_nan=False) + "\n\n"
        finally:
            # Native torch forward cannot be interrupted safely. Its worker owns
            # cleanup and deletes the sanitized directory immediately on return.
            job.cancel.set()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
