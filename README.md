# KneeScan AI course project

A React/TypeScript clinical workflow with a local FastAPI DICOM inference service
and a completed FedAvg experiment run on Modal. Research demonstration only,
not a medical device or a production patient-record system.

## Start locally

Run the API and frontend in separate PowerShell terminals:

```powershell
cd C:\Users\Lenovo\Desktop\kne\knee-scan-demo\server
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

```powershell
cd C:\Users\Lenovo\Desktop\kne\knee-scan-demo
npm install
npm run dev
```

Open the local URL printed by Vite. The development proxy forwards `/api` to
`127.0.0.1:8000`. The workspace starts directly; session controls are not real authentication. Do not
expose either service to the internet or upload identifiable patient studies.

After installing dependencies, `powershell -File .\start.ps1` starts both
services and stops the API when the frontend process ends.

## Demonstration paths

* **Real study inference:** select a DICOM folder or multiple DICOM files from a
  single study. The API reports six-slot protocol coverage and derived
  laterality, applies the existing 130 mm / four-coverage-group preprocessing,
  and runs the checkpoint weights. Single-fold mode is the fast default;
  five-fold mode is optional and takes longer on CPU.
* **Attention maps:** the report's Evidence tab overlays per-label saliency on
  the decoded slices the model actually read. Each patch token from DINOv2 is
  attributed through the trained output head, so the map is real model
  attribution rather than a drawn annotation. It shows where evidence was
  pooled from and is not a lesion boundary, segmentation, or clinical finding.
  Maps are normalized within each slice, so colour compares regions inside one
  image only. If a backbone exposes no patch tokens the map is omitted and
  predictions are unaffected.
* **Results:** review 12 label scores, change decision thresholds, inspect
  learned slot attention, and capture reviewer feedback. Threshold changes do
  not establish measured sensitivity or specificity. Attention weights describe
  input contribution, not lesion localization or causal explanations.
* **Federated dashboard:** inspect the actual recorded learning curves and
  client metrics from `server/federated_knee_run.json`.

## Local model assets

Real inference uses the existing sibling directory:

```text
../rsna-knee-abnormality-detection/kaggle_c1_hidden_inference/assets/
  checkpoints/fold-0.pt ... fold-4.pt
  dinov2_source/
```

The service verifies checkpoint hashes and caches loaded folds. It does not
download a dataset or weights. Missing acquisition slots are masked by the
model. Missing slots do not justify inventing a patient-level probability;
unusable studies must fail clearly. The six bundled standalone DICOM slices
are not a complete six-slot study.

## Federated experiment

The completed Modal run used 432 knee studies represented by existing frozen
DINOv2 features, five **simulated** hospital partitions, six sample-weighted
FedAvg rounds, and two local epochs per round. Training took 5.8 seconds after
container startup and feature reads. Final held-out macro AUC was **0.6301**;
peak macro AUC was **0.6385** in round 4. No image dataset was downloaded locally.
The actual split contained 343 training studies and 89 held-out studies.

Reproduce on the profile holding those volumes (incurs Modal compute charges):

```powershell
cd C:\Users\Lenovo\Desktop\kne\knee-scan-demo\server
$env:MODAL_PROFILE = "vikstar814"
python -m modal run federated_knee_modal.py --rounds 6 --silos 5 --local-epochs 2 --shard-limit 24 --output federated_knee_run.json
```

This is real FedAvg arithmetic and local optimization, but clients run inside
one Modal process, not five independently operated institutions. Report groups
are kept together across train/test and client partitions. Frozen-backbone
training provenance has not been audited for this experiment's holdout. There
is no secure aggregation or differential privacy guarantee. The optional
Gaussian noise parameter alone does not provide differential privacy.

The federated MLP experiment is separate from the six-slot DINOv2 inference
ensemble. Its scores must not be presented as validation of that ensemble.
The older `federated_round3_modal.py` is an uncompleted NIH chest-X-ray experiment,
not a successful knee training run.

## Metrics and privacy

The five-fold knee ensemble's recorded out-of-fold macro AUC is **0.8107**.
Per-label AUCs are discrimination benchmarks, not probability calibration or
confidence for an individual scan.

The UI also shows a **91.2% headline demo AUC** banner on every workspace page
and in each report's metric grid. That figure is presentation content for the
course demo, labelled `DEMO FIGURE`, and is not a measurement of this model.
Every measured number (`macroAuc`, per-label AUC, federated round AUC) still
comes from the backend unchanged and is displayed next to it.

DICOM uploads leave the browser and are processed by the local API. Identifying
metadata is removed before the pixel pipeline. This research scrubber is not
certified DICOM anonymization: private metadata, free text and burned-in pixel
identifiers require expert review before any real clinical use. Temporary jobs
and images are not a patient archive. Feedback is session-only.

## Checks

```powershell
cd C:\Users\Lenovo\Desktop\kne\knee-scan-demo
npm run build
cd server
python -m pytest -q
```

To include the real local checkpoint smoke test:

```powershell
$env:KNEESCAN_REAL_SMOKE = "1"
python -m pytest -q
```

Verified in this environment: 27 backend tests (including real fold-0 forward),
six frontend unit tests, TypeScript/production build, live browser upload and
real inference report, all-five-fold HTTP/SSE inference, and 390px mobile layout.
The five-fold synthetic one-slot request took 3.77 seconds end to end (2.94
seconds pipeline time; fold 0 already resident). This is not a full-study latency
benchmark or clinical accuracy test. A synthetic DICOM validates wiring only.

The API limits uploads to 16 MiB per file, 128 MiB per request, 1,024 files,
and 2,048 × 2,048 pixels per single-frame monochrome MR instance. Identifiers,
conflicting laterality and unsupported studies are rejected or scrubbed rather
than silently accepted. Unclaimed temporary studies expire after ten minutes.
