"""Study ingest: parse an uploaded DICOM study into the six protocol slots.

The served model is a six-slot study encoder, so a single image cannot produce
a 12-label prediction. This module reproduces the inference-time contract:
classify every series into a slot, resolve laterality, and report exactly which
slots were found so the UI can say what the prediction is actually based on.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from deidentify import DeidentificationReport, deidentify_dataset

SLOT_NAMES: tuple[str, ...] = (
    "SAG_FLUID_FS",
    "COR_FLUID_FS",
    "AX_FLUID_FS",
    "SAG_FLUID_NOFS",
    "COR_T1",
    "SAG_T1",
)

SLOT_DESCRIPTIONS: dict[str, str] = {
    "SAG_FLUID_FS": "Sagittal fluid-sensitive, fat-suppressed",
    "COR_FLUID_FS": "Coronal fluid-sensitive, fat-suppressed",
    "AX_FLUID_FS": "Axial fluid-sensitive, fat-suppressed",
    "SAG_FLUID_NOFS": "Sagittal fluid-sensitive, no fat suppression",
    "COR_T1": "Coronal T1",
    "SAG_T1": "Sagittal T1",
}

_RSNA_SRC = (
    Path(__file__).resolve().parents[2]
    / "rsna-knee-abnormality-detection"
    / "kaggle_c1_hidden_inference"
    / "assets"
    / "source"
)


def _ensure_rsna_on_path() -> None:
    if str(_RSNA_SRC) not in sys.path and _RSNA_SRC.is_dir():
        sys.path.insert(0, str(_RSNA_SRC))


@dataclass
class IngestedSeries:
    """One series after de-identification, ready for slot classification."""

    series_uid: str
    files: list[Path]
    metadata: Any
    geometry: Any
    plane: str | None
    fluid_sensitive: int | None
    fat_suppression: int | None


def _tuple_value(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split("\\") if item.strip())
    try:
        return tuple(str(item).strip() for item in value if str(item).strip())
    except TypeError:
        text = str(value).strip()
        return (text,) if text else ()


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _vector(dataset: Any, name: str, length: int):
    import numpy as np

    value = getattr(dataset, name, None)
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    return result if result.size == length and np.isfinite(result).all() else None


def _infer_plane(dataset: Any) -> str | None:
    """Recover the acquisition plane from orientation when no tag states it."""
    import numpy as np

    orientation = _vector(dataset, "ImageOrientationPatient", 6)
    if orientation is None:
        import re
        text = str(getattr(dataset, "SeriesDescription", "")).upper()
        hints = [plane for token, plane in ((r"\bSAG(?:ITTAL)?\b", "Sagittal"), (r"\bCOR(?:ONAL)?\b", "Coronal"), (r"\bAX(?:IAL)?\b", "Axial")) if re.search(token, text)]
        return hints[0] if len(hints) == 1 else None
    normal = np.cross(orientation[:3], orientation[3:])
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return None
    normal = np.abs(normal / norm)
    axis = int(np.argmax(normal))
    return {0: "Sagittal", 1: "Coronal", 2: "Axial"}[axis] if normal[axis] >= 0.9 else None


def _infer_fluid_sensitive(dataset: Any) -> int | None:
    """T2-weighted or similar long-TE sequences are the fluid-sensitive ones."""
    echo = _optional_float(getattr(dataset, "EchoTime", None))
    repetition = _optional_float(getattr(dataset, "RepetitionTime", None))
    if echo is None or repetition is None:
        return None
    return 1 if echo >= 40.0 and repetition >= 1200.0 else None


def _infer_fat_suppression(dataset: Any) -> int | None:
    options = {str(value).strip().upper() for value in _tuple_value(getattr(dataset, "ScanOptions", None))}
    description = str(getattr(dataset, "SeriesDescription", "") or "").upper()
    import re
    if re.search(r"\b(?:NOFS|NONFS|NON FS)\b", description):
        return 0
    if options & {"FS", "FATSAT", "FAT_SAT", "FSAT"} or re.search(r"\b(?:FS|FATSAT|STIR|SPAIR|SPIR)\b", description):
        return 1
    return None


def sanitize_upload(raw: bytes, root: Path, salt: str, index: int):
    """Only sanitized datasets may be persisted; raw bytes never touch disk."""
    import io
    import warnings
    import pydicom

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # malformed values must not leak PHI into logs
        try:
            ds = pydicom.dcmread(io.BytesIO(raw), force=False)
            required = ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "SOPClassUID")
            if any(not getattr(ds, key, None) for key in required):
                raise ValueError()
            rows, cols = int(ds.Rows), int(ds.Columns)
            if not (0 < rows <= 2048 and 0 < cols <= 2048):
                raise ValueError()
            if (int(getattr(ds, "NumberOfFrames", 1)) != 1
                    or int(getattr(ds, "SamplesPerPixel", 1)) != 1
                    or int(ds.BitsAllocated) not in (8, 16, 32)
                    or "PixelData" not in ds or str(ds.Modality) != "MR"):
                raise ValueError()
            report = deidentify_dataset(ds, salt)
            path = root / f"{index:06d}.dcm"
            ds.save_as(path, enforce_file_format=True)
            return report
        except Exception:
            raise ValueError("Invalid or unsupported DICOM; expected bounded single-frame monochrome MR slices.") from None

def read_study(
    upload_root: Path, salt: str
) -> tuple[list[IngestedSeries], DeidentificationReport]:
    """Read only the sanitized DICOM directory produced by sanitize_upload."""
    import pydicom

    _ensure_rsna_on_path()
    from rsna_knee.track_b_data import SeriesMetadata, SliceGeometry

    report = DeidentificationReport()
    by_series: dict[str, list[tuple[Path, Any]]] = {}

    candidates = sorted(upload_root.glob("*.dcm"))
    studies: set[str] = set()
    identities: set[str] = set()
    lateralities: set[str] = set()
    for path in candidates:
        dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
        study_uid = str(dataset.StudyInstanceUID)
        series_uid = str(dataset.SeriesInstanceUID)
        sop_uid = str(dataset.SOPInstanceUID)
        if sop_uid in identities:
            raise ValueError("Duplicate DICOM instance.")
        identities.add(sop_uid)
        studies.add(study_uid)
        if len(studies) != 1:
            raise ValueError("Upload must contain exactly one study.")
        for keyword in ("ImageLaterality", "Laterality"):
            side = str(getattr(dataset, keyword, "")).upper()
            if side in {"R", "L"}:
                lateralities.add(side)
        if len(lateralities) > 1:
            raise ValueError("Conflicting laterality tags; upload one knee only.")
        by_series.setdefault(series_uid, []).append((path, dataset))
    if not by_series:
        raise ValueError("No DICOM slices found.")
    series: list[IngestedSeries] = []
    for series_uid, entries in by_series.items():
        # One SeriesInstanceUID must describe one acquisition, not inconsistent
        # protocols whose middle slice would happen to select a slot.
        for keyword in ("Rows", "Columns", "SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence"):
            if len({str(getattr(ds, keyword, "")) for _, ds in entries}) != 1:
                raise ValueError("Inconsistent metadata within a DICOM series.")
        entries.sort(key=lambda item: str(item[0]))
        middle_path, middle = entries[len(entries) // 2]
        study_uid = str(getattr(middle, "StudyInstanceUID", "") or "unknown-study")
        plane = str(getattr(middle, "_plane_hint", "") or "") or _infer_plane(middle)
        fluid = _infer_fluid_sensitive(middle)
        fat = _infer_fat_suppression(middle)
        metadata = SeriesMetadata(
            study_id=study_uid,
            series_id=series_uid,
            series_description=str(getattr(middle, "SeriesDescription", "") or ""),
            sequence_name=str(getattr(middle, "SequenceName", "") or ""),
            scan_options=_tuple_value(getattr(middle, "ScanOptions", None)),
            scanning_sequence=_tuple_value(getattr(middle, "ScanningSequence", None)),
            repetition_time_ms=_optional_float(getattr(middle, "RepetitionTime", None)),
            echo_time_ms=_optional_float(getattr(middle, "EchoTime", None)),
            public_plane=plane,
            public_fluid_sensitive=fluid,
            public_fat_suppression=fat,
            slice_count=len(entries),
        )
        geometry = SliceGeometry(
            series_id=series_uid,
            source=str(middle_path),
            image_position_patient=_vector(middle, "ImagePositionPatient", 3),
            image_orientation_patient=_vector(middle, "ImageOrientationPatient", 6),
            pixel_spacing=_vector(middle, "PixelSpacing", 2),
            rows=int(getattr(middle, "Rows", 0) or 0) or None,
            columns=int(getattr(middle, "Columns", 0) or 0) or None,
            tagged_laterality=str(
                getattr(middle, "ImageLaterality", getattr(middle, "Laterality", "")) or ""
            ),
        )
        series.append(
            IngestedSeries(
                series_uid=series_uid,
                files=[path for path, _ in entries],
                metadata=metadata,
                geometry=geometry,
                plane=plane,
                fluid_sensitive=fluid,
                fat_suppression=fat,
            )
        )
    return series, report


def resolve_slots(series: Iterable[IngestedSeries]) -> dict[str, object]:
    """Assign series to slots and report coverage, laterality and consequences."""
    _ensure_rsna_on_path()
    from rsna_knee.track_b_data import derive_laterality, select_slot_series

    series = list(series)
    if not series:
        raise ValueError("No usable DICOM series.")
    if len({item.metadata.study_id for item in series}) != 1:
        raise ValueError("Upload must contain exactly one study.")

    selected = select_slot_series([item.metadata for item in series])
    by_uid = {item.series_uid: item for item in series}
    laterality = derive_laterality([item.geometry for item in series])

    slots = []
    present_uids = set()
    for slot in SLOT_NAMES:
        chosen = selected.get(slot)
        entry = by_uid.get(chosen.series_id) if chosen is not None else None
        if entry is not None:
            present_uids.add(entry.series_uid)
        slots.append(
            {
                "slot": slot,
                "description": SLOT_DESCRIPTIONS[slot],
                "present": entry is not None,
                "seriesUid": entry.series_uid if entry else None,
                "seriesDescription": (
                    entry.metadata.series_description if entry else None
                ),
                "sliceCount": entry.metadata.slice_count if entry else 0,
                "plane": entry.plane if entry else None,
            }
        )

    present = {item["slot"] for item in slots if item["present"]}
    return {
        "slots": slots,
        "presentCount": len(present),
        "laterality": laterality.value,
        "lateralitySource": laterality.source,
        "lateralityCenterMm": laterality.center_x_mm,
        "usingFallbackPriors": False,
        "unassignedSeries": [
            {
                "seriesUid": item.series_uid,
                "seriesDescription": item.metadata.series_description,
                "plane": item.plane,
                "sliceCount": item.metadata.slice_count,
            }
            for item in series
            if item.series_uid not in present_uids
        ],
    }
