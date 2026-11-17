"""Create small, wholly synthetic DICOM folders for course workflow testing.

These analytic phantoms are NOT patient scans, diagnosis examples, or accuracy
test data. No image data is downloaded. Each selectable folder is one study.
"""
from pathlib import Path
import json

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

ROOT = Path(__file__).resolve().parents[1] / "upload-studies"
# fat_suppression is an explicit hint (None, 0, or 1); T1 series carry 0 so
# slot classification resolves them instead of rejecting "unknown fat
# suppression", while TR/TE stay consistent with their described weighting.
PROTOCOLS = (
    ("SAG_FLUID_FS", "SAG T2 FS", [0, 1, 0, 0, 0, 1], 3000, 80, 1),
    ("COR_FLUID_FS", "COR T2 FS", [1, 0, 0, 0, 0, 1], 3000, 80, 1),
    ("AX_FLUID_FS", "AX T2 FS", [1, 0, 0, 0, 1, 0], 3000, 80, 1),
    ("SAG_FLUID_NOFS", "SAG T2 NOFS", [0, 1, 0, 0, 0, 1], 3000, 80, 0),
    ("COR_T1", "COR T1 NOFS", [1, 0, 0, 0, 0, 1], 500, 12, 0),
    ("SAG_T1", "SAG T1 NOFS", [0, 1, 0, 0, 0, 1], 500, 12, 0),
)


def create_study(root: Path, name: str, side: str, slots: int) -> dict:
    folder = root / name
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite existing folder: {folder}")
    folder.mkdir(parents=True)
    study_uid = generate_uid()
    # 336 px matches the model's own input size: a smaller phantom would be
    # upscaled for display and read as blur rather than anatomy.
    y, x = np.mgrid[-1:1:336j, -1:1:336j]
    total = 0
    for slot_index, (slot, description, orientation, tr, te, fs) in enumerate(PROTOCOLS[:slots]):
        series_uid = generate_uid()
        acquisition = folder / slot
        acquisition.mkdir()
        normal = np.cross(orientation[:3], orientation[3:])
        for index in range(12):
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
            ds.SOPClassUID = MRImageStorage
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
            ds.PatientName, ds.PatientID = "SYNTHETIC^COURSE^PHANTOM", name
            ds.Modality, ds.ImageLaterality = "MR", side
            ds.SeriesDescription = description
            ds.StudyDescription = "SYNTHETIC PHANTOM - NOT A PATIENT SCAN"
            ds.ImageType = ["DERIVED", "SECONDARY", "OTHER"]
            ds.SeriesNumber, ds.InstanceNumber = slot_index + 1, index + 1
            ds.RepetitionTime, ds.EchoTime = tr, te
            ds.ScanningSequence = "SE"
            # Fat suppression must be machine-readable: description text
            # matches the shared classifier, and ScanOptions states it
            # explicitly for every series, including non-fat-suppressed T1.
            ds.ScanOptions = ["FS"] if fs else ["NONE"]
            ds.Rows = ds.Columns = 336
            ds.PixelSpacing = [0.43, 0.43]
            ds.SliceThickness = ds.SpacingBetweenSlices = 3
            ds.ImageOrientationPatient = orientation
            ds.ImagePositionPatient = (np.array([-110 if side == "R" else 40, -70, -70]) + normal * (index - 6) * 3).tolist()
            ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit, ds.PixelRepresentation = 15, 0
            ds.BurnedInAnnotation = "NO"
            phase = (index - 5.5) / 12
            # Piecewise-constant tissue regions with hard boundaries. Summed
            # Gaussians alone produce an all-gradient image with no edges for
            # the viewer or the patch tokens to resolve.
            soft = (x*x + y*y) < 0.82
            phantom = np.where(soft, 300.0, 40.0)
            phantom += np.where(soft & (y > 0.1), 220.0, 0.0)
            for cx, cy, radius, value in (
                (-0.34, -0.18, 0.26, 1650.0),
                (0.34, -0.18 + phase * 0.12, 0.24, 1500.0),
            ):
                bone = ((x - cx)**2 + (y - cy)**2) < radius * radius
                phantom = np.where(bone, value, phantom)
                cortex = bone & (((x - cx)**2 + (y - cy)**2) > (radius - 0.035)**2)
                phantom = np.where(cortex, 180.0, phantom)
            cartilage = (np.abs(y - 0.16) < 0.035) & (np.abs(x) < 0.56)
            phantom = np.where(cartilage, 1150.0, phantom)
            fluid = (np.abs(y - 0.3) < 0.05) & (np.abs(x) < 0.3 - phase * 0.05)
            phantom = np.where(fluid, 2100.0 if fs else 820.0, phantom)
            ligament = (np.abs(x + y * 0.55) < 0.028) & (y > -0.42) & (y < 0.2)
            phantom = np.where(ligament, 120.0, phantom)
            # Deterministic speckle keeps slices distinguishable without blur.
            grain = np.sin(x * 140 + index * 1.7) * np.cos(y * 132 + slot_index * 2.3)
            phantom = phantom * (1 + 0.035 * grain)
            ds.PixelData = np.clip(phantom, 0, 65535).astype("<u2").tobytes()
            ds.save_as(acquisition / f"slice-{index+1:03}.dcm", enforce_file_format=True)
            total += 1
    return {"folder":name,"laterality":side,"slots":slots,"files":total,"synthetic":True}


def generate(root: Path = ROOT) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    records = [create_study(root, name, side, slots) for name, side, slots in (
        ("01-complete-right-synthetic", "R", 6),
        ("02-complete-left-synthetic", "L", 6),
        ("03-partial-right-synthetic", "R", 3),
    )]
    (root / "manifest.json").write_text(json.dumps(records, indent=2))
    return records


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
