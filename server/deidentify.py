"""Conservative demo metadata sanitizer; not a clinical de-identification guarantee."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

WARNINGS = [
    "Research demo only: metadata sanitization is not a clinical de-identification guarantee.",
    "Burned-in pixel annotations are not removed; inspect images before upload.",
]
TEXT_KEEP = {
    "Modality", "PhotometricInterpretation", "ImageLaterality", "Laterality",
    "ScanOptions", "ScanningSequence", "SequenceVariant", "PatientPosition",
    "PixelPresentation", "VolumetricProperties", "PresentationLUTShape",
}
HINTS = re.compile(r"\b(?:SAG(?:ITTAL)?|COR(?:ONAL)?|AX(?:IAL)?|T1W?|T2W?|PDW?|FS|NOFS|NONFS|FATSAT|FAT_SAT|FSAT|STIR|SPAIR|SPIR|TIRM|FATSUP|LOCALIZER|LOCALISER|SCOUT|SURVEY|DWI|ADC|GRE)\b", re.I)

@dataclass
class DeidentificationReport:
    removed: list[str] = field(default_factory=list)
    blanked: list[str] = field(default_factory=list)
    pseudonymised: list[str] = field(default_factory=list)
    private_tags_removed: int = 0
    files_processed: int = 0
    warnings: list[str] = field(default_factory=lambda: list(WARNINGS))

    def merge(self, other: "DeidentificationReport") -> None:
        for key in ("removed", "blanked", "pseudonymised", "warnings"):
            setattr(self, key, sorted(set(getattr(self, key)) | set(getattr(other, key))))
        self.private_tags_removed += other.private_tags_removed
        self.files_processed += other.files_processed

    def as_dict(self) -> dict[str, Any]:
        return dict(removed=sorted(set(self.removed)), blanked=sorted(set(self.blanked)),
                    pseudonymised=sorted(set(self.pseudonymised)),
                    privateTagsRemoved=self.private_tags_removed,
                    filesProcessed=self.files_processed, warnings=self.warnings)


def pseudonymise(value: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()
    return f"2.25.{int(digest[:32], 16)}"


def deidentify_dataset(dataset: Any, salt: str) -> DeidentificationReport:
    """Recursively strip text/private tags and remap all instance UID references."""
    report = DeidentificationReport(files_processed=1)
    if str(getattr(dataset, "BurnedInAnnotation", "")).upper() == "YES":
        report.warnings.append("Input explicitly declares burned-in annotations; pixels remain unchanged.")

    def clean(ds):
        for element in list(ds):
            key = element.keyword or str(element.tag)
            if (element.tag.is_private or 0x5000 <= element.tag.group <= 0x60FF
                    or element.VR == "UN"):
                del ds[element.tag]
                report.private_tags_removed += int(element.tag.is_private)
                report.removed.append(key)
            elif element.VR == "SQ":
                for child in element.value:
                    clean(child)
            elif element.VR == "UI":
                # Class/transfer syntax UIDs describe encoding, not identity.
                if "ClassUID" not in key and key != "TransferSyntaxUID":
                    values = element.value if element.VM > 1 else [element.value]
                    remapped = [pseudonymise(str(value), salt) for value in values if value]
                    element.value = remapped if element.VM > 1 else (remapped[0] if remapped else "")
                    report.pseudonymised.append(key)
            elif element.VR in {"DA", "DT", "TM"}:
                element.value = ""
                report.blanked.append(key)
            elif key in {"SeriesDescription", "SequenceName"}:
                # Keep protocol tokens only; free text may contain names/IDs.
                text = str(element.value).upper()
                text = re.sub(r"\bNON[\s_-]+FS\b|\bNO[\s_-]+FS\b", "NOFS", text)
                text = re.sub(r"\bFAT[\s_-]+(?:SAT|SUPP)(?:RESSION)?\b", "FS", text)
                element.value = " ".join(HINTS.findall(text))
                report.blanked.append(key + ":free-text")
            elif element.VR in {"PN", "LO", "LT", "SH", "ST", "UT", "UC", "UR", "AE", "AS", "CS"} and key not in TEXT_KEEP:
                del ds[element.tag]
                report.removed.append(key)
    clean(dataset)
    if getattr(dataset, "file_meta", None) is not None:
        clean(dataset.file_meta)
        dataset.file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
        dataset.file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
        dataset.file_meta.ImplementationClassUID = "2.25.2087733307490612016"
    dataset.PatientIdentityRemoved = "YES"
    dataset.DeidentificationMethod = "Research demo metadata sanitizer; pixels unchanged"
    dataset.preamble = b"\0" * 128
    return report
