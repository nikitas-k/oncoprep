from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)

REQUIRED_MODALITIES = ("T1", "T1ce", "T2", "FLAIR")


@dataclass(frozen=True)
class BratsSubject:
    subject_id: str
    session_id: str | None
    modality_paths: dict[str, Path]


class BratsValidationError(ValueError):
    pass


def _subject_key(subject: str, session: str | None) -> str:
    if session:
        return f"sub-{subject}_ses-{session}"
    return f"sub-{subject}"


def find_brats_modalities(bids_root: Path, subject: str, session: str | None) -> BratsSubject:
    anat_root = bids_root / f"sub-{subject}"
    if session:
        anat_root = anat_root / f"ses-{session}"
    anat_root = anat_root / "anat"

    modality_paths: dict[str, Path] = {}
    for modality in REQUIRED_MODALITIES:
        pattern = f"sub-{subject}"
        if session:
            pattern += f"_ses-{session}"
        pattern += f"_*{modality}.nii"
        matches = sorted(anat_root.glob(pattern))
        if not matches:
            raise BratsValidationError(
                f"Missing modality {modality} for {_subject_key(subject, session)} in {anat_root}"
            )
        modality_paths[modality] = matches[0]

    return BratsSubject(subject_id=subject, session_id=session, modality_paths=modality_paths)


def validate_brats_dataset(bids_root: Path, subjects: Iterable[str]) -> list[BratsSubject]:
    validated: list[BratsSubject] = []
    for subject in subjects:
        validated.append(find_brats_modalities(bids_root, subject, session=None))
        LOGGER.info("Validated BraTS modalities for sub-%s", subject)
    return validated
