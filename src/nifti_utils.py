"""NIfTI loading + slicing helpers.

Everything here is reoriented to canonical RAS on load, so slice indices
and axis meanings (axial/coronal/sagittal) behave consistently in the UI
regardless of how a given file was stored on disk (this dataset's raw
ToothFairy3 files are LPS).
"""
from __future__ import annotations

from dataclasses import dataclass

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class Volume:
    """A loaded, canonically-oriented 3D volume."""
    data: np.ndarray          # shape (X, Y, Z) in RAS order
    zooms: tuple[float, float, float]
    path: str


def load_volume(path: str, dtype=np.float32) -> Volume:
    """Load a NIfTI file, reoriented to closest-canonical (RAS)."""
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    data = np.asanyarray(img.dataobj, dtype=dtype)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return Volume(data=data, zooms=zooms, path=path)


def check_alignment(reference: Volume, other: Volume, tol: float = 1e-2) -> str | None:
    """Return an error string if `other` doesn't align with `reference`,
    or None if it's fine to overlay. Never silently resizes/guesses.
    """
    if reference.data.shape != other.data.shape:
        return (
            f"shape mismatch: reference {reference.data.shape} "
            f"vs overlay {other.data.shape} ({other.path})"
        )
    for a, b in zip(reference.zooms, other.zooms):
        if abs(a - b) > tol:
            return (
                f"voxel spacing mismatch: reference {reference.zooms} "
                f"vs overlay {other.zooms} ({other.path})"
            )
    return None


# axis index within a canonical-RAS (X, Y, Z) array for each view plane
PLANE_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}


def slice_count(volume: Volume, plane: str) -> int:
    return volume.data.shape[PLANE_AXIS[plane]]


def get_slice(volume: Volume, plane: str, index: int) -> np.ndarray:
    """2D slice through `volume` at `index` along `plane`, oriented for
    on-screen display (row0 = top of image)."""
    axis = PLANE_AXIS[plane]
    index = int(np.clip(index, 0, volume.data.shape[axis] - 1))
    if axis == 2:
        sl = volume.data[:, :, index]
    elif axis == 1:
        sl = volume.data[:, index, :]
    else:
        sl = volume.data[index, :, :]
    # RAS array -> displayed with anatomical "up" at the top of the image
    return np.rot90(sl)
