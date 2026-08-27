"""Composite a CBCT slice + label-map overlay into a displayable RGB image,
and build the Plotly figure (pan/zoom come for free from px.imshow).
"""
from __future__ import annotations

import numpy as np
import plotly.express as px

from . import nifti_utils


def window_to_uint8(slice_2d: np.ndarray, center: float, width: float) -> np.ndarray:
    lo = center - width / 2
    hi = center + width / 2
    clipped = np.clip(slice_2d, lo, hi)
    scaled = (clipped - lo) / max(hi - lo, 1e-6)
    return (scaled * 255).astype(np.uint8)


def composite_rgba(
    base_gray_u8: np.ndarray,
    overlays: list[tuple[np.ndarray, dict[int, str], float]],
) -> np.ndarray:
    """base_gray_u8: (H, W) uint8 grayscale CBCT slice.
    overlays: list of (label_slice, id_to_color, opacity), drawn in order.
    Returns an (H, W, 3) uint8 RGB array.
    """
    rgb = np.stack([base_gray_u8] * 3, axis=-1).astype(np.float32)

    for label_slice, id_to_color, opacity in overlays:
        if opacity <= 0:
            continue
        ids_present = np.unique(label_slice)
        for label_id in ids_present:
            if label_id == 0:
                continue
            color_str = id_to_color.get(int(label_id))
            if color_str is None:
                continue
            r, g, b = _parse_rgb(color_str)
            mask = label_slice == label_id
            rgb[mask] = (1 - opacity) * rgb[mask] + opacity * np.array([r, g, b])

    return np.clip(rgb, 0, 255).astype(np.uint8)


def _parse_rgb(color_str: str) -> tuple[int, int, int]:
    nums = color_str[color_str.index("(") + 1: color_str.index(")")].split(",")
    return int(nums[0]), int(nums[1]), int(nums[2])


def build_figure(rgb_image: np.ndarray):
    fig = px.imshow(rgb_image)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan",
        coloraxis_showscale=False,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def composite_stack(
    image_vol,
    view_plane: str,
    window_center: float,
    window_width: float,
    indices: list[int],
    layer_specs: list[tuple],  # (Volume, id_to_color, opacity) per enabled overlay layer
) -> np.ndarray:
    """Composite a *contiguous window* of slices (not the whole volume —
    see build_scrub_figure) into one (len(indices), H, W, 3) uint8 stack,
    suitable for a Plotly animation-frame slider that scrubs client-side.
    """
    frames = []
    for idx in indices:
        base = nifti_utils.get_slice(image_vol, view_plane, idx)
        base_u8 = window_to_uint8(base, window_center, window_width)
        overlays = [
            (nifti_utils.get_slice(vol, view_plane, idx), colors, opacity)
            for vol, colors, opacity in layer_specs
        ]
        frames.append(composite_rgba(base_u8, overlays))
    return np.stack(frames, axis=0)


def build_scrub_figure(stack: np.ndarray, indices: list[int], start_pos: int):
    """Plotly figure with a native animation-frame slider — once built,
    dragging through `indices` happens entirely in the browser (no
    Streamlit rerun per tick), which is what makes it feel like a real
    image viewer instead of a web form.
    """
    fig = px.imshow(stack, animation_frame=0, binary_string=True)

    start_pos = int(np.clip(start_pos, 0, len(indices) - 1))
    fig.data[0].source = fig.frames[start_pos].data[0].source

    if fig.layout.sliders:
        slider = fig.layout.sliders[0]
        slider.active = start_pos
        slider.currentvalue = {"prefix": "Slice "}
        for step, real_index in zip(slider.steps, indices):
            step.label = str(real_index)
        # snap immediately on drag — no fade/tween lag
        for step in slider.steps:
            step.args[1]["transition"]["duration"] = 0
            step.args[1]["frame"]["duration"] = 0

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan",
        coloraxis_showscale=False,
        updatemenus=[],  # drop the play/pause button — this isn't a movie
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def visible_ids_in_slice(label_slice: np.ndarray) -> list[int]:
    ids = np.unique(label_slice)
    return [int(i) for i in ids if i != 0]
