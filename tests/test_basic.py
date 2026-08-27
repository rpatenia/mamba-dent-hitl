"""Basic tests, including against one real case from the actual dataset
(ToothFairy3F_001) per the project's testing requirement. Skips the
real-data tests gracefully if the dataset isn't present on this machine.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_loader, nifti_utils, validation, viewer

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config", "config.yaml")

requires_dataset = pytest.mark.skipif(
    not os.path.exists(r"C:\Users\Ryan\Desktop\ToothFairy3\dataset.json"),
    reason="ToothFairy3 dataset not present on this machine",
)


# ------------------------------------------------------------ pure logic --

def test_window_to_uint8_range():
    sl = np.array([[-1000.0, 0.0, 3500.0]])
    out = viewer.window_to_uint8(sl, center=1200, width=3000)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
    assert out[0, 0] < out[0, 1] < out[0, 2]  # monotonic with intensity


def test_composite_rgba_no_overlay_is_grayscale():
    base = np.array([[0, 128, 255]], dtype=np.uint8)
    rgb = viewer.composite_rgba(base, overlays=[])
    assert rgb.shape == (1, 3, 3)
    assert (rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 1] == rgb[..., 2]).all()


def test_composite_rgba_applies_overlay_color():
    base = np.zeros((2, 2), dtype=np.uint8)
    labels = np.array([[0, 5], [0, 0]], dtype=np.int32)
    colors = {5: "rgb(255,0,0)"}
    rgb = viewer.composite_rgba(base, overlays=[(labels, colors, 1.0)])
    assert tuple(rgb[0, 1]) == (255, 0, 0)
    assert tuple(rgb[0, 0]) == (0, 0, 0)  # background untouched


def test_composite_stack_shape_and_content():
    image_vol = nifti_utils.Volume(
        data=np.zeros((4, 4, 6), dtype=np.float32), zooms=(0.3, 0.3, 0.3), path="fake_image",
    )
    label_data = np.zeros((4, 4, 6), dtype=np.int32)
    label_data[1, 1, :] = 7  # a labeled voxel present in every slice
    label_vol = nifti_utils.Volume(data=label_data, zooms=(0.3, 0.3, 0.3), path="fake_label")
    colors = {7: "rgb(255,0,0)"}

    indices = [1, 2, 3]
    stack = viewer.composite_stack(
        image_vol, "axial", window_center=0, window_width=1,
        indices=indices, layer_specs=[(label_vol, colors, 1.0)],
    )
    assert stack.shape == (3, 4, 4, 3)
    assert stack.dtype == np.uint8
    # the labeled voxel should be colored red in every frame of the window
    for frame in stack:
        assert (frame == np.array([255, 0, 0])).all(axis=-1).any()


def test_build_scrub_figure_slider_labels_and_start_position():
    stack = np.zeros((5, 3, 3, 3), dtype=np.uint8)
    indices = [10, 11, 12, 13, 14]  # real slice numbers, not 0-based frame position
    fig = viewer.build_scrub_figure(stack, indices, start_pos=2)

    assert len(fig.frames) == 5
    slider = fig.layout.sliders[0]
    assert slider.active == 2
    assert [s.label for s in slider.steps] == ["10", "11", "12", "13", "14"]
    # the displayed trace should match the requested start frame, not frame 0
    assert fig.data[0].source == fig.frames[2].data[0].source


def test_db_backend_is_configured_reflects_secrets(monkeypatch):
    """Deliberately monkeypatches st.secrets rather than relying on
    whatever secrets.toml happens to exist on this machine — this
    machine now has real Supabase credentials configured for actual use,
    but the test should still deterministically cover both states."""
    import streamlit as st
    from src import db_backend

    monkeypatch.setattr(st, "secrets", {})
    assert db_backend.is_configured() is False


def test_hf_data_source_is_configured_reflects_secrets(monkeypatch):
    import streamlit as st
    from src import hf_data_source

    monkeypatch.setattr(st, "secrets", {})
    assert hf_data_source.is_configured() is False

    monkeypatch.setattr(st, "secrets", {"huggingface": {"token": "x", "repo_id": "y/z"}})
    assert hf_data_source.is_configured() is True


def test_hf_data_source_discover_cases_matches_local_logic(monkeypatch):
    """Unit-tests the Hub file-matching logic against a fake repo file
    listing — doesn't need real network access or the (still uploading)
    real dataset, and can't accidentally load a multi-hundred-MB volume
    (this machine is memory-constrained; a background dataset upload is
    also using RAM right now)."""
    import streamlit as st
    from src import hf_data_source

    monkeypatch.setattr(st, "secrets", {"huggingface": {"token": "x", "repo_id": "y/z"}})

    fake_files = [
        "dataset.json",
        "imagesTr/ToothFairy3F_001_0000.nii.gz",
        "labelsTr/ToothFairy3F_001.nii.gz",
        "imagesTr/ToothFairy3F_002_0000.nii.gz",
        # case 2 has no matching label file yet — layer_paths should just omit it
        "README.md",  # must not be mistaken for a case
    ]

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            assert repo_id == "y/z"
            assert repo_type == "dataset"
            return fake_files

    monkeypatch.setattr(hf_data_source, "_api", lambda: FakeApi())

    cfg = data_loader.load_config(CONFIG_PATH)
    cases = {c.case_id: c for c in hf_data_source.discover_cases(cfg)}

    assert set(cases.keys()) == {"ToothFairy3F_001", "ToothFairy3F_002"}
    assert cases["ToothFairy3F_001"].image_path == "imagesTr/ToothFairy3F_001_0000.nii.gz"
    assert cases["ToothFairy3F_001"].layer_paths == {"segmentation": "labelsTr/ToothFairy3F_001.nii.gz"}
    assert cases["ToothFairy3F_002"].layer_paths == {}  # no label file yet — must not error


def test_build_label_colors_deterministic_and_skips_background():
    colors_a = data_loader.build_label_colors([0, 1, 2, 3])
    colors_b = data_loader.build_label_colors([0, 1, 2, 3])
    assert colors_a == colors_b
    assert colors_a[0] == "rgba(0,0,0,0)"
    assert len({colors_a[1], colors_a[2], colors_a[3]}) == 3  # distinct


# --------------------------------------------------------------- results --

def test_validation_append_and_load_roundtrip(tmp_path):
    csv_path = str(tmp_path / "validation_results.csv")

    validation.append_result(csv_path, "case_001", "segmentation", "reviewer_01", "PASS", "looks good")
    validation.append_result(csv_path, "case_002", "segmentation", "reviewer_01", "NEEDS_CORRECTION", "")

    df = validation.load_results(csv_path)
    assert len(df) == 2
    assert set(df["case_id"]) == {"case_001", "case_002"}

    # a second decision on the same case/label must NOT overwrite the first
    validation.append_result(csv_path, "case_001", "segmentation", "reviewer_02", "REJECT", "disagree")
    df2 = validation.load_results(csv_path)
    assert len(df2) == 3
    assert (df2["case_id"] == "case_001").sum() == 2


def test_validation_rejects_bad_result(tmp_path):
    with pytest.raises(ValueError):
        validation.append_result(str(tmp_path / "r.csv"), "c1", "segmentation", "r1", "MAYBE", "")


def test_summary_stats():
    import pandas as pd
    df = pd.DataFrame([
        {"case_id": "c1", "label_type": "segmentation", "reviewer": "r1", "result": "PASS", "notes": "", "timestamp": ""},
        {"case_id": "c2", "label_type": "segmentation", "reviewer": "r1", "result": "REJECT", "notes": "", "timestamp": ""},
    ])
    stats = validation.summary_stats(df, total_cases=5, label_types=["segmentation"])
    assert stats["reviewed_cases"] == 2
    assert stats["remaining_cases"] == 3
    assert stats["PASS"] == 1 and stats["REJECT"] == 1


# ------------------------------------------------------- real dataset IO --

@requires_dataset
def test_load_config():
    cfg = data_loader.load_config(CONFIG_PATH)
    assert os.path.isdir(cfg["data"]["root"])


@requires_dataset
def test_discover_cases_finds_all_532():
    cfg = data_loader.load_config(CONFIG_PATH)
    cases = data_loader.discover_cases(cfg)
    assert len(cases) == 532
    assert all(c.layer_paths.get("segmentation") for c in cases)


@requires_dataset
def test_label_names_match_frozen_78_class_taxonomy():
    cfg = data_loader.load_config(CONFIG_PATH)
    names = data_loader.load_label_names(cfg)
    assert len(names) == 78
    assert names[0] == "background"


@requires_dataset
def test_next_and_previous_buttons_actually_change_the_case():
    """Regression test for the bug where Previous/Next silently did
    nothing: a keyed st.selectbox's session_state value was overriding
    the `index=` computed from the button's change to ss.case_id on the
    very same rerun, snapping the case back."""
    from streamlit.testing.v1 import AppTest

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    at = AppTest.from_file(os.path.join(project_root, "app.py"))
    at.run(timeout=30)
    assert not at.exception

    first_case = at.session_state["case_id"]

    next_button = next(b for b in at.button if b.label == "Next →")
    next_button.click().run(timeout=30)
    assert not at.exception
    second_case = at.session_state["case_id"]
    assert second_case != first_case, "Next did not change the case"

    prev_button = next(b for b in at.button if b.label == "← Previous")
    prev_button.click().run(timeout=30)
    assert not at.exception
    assert at.session_state["case_id"] == first_case, "Previous did not return to the prior case"


@requires_dataset
def test_pass_button_saves_immediately_no_reviewer_id_needed(monkeypatch):
    """Regression test: PASS/NEEDS CORRECTION/REJECT used to silently
    require a Reviewer ID field that no longer exists (single-reviewer
    setup) — clicking a decision did nothing but show an easy-to-miss
    warning. Carefully preserves+restores the real results CSV so this
    test doesn't leave a fake entry in it.

    Forces the CSV backend regardless of whatever real secrets.toml
    exists on this machine — this must stay a deterministic, offline
    unit test, not a test that writes into a live Supabase table on
    every run. The Supabase path was verified separately, live, once
    (see project memory / conversation history), not via this suite.
    """
    import pandas as pd
    from src import db_backend
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(db_backend, "is_configured", lambda: False)

    cfg = data_loader.load_config(CONFIG_PATH)
    csv_path = cfg["results"]["csv_path"]
    original_bytes = None
    if os.path.exists(csv_path):
        with open(csv_path, "rb") as f:
            original_bytes = f.read()

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        at = AppTest.from_file(os.path.join(project_root, "app.py"))
        at.run(timeout=30)
        assert not at.exception
        first_case = at.session_state["case_id"]

        pass_button = next(b for b in at.button if "PASS" in b.label)
        pass_button.click().run(timeout=30)
        assert not at.exception
        assert not any("Reviewer ID" in str(w) for w in at.warning), \
            "should never prompt for a reviewer ID anymore"

        df = pd.read_csv(csv_path)
        match = df[(df["case_id"] == first_case) & (df["label_type"] == "segmentation")
                    & (df["result"] == "PASS")]
        assert len(match) >= 1, "clicking PASS did not log a decision"
        assert match.iloc[-1]["reviewer"] == cfg["review"]["reviewer_id"]
    finally:
        if original_bytes is None:
            if os.path.exists(csv_path):
                os.remove(csv_path)
        else:
            with open(csv_path, "wb") as f:
                f.write(original_bytes)


@requires_dataset
def test_real_case_loads_and_aligns():
    cfg = data_loader.load_config(CONFIG_PATH)
    cases = {c.case_id: c for c in data_loader.discover_cases(cfg)}
    case = cases["ToothFairy3F_001"]

    # int16/uint8, not float32/int32 — matches what app.py actually loads
    # with in production (memory: a full-size case is ~550MB at
    # float32/int32 vs ~206MB here, which matters on a constrained host)
    image_vol = nifti_utils.load_volume(case.image_path, dtype=np.int16)
    label_vol = nifti_utils.load_volume(case.layer_paths["segmentation"], dtype=np.uint8)

    assert image_vol.data.shape == label_vol.data.shape
    err = nifti_utils.check_alignment(image_vol, label_vol)
    assert err is None, err
    # int16 rounds fractional HU values — confirm it didn't wreck the data
    assert -1500 < image_vol.data.min() and image_vol.data.max() < 5000
    assert label_vol.data.max() <= 148  # highest real ID (pulps); confirms no uint8 wraparound

    # every declared view plane should yield a sensible slice count / slice
    for plane in ("axial", "coronal", "sagittal"):
        n = nifti_utils.slice_count(image_vol, plane)
        assert n > 0
        sl = nifti_utils.get_slice(image_vol, plane, n // 2)
        assert sl.ndim == 2 and sl.size > 0

    # the label volume should contain real tooth/jaw IDs, not just background
    assert (label_vol.data != 0).any()
