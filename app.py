"""Mamba-Dent HITL — Streamlit app entrypoint.

Run with:
    streamlit run app.py

MVP scope (see README.md): only the "segmentation" layer — the 78-class
ToothFairy3 label maps — is reviewable today, because that's the only
per-case label file that actually exists. "apex" / "nerve_centerline" are
declared in config/config.yaml but inert until Mamba-Dent's aux-label
generation produces real files; nothing here needs to change when it does.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import data_loader, db_backend, hf_data_source, nifti_utils, validation, viewer

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.yaml")

st.set_page_config(page_title="Mamba-Dent HITL", layout="wide")


# ---------------------------------------------------------------- caching --

@st.cache_data(show_spinner=False)
def _load_config():
    return data_loader.load_config(CONFIG_PATH)


@st.cache_data(show_spinner="Scanning dataset...")
def _discover_cases():
    cfg = _load_config()
    if hf_data_source.is_configured():
        return hf_data_source.discover_cases(cfg)
    return data_loader.discover_cases(cfg)


@st.cache_data(show_spinner=False)
def _label_names():
    cfg = _load_config()
    if hf_data_source.is_configured():
        return hf_data_source.load_label_names(cfg)
    return data_loader.load_label_names(cfg)


@st.cache_data(show_spinner=False)
def _label_colors():
    return data_loader.build_label_colors(list(_label_names().keys()))


def _load_one_volume(path: str, dtype):
    """A Case's image_path/layer_paths hold either a real local path
    (local-disk mode) or a repo-relative path (Hugging Face Hub mode).
    In Hub mode, the downloaded file is deleted right after nibabel
    reads it into memory — otherwise every case ever viewed this session
    stays on disk forever, which on a hosted container with a
    memory-backed filesystem is effectively an unbounded memory leak on
    top of the in-memory volume cache (see _load_case_volumes)."""
    if hf_data_source.is_configured():
        local_path = hf_data_source.resolve_local_path(path)
        vol = nifti_utils.load_volume(local_path, dtype=dtype)
        hf_data_source.cleanup_local_path(local_path)
        return vol
    return nifti_utils.load_volume(path, dtype=dtype)


@st.cache_data(show_spinner="Loading case...", max_entries=1)
def _load_case_volumes(image_path: str, layer_items: tuple[tuple[str, str], ...]):
    """Loads exactly one case's image + available overlay layers.
    max_entries=1: keep only the current case's raw volumes in memory.
    Combined with the scrub-stack cache (also max_entries=1) this bounds
    memory to roughly one case's worth at a time.

    dtype choices matter a lot here — a full-size case at float32/int32
    is ~550MB just for the two raw arrays, which is enough on its own to
    trip a memory limit (this machine's, or a hosted free tier's). int16
    for the image (measured -1000..3500 HU, fits easily, and the display
    path already quantizes to 256 levels across a ~3000 HU window — a
    <1 HU rounding difference from int16 is invisible at that scale) and
    uint8 for labels (matches the *original on-disk dtype* exactly — IDs
    only run 0-148 — this was pure unnecessary widening before, not a
    tradeoff) cut that to ~206MB."""
    cfg = _load_config()
    kind_by_key = {l["key"]: l["kind"] for l in cfg["layers"]}

    image_vol = _load_one_volume(image_path, dtype=np.int16)
    layers, errors = {}, {}
    for key, path in layer_items:
        dtype = np.uint8 if kind_by_key.get(key) == "label_map" else np.int16
        vol = _load_one_volume(path, dtype=dtype)
        err = nifti_utils.check_alignment(image_vol, vol)
        if err:
            errors[key] = err
        else:
            layers[key] = vol
    return image_vol, layers, errors


@st.cache_data(show_spinner=False)
def _label_ids_in_volume(path: str) -> list[int]:
    """All label IDs present anywhere in a segmentation volume — used for
    a stable "what's in this case" legend, rather than recomputing (and
    losing, once slice position moved client-side) a per-slice one.
    Only the small returned ID list is cached (Streamlit caches return
    values, not locals), but uint8 still matters for the peak memory
    this function touches while it runs — see _load_case_volumes."""
    vol = _load_one_volume(path, dtype=np.uint8)
    return sorted(int(i) for i in np.unique(vol.data) if i != 0)


@st.cache_data(show_spinner="Rendering slices...", max_entries=1)
def _build_scrub_stack(image_path: str, layer_items: tuple[tuple[str, str], ...],
                        view_plane: str, ct_center: float, ct_width: float,
                        opacity: float, enabled_keys: tuple[str, ...],
                        center_idx: int, radius: int):
    """Composite a window of `2*radius+1` slices around center_idx into a
    single stack for build_scrub_figure(). max_entries=1: this holds a
    decompressed image stack in memory (tens of MB) on top of the
    already-cached volumes, so we don't let more than one accumulate."""
    cfg = _load_config()
    kind_by_key = {l["key"]: l["kind"] for l in cfg["layers"]}
    image_vol, layer_vols, errors = _load_case_volumes(image_path, layer_items)

    n = nifti_utils.slice_count(image_vol, view_plane)
    center_idx = int(np.clip(center_idx, 0, n - 1))
    lo, hi = max(0, center_idx - radius), min(n - 1, center_idx + radius)
    indices = list(range(lo, hi + 1))

    colors = _label_colors()
    layer_specs = [
        (layer_vols[key], colors, opacity)
        for key in enabled_keys
        if key in layer_vols and kind_by_key.get(key) == "label_map"
    ]
    stack = viewer.composite_stack(image_vol, view_plane, ct_center, ct_width, indices, layer_specs)
    return stack, indices, errors


def _load_results_df():
    # Auto-selects storage: a Supabase (or any Postgres) connection string
    # under [connections.validation_db] in secrets.toml means results go
    # there instead of the local CSV — needed wherever local disk isn't
    # guaranteed to survive a restart/redeploy. See README "Hosting".
    if db_backend.is_configured():
        return db_backend.load_results()
    return _load_results_df_csv()


@st.cache_data(show_spinner=False)
def _load_results_df_csv():
    return validation.load_results(_load_config()["results"]["csv_path"])


# ------------------------------------------------------------------ setup --

cfg = _load_config()
cases = _discover_cases()
label_names = _label_names()
label_colors = _label_colors()

# Layers actually wired up (dir set) AND marked reviewable — this is what
# keeps apex/nerve_centerline out of the review UI until they're real.
active_layers = [l for l in cfg["layers"] if l.get("dir")]
reviewable_label_types = [l["key"] for l in active_layers if l.get("reviewable")]

if not cases:
    st.error(f"No cases found under {cfg['data']['root']} — check config/config.yaml.")
    st.stop()

case_ids = [c.case_id for c in cases]
case_by_id = {c.case_id: c for c in cases}

ss = st.session_state
ss.setdefault("case_id", None)
ss.setdefault("label_type", reviewable_label_types[0] if reviewable_label_types else None)
ss.setdefault("view_plane", "axial")
ss.setdefault("slice_idx", None)
ss.setdefault("opacity", 0.45)
ss.setdefault("window_center", cfg["display"]["default_window_center"])
ss.setdefault("window_width", cfg["display"]["default_window_width"])
ss.setdefault("layer_toggles", {l["key"]: l.get("enabled_by_default", False) for l in active_layers})

results_df = _load_results_df()
reviewed_pairs = validation.reviewed_case_label_pairs(results_df)


def _is_reviewed(case_id: str, label_type: str) -> bool:
    return (case_id, label_type) in reviewed_pairs


def _pick_first_unreviewed() -> str:
    for cid in case_ids:
        if not _is_reviewed(cid, ss.label_type):
            return cid
    return case_ids[0]


if ss.case_id is None or ss.case_id not in case_by_id:
    ss.case_id = _pick_first_unreviewed()


# ---------------------------------------------------------------- sidebar --

with st.sidebar:
    st.header("Mamba-Dent HITL")
    st.caption("Annotation validation")

    st.divider()
    stats = validation.summary_stats(results_df, len(cases), reviewable_label_types)
    st.subheader("Validation summary")
    st.metric("Total cases", stats["total_cases"])
    c1, c2 = st.columns(2)
    c1.metric("Reviewed", stats["reviewed_cases"])
    c2.metric("Remaining", stats["remaining_cases"])
    st.progress(stats["reviewed_cases"] / max(stats["total_cases"], 1))

    r1, r2, r3 = st.columns(3)
    r1.metric("PASS", stats["PASS"])
    r2.metric("NEEDS FIX", stats["NEEDS_CORRECTION"])
    r3.metric("REJECT", stats["REJECT"])

    for lt, counts in stats["by_label_type"].items():
        with st.expander(f"By label: {lt}"):
            st.write(counts)

    st.divider()
    st.subheader("Results")
    if db_backend.is_configured():
        st.caption("Storage: **Supabase** (secrets.toml configured) — decisions "
                   "survive restarts/redeploys. Download here works from any browser.")
    else:
        st.caption("Storage: **local CSV** — lives on whatever machine is running "
                   "this app. Not durable on an ephemeral host (see README) unless "
                   "secrets.toml's [connections.validation_db] is set.")
    if results_df.empty:
        st.caption("No decisions recorded yet.")
    else:
        st.download_button(
            "⬇ Download results CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="validation_results.csv",
            mime="text/csv",
            width='stretch',
        )
        with st.expander(f"View all {len(results_df)} recorded decisions"):
            st.dataframe(results_df, width='stretch', hide_index=True)


# ----------------------------------------------------------------- header --

st.title("Mamba-Dent HITL — Annotation Validation")

top_l, top_m, top_r = st.columns([2, 3, 2])
with top_l:
    case_pos = case_ids.index(ss.case_id)
    # NOTE: deliberately no `key=` here. A keyed selectbox's session_state
    # value wins over `index` on every rerun after the first, which was
    # silently overwriting Previous/Next/auto-advance's change to
    # ss.case_id back to the old case on the very same rerun ("the
    # buttons don't do anything"). Leaving it unkeyed makes `index`
    # authoritative every render, driven purely by ss.case_id.
    # format_func must be a pure function of `cid` — no reading of
    # st.session_state inside it. Snapshot the current label_type/status
    # here rather than closing over `ss` live.
    _current_label_type = ss.label_type
    _status_by_case = {cid: _is_reviewed(cid, _current_label_type) for cid in case_ids}
    chosen = st.selectbox("Case", options=case_ids, index=case_pos,
                           format_func=lambda cid: f"{'✓' if _status_by_case[cid] else '○'} {cid}")
    if chosen != ss.case_id:
        ss.case_id = chosen
        ss.slice_idx = None  # reset slice on case change
with top_m:
    st.write("")
    st.write(f"**Progress:** case {case_pos + 1} / {len(cases)}")
with top_r:
    st.write("")
    nav_l, nav_r = st.columns(2)
    if nav_l.button("← Previous", width='stretch', disabled=case_pos == 0):
        ss.case_id = case_ids[case_pos - 1]
        ss.slice_idx = None
        st.rerun()
    if nav_r.button("Next →", width='stretch', disabled=case_pos == len(cases) - 1):
        ss.case_id = case_ids[case_pos + 1]
        ss.slice_idx = None
        st.rerun()

if len(reviewable_label_types) > 1:
    ss.label_type = st.radio("Label being reviewed", reviewable_label_types, horizontal=True,
                              index=reviewable_label_types.index(ss.label_type))
else:
    st.caption(f"Label being reviewed: **{ss.label_type}** "
               f"(apex / nerve_centerline aren't generated yet — see README)")


# ------------------------------------------------------------------ load --

case = case_by_id[ss.case_id]
layer_items = tuple(sorted(case.layer_paths.items()))
image_vol, layer_vols, layer_errors = _load_case_volumes(case.image_path, layer_items)

for key, err in layer_errors.items():
    st.error(f"Layer '{key}' not displayed — alignment check failed: {err}")

n_slices = nifti_utils.slice_count(image_vol, ss.view_plane)
if ss.slice_idx is None:
    ss.slice_idx = n_slices // 2
ss.slice_idx = min(ss.slice_idx, n_slices - 1)


# ------------------------------------------------------------- main view --

col_img, col_ctrl = st.columns([3, 1])

with col_ctrl:
    st.subheader("Layers")
    for layer in active_layers:
        key = layer["key"]
        available = key in layer_vols
        ss.layer_toggles[key] = st.checkbox(
            layer["label"], value=ss.layer_toggles.get(key, False),
            disabled=not available,
            help=None if available else "No file generated for this case yet",
        )

    st.subheader("Display")
    ss.opacity = st.slider("Overlay opacity", 0.0, 1.0, ss.opacity, 0.05)
    ss.view_plane = st.radio("View plane", ["axial", "coronal", "sagittal"],
                              index=["axial", "coronal", "sagittal"].index(ss.view_plane),
                              horizontal=True)
    ss.slice_idx = st.slider(f"Jump to slice ({ss.view_plane})", 0, n_slices - 1, ss.slice_idx)
    st.caption(
        f"That jump slider repositions a ±{cfg['display']['scrub_radius']}-slice window "
        "(~1-2s to rebuild). The slider **under the image** scrubs within that window "
        "instantly, in the browser — drag that one for fine navigation."
    )

    with st.expander("Window / contrast"):
        ss.window_center = st.slider("Window center", -1000, 4000, int(ss.window_center), 50)
        ss.window_width = st.slider("Window width", 1, 6000, int(ss.window_width), 50)

    st.subheader("Legend (all labels in this case)")

with col_img:
    # alignment errors are already reported above via layer_errors — this
    # call reuses the same (cached) _load_case_volumes internally, so its
    # own error dict would just be a duplicate.
    enabled_keys = tuple(sorted(k for k, v in ss.layer_toggles.items() if v))
    stack, window_indices, _errs = _build_scrub_stack(
        case.image_path, layer_items, ss.view_plane,
        ss.window_center, ss.window_width, ss.opacity, enabled_keys,
        ss.slice_idx, cfg["display"]["scrub_radius"],
    )
    start_pos = ss.slice_idx - window_indices[0]
    fig = viewer.build_scrub_figure(stack, window_indices, start_pos)
    st.plotly_chart(fig, width='stretch', config={"scrollZoom": True})

    with col_ctrl:
        visible_ids: list[int] = []
        for key in enabled_keys:
            if key in layer_vols:
                visible_ids.extend(_label_ids_in_volume(case.layer_paths[key]))
        if visible_ids:
            for lid in sorted(set(visible_ids)):
                swatch = label_colors.get(lid, "rgb(128,128,128)")
                name = label_names.get(lid, f"id {lid}")
                st.markdown(
                    f'<span style="display:inline-block;width:10px;height:10px;'
                    f'background:{swatch};margin-right:6px;border-radius:2px;"></span>{name}',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No overlay layer enabled, or nothing labeled in this case.")


# ---------------------------------------------------------------- review --

st.divider()
st.subheader(f"Review decision — {ss.case_id} / {ss.label_type}")

already = _is_reviewed(ss.case_id, ss.label_type)
if already:
    st.info("This case/label already has at least one recorded decision "
            "(submitting again adds another entry, it won't overwrite the earlier one).")

b1, b2, b3 = st.columns(3)
decision = None
if b1.button("✅ PASS", width='stretch'):
    decision = "PASS"
if b2.button("⚠️ NEEDS CORRECTION", width='stretch'):
    decision = "NEEDS_CORRECTION"
if b3.button("❌ REJECT", width='stretch'):
    decision = "REJECT"

notes = st.text_area("Reviewer notes", key=f"notes_{ss.case_id}_{ss.label_type}",
                      placeholder="e.g. apex point misplaced, centerline deviates from canal...")

if decision:
    if db_backend.is_configured():
        db_backend.append_result(ss.case_id, ss.label_type, cfg["review"]["reviewer_id"], decision, notes)
    else:
        validation.append_result(
            cfg["results"]["csv_path"], ss.case_id, ss.label_type,
            cfg["review"]["reviewer_id"], decision, notes,
        )
        _load_results_df_csv.clear()  # only the CSV path caches; the DB path queries fresh (ttl=0)
    st.success(f"Saved: {ss.case_id} / {ss.label_type} = {decision}")
    next_id = None
    for cid in case_ids[case_pos + 1:] + case_ids[:case_pos + 1]:
        if not _is_reviewed(cid, ss.label_type) and cid != ss.case_id:
            next_id = cid
            break
    if next_id:
        ss.case_id = next_id
        ss.slice_idx = None
    st.rerun()
