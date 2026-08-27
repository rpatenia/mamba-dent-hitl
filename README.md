# Mamba-Dent HITL

A browser-based human-in-the-loop validation viewer for the Mamba-Dent thesis
pipeline. A reviewer opens a page, browses a CBCT case with its labels
overlaid, and records PASS / NEEDS CORRECTION / REJECT per case — no 3D
Slicer, no ITK-SNAP, no dataset download.

## Current scope — read this first

**This MVP reviews the ToothFairy3 78-class segmentation masks, not
apex/nerve labels.** The original ask was a viewer for automatically
generated auxiliary labels (tooth apex points, nerve centerlines), but as
of 2026-08-27 those don't exist yet — `preprocess.py` hasn't been run
against the raw ToothFairy3 data, and the aux-label generation code
(`data_utils.py`) hasn't been written.

What *does* exist, and what this MVP points at instead: 532 raw CBCT
volumes and their official 78-class segmentation label maps
(`ToothFairy3/imagesTr` + `labelsTr`). Getting a human to sanity-check
that label map — which was itself derived programmatically
(`label_map_v1.json`, frozen 2026-08-23) — is a real, useful validation
task on its own, and it's what's reviewable today.

**Nothing about the architecture is segmentation-specific.** `apex` and
`nerve_centerline` are already declared as layers in `config/config.yaml`
with `dir: null`. Once those files exist, do this:

1. Set each layer's `dir` (and confirm `suffix`) in `config/config.yaml`.
2. Confirm `kind` still matches how the label is actually encoded — the
   config assumes a per-voxel integer label map. If apex points end up
   as sparse coordinates rather than a mask, `nifti_utils`/`viewer`'s
   compositing will need a small extension (see "Adding a new layer
   kind" below) — everything else (case discovery, alignment checks,
   the review UI, CSV logging) needs no changes.

No apex/nerve code was written against invented/mocked data — see the
conversation this came out of for why.

## Install & run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (defaults to `http://localhost:8501`).

## Configuration

Everything dataset-specific lives in `config/config.yaml`: where the data
lives, how a case ID is parsed out of a filename, what layers exist and
where their files are. Point it at a different dataset by editing that
file — no code changes needed for the current label-map-style layer kind.

## Data format (as actually inspected — not assumed)

Checked directly against `ToothFairy3F_001` and a few other cases before
writing any loading code:

- CBCT images: `.nii.gz`, per-case shape varies (e.g. 512×512×262,
  410×410×274), 0.3mm isotropic spacing, LPS orientation on disk,
  intensity roughly -1000..3500 (HU-like). Loaded and reoriented to
  canonical RAS so the viewer's slice axes are consistent across cases.
- Segmentation labels: `.nii.gz`, `uint8`, same shape/spacing as the
  matching image, sparse label IDs (0, 1–48, 103–105, 111+...) defined
  in `ToothFairy3/dataset.json`'s `labels` dict (78 entries total,
  matches the thesis's frozen class count) — used directly as the
  ID→name source rather than a separately maintained copy.
- Every overlay is checked for shape + voxel-spacing alignment against
  the CBCT before being drawn; a mismatch shows an error banner instead
  of a silently wrong overlay.

## Project structure

```
mamba-dent-hitl/
├── app.py                       # Streamlit UI + wiring
├── config/config.yaml           # dataset paths, layers, display defaults
├── src/
│   ├── data_loader.py           # local-disk case discovery, label names/colors
│   ├── hf_data_source.py        # Hugging Face Hub case discovery — remote alternative to data_loader
│   ├── nifti_utils.py           # NIfTI I/O, canonical reorientation, slicing, alignment checks
│   ├── viewer.py                # windowing, overlay compositing, Plotly figure
│   ├── validation.py            # append-only CSV logging + summary stats
│   └── db_backend.py            # Supabase/Postgres results storage — remote alternative to validation.py's CSV
├── results/validation_results.csv   # created on first submitted decision (local-CSV mode only)
└── tests/test_basic.py          # unit tests + tests against real case ToothFairy3F_001
```

No separate `logging_utils.py` — logging is small enough to live in
`validation.py` alongside the CSV schema it writes, rather than adding an
empty extra module.

## Workflow

1. The app opens the first case that has no recorded decision yet for the
   currently selected label type (right now, just `segmentation`).
2. Toggle layers, adjust overlay opacity, switch view plane
   (axial/coronal/sagittal). Pan/zoom on the image itself come from
   Plotly's built-in toolbar.
3. Two slice controls, deliberately different speeds (see "Slice
   navigation" below): the sidebar's **"Jump to slice"** slider
   repositions a scrub window (~1-2s rebuild); the slider **under the
   image** scrubs within that window instantly, in the browser.
4. Click **PASS**, **NEEDS CORRECTION**, or **REJECT**; optionally add a
   note first. Submitting saves the decision and advances to the next
   unreviewed case — it never overwrites a prior decision, including
   across restarts (see "Where results go" below for CSV vs. Supabase).
5. The sidebar shows live totals: reviewed/remaining, PASS/NEEDS
   CORRECTION/REJECT counts, broken out per label type, plus a
   **Results** section to view/download everything recorded so far.

## Slice navigation: two speeds, on purpose

A single Streamlit slider round-trips to the server on every tick, which
doesn't feel like a real image viewer. A fully client-side slider across
the *whole* volume (Plotly animation frames) was the other extreme:
benchmarked at ~10s to build and ~50MB shipped to the browser per
settings change for a 262-slice case — too slow, and too much RAM for a
machine that already runs low on free memory.

The compromise: `_build_scrub_stack()` in `app.py` pre-renders a window of
`2*scrub_radius+1` slices (default radius 20, ~1.5s / ~9MB) around
whatever slice you're centered on, as Plotly animation frames. Dragging
*within* that window is instant and never touches the server — that's
the "like 3D Slicer" part. Moving further requires the coarser "Jump to
slice" slider, which rebuilds the window (a visible ~1-2s pause). Tune
`display.scrub_radius` in `config/config.yaml` if you want a bigger
window and have RAM to spare, or a smaller one for a faster rebuild.

## Bugs fixed after first use

- **Previous/Next did nothing.** The case selectbox had both `key="case_select"`
  and a computed `index=`; a keyed widget's `st.session_state` value wins
  over `index` on every rerun after the first, so the buttons' change to
  the current case was silently overwritten back on the same rerun.
  Fixed by dropping the `key` and driving the widget purely off `index`
  (see the comment in `app.py`). Caught by
  `tests/test_next_and_previous_buttons_actually_change_the_case`, which
  actually clicks the buttons via `streamlit.testing.v1.AppTest` rather
  than just asserting the script runs.
- Relatedly: the selectbox's `format_func` originally read
  `st.session_state` live inside the closure. That's fine for a real
  browser (the frontend never re-invokes Python's `format_func` after
  first render) but broke `AppTest`'s widget-state introspection between
  simulated runs, masking the real fix while writing the regression test
  above. Now it closes over a plain snapshot dict instead — no
  session-state reads inside the lambda.

## Hosting (Streamlit Community Cloud / any ephemeral host)

Two separate things don't survive a hosted, ephemeral container by
default: **results** (local disk gets wiped on restart/redeploy) and
**the dataset itself** (28GB, only ever existed on this Windows machine
— never fits in the GitHub repo the host deploys from). Both are solved
the same way: a secrets section present → the app routes to a remote
backend instead of local files. Neither needs a code change to switch.

### Results storage — Supabase (`src/db_backend.py`)

- **No `[connections.validation_db]` in secrets** → local CSV (default;
  local dev needs zero setup).
- **That section present** → every read/write goes to Postgres instead,
  surviving redeploys. Sidebar's "Results" section shows which is active.

Setup (account-side steps are yours to do — I can't create a Supabase
project or generate credentials for you):

1. Create a free project at [supabase.com](https://supabase.com).
2. Project Settings → Database → Connection string → **Session pooler**
   URI (better suited to a serverless/ephemeral host than the direct
   connection string — that one requires IPv6, which hosts like this
   typically don't have outbound), with your DB password filled in.
3. Copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml`
   (gitignored) and paste it into `[connections.validation_db]`.
4. Paste the same content into the hosted app's **Settings → Secrets**.
5. `src/db_backend.py` creates the `validation_results` table
   automatically on first use.

**Live-verified, not just written-to-spec:** connected to a real Supabase
project, confirmed table auto-creation, clicked PASS through the actual
app and confirmed the row landed via a direct query, confirmed the
sidebar reads it back correctly, then cleaned up the test row. This
actually works, checked end-to-end — not an assumption.

### Dataset access — Hugging Face Hub (`src/hf_data_source.py`)

- **No `[huggingface]` in secrets** → reads local `ToothFairy3\` files
  (default; what local dev uses).
- **That section present** → case discovery lists files in a private HF
  Hub dataset repo instead, and each file is fetched on demand via
  `hf_hub_download` (cached locally after first access, so revisiting a
  case in the same session doesn't re-download).

Setup:

1. Get a token at huggingface.co → Settings → Access Tokens (needs
   `repo.content.read` at minimum on the target repo).
2. Add `[huggingface]` to secrets (locally and/or in the hosted app's
   Settings → Secrets) — see `.streamlit/secrets.toml.example`.

**Upload status: complete.** All 532 image files + 532 label files +
`dataset.json` confirmed present in the private repo
`vvyern/toothfairy3-mamba-dent-hitl` via `upload_large_folder` (resumable
— it was interrupted twice along the way, see below, and resumed
correctly each time without re-uploading already-committed files). First
attempt crashed with native out-of-memory errors inside `hf_xet` (HF's
Rust chunked-transfer backend) running 4 parallel workers on this
RAM-constrained machine; resumed with `HF_HUB_DISABLE_XET=1` and
`num_workers=1` — slower, but stable. If this ever needs re-running for
a dataset update, keep those two settings; don't just crank workers back
up.

**Live-verified end-to-end**, not just unit-tested against a mock:
`discover_cases` against the real Hub API found all 532 cases correctly;
`resolve_local_path` really downloaded a case's image + label files;
`nifti_utils.load_volume` loaded them correctly (370×370×164, 0.3mm iso —
matches the local-mode format); the alignment check passed; `slice_count`
/`get_slice` worked across the loaded volume; and `load_label_names`
correctly pulled all 78 classes from the Hub-hosted `dataset.json`. This
was checked against `ToothFairy3S_0000` specifically (a small case, to
stay within this machine's tight free RAM during the check) — a full-size
case (~100-200MB) uses the identical code path, just more memory per
load, which is exactly what `max_entries=1` caching on `_load_case_volumes`
exists to bound.

### Troubleshooting: hard crash, no traceback, blank "Oh no." page

Hit this on first real deploy. Streamlit's generic "Oh no. Error running
app." page (not a normal Python traceback box) means the process died
hard enough that Streamlit itself never got to log a catchable
exception — consistent with an OS-level OOM kill or a native-extension
crash. The logs showed nothing at all past `Uvicorn server started`,
which is the signature of exactly that (a graceful Python error would
have logged *something*).

Root cause suspected (not fully confirmed — see below): `requirements.txt`
used loose `>=` bounds, so the host pulled whatever was newest at deploy
time — **Python 3.14** (vs. the 3.12 this was built and tested against)
and **pandas 3.0.5** (a major-version jump from the tested 2.2.3), among
others. Streamlit's own build log even showed it patching around a known
pyarrow segfault specific to that Python 3.14 environment — a sign the
platform itself has live compatibility rough edges right now.

**First fix attempt was wrong, worth recording why:** added `runtime.txt`
(`python-3.12`) and pinned every package to the exact *older* version
tested locally. `runtime.txt` had no effect — the next build still showed
Python 3.14 (possibly only read on an app's *initial* creation, not a
reboot of an existing one — untested). Worse, the old pins actively broke
the build: `pillow==11.0.0` and `numpy==2.1.3` predate Python 3.14 and
have no prebuilt wheel for it, so pip tried to compile Pillow from source
and hard-failed on missing `zlib` headers. That's a strictly worse
failure mode (build never completes) than the original (builds fine,
crashes after starting).

**Actual fix:** pin `requirements.txt` to the exact versions confirmed to
already install successfully in the real target environment (lifted
directly from a deploy log that reached `Uvicorn server started` cleanly
— streamlit 1.62.0, numpy 2.5.2, pandas 3.0.5, pillow 12.3.0,
huggingface_hub 1.28.0, etc.), rather than forcing older local-tested
versions. Verified these exact pins locally before pushing again: built
a throwaway venv, installed them, ran the full test suite against
it (18/18 pass) — confirms pandas 3.0's API changes don't break
`validation.py`, not just that install succeeds.

**Still open:** whether this actually fixes the original runtime crash
(vs. just being a correct dependency pin) isn't confirmed yet — that
needs a real redeploy + a period of use to see if it recurs. `runtime.txt`
is left in place (harmless) but nothing here depends on it actually
working. If a hard-crash-no-log happens again after this, check actual
resource usage next (Community Cloud's free tier has real memory limits,
and this app's per-case CBCT loading is genuinely memory-heavy) rather
than re-suspecting dependency versions a third time.

## `validation_results.csv` schema

```
case_id,label_type,reviewer,result,notes,timestamp
ToothFairy3F_001,segmentation,reviewer_01,PASS,"looks correct",2026-08-27 10:32:00
```

## Adding a new layer kind

The current `kind: label_map` path assumes "one integer ID per voxel,
color it categorically." If apex points arrive as sparse (x, y, z)
coordinates rather than a mask:

- Add a `kind: points` branch in `nifti_utils`/`viewer` that draws
  markers instead of a colored mask (a `data_loader`/`viewer` change,
  not an `app.py` change — the UI already treats layers generically).
- Leave `label_map`-kind layers (segmentation, and nerve centerline if
  it ships as a mask) exactly as they are.

## Performance notes

- A full-size case's raw arrays measured at **~206MB in memory**
  (image + segmentation together) — down from an initial ~549MB at
  float32/int32, cut by loading image as `int16` and labels as `uint8`
  instead (see the docstring on `_load_case_volumes` in `app.py` for
  the reasoning — this was a real fix for repeated hard crashes on a
  memory-constrained hosted deploy, not just a nice-to-have). Plus the
  scrub-stack cache on top (tens of MB, bounded by `scrub_radius`).
  `@st.cache_data(max_entries=1)` on both keeps only the current case in
  memory, never the whole dataset — deliberately 1, not more, given how
  tight this can get.
- NIfTI reads go through `nibabel`'s standard loader (not a memory-mapped
  partial read) — for `.nii.gz` this means a full per-case decompress on
  first load of that case, which is what the cache is there to amortize
  across slice/opacity/window tweaks on the same case.

## Known limitations (honest, not hidden)

- Single-reviewer setup by design — no Reviewer ID prompt; every decision
  is logged under `review.reviewer_id` in config.yaml (default `"reviewer"`).
  If a second person ever needs to review too, this needs revisiting (a
  reviewer identity of some kind, and "already reviewed" would need to
  stop being purely per case+label_type).
- No point-drag correction editor (per spec, out of scope for v1) —
  `NEEDS_CORRECTION` + a note is the mechanism for now.
- `hf_hub_download`'s local cache (Hub dataset mode) isn't bounded —
  a long session touching many different cases accumulates disk use on
  the host with nothing evicting it. Watch for this if the hosted app's
  disk fills up; not addressed yet.
- The same write-scoped HF token was reused for the deployed app's read
  access rather than generating a separate read-only one — tighter to
  scope down later, not a blocker now for a private single-owner repo.
- The Hub-mode load path was live-verified against a small case
  (`ToothFairy3S_0000`, ~370×370×164) to keep the check within this
  machine's tight free RAM — not yet against a full-size (~512×512×260,
  100-200MB) case specifically, though it's the identical code path.

## Tests

```bash
pytest tests/ -v
```

11 tests, including several run against the real `ToothFairy3F_001` case
(shape/spacing/orientation, alignment check, all three view planes) —
skipped automatically on a machine without the dataset present. Also
smoke-tested via `streamlit.testing.v1.AppTest` (runs the actual script
headlessly and asserts no exception) and manually launched with
`streamlit run` to confirm it serves.
