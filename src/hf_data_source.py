"""Hugging Face Hub-backed dataset access — the remote alternative to
data_loader.py's local-disk reads, for when this app runs somewhere that
doesn't have the 28GB ToothFairy3 dataset on local disk (a hosted
deployment). Mirrors db_backend.py's shape: is_configured() gates
whether app.py uses this module or the local-disk one, auto-detected
from secrets, never erroring when secrets are simply absent.

Case files are downloaded via hf_hub_download into a fixed local_dir
(not the default global ~/.cache/huggingface/hub) specifically so
app.py can delete each file right after loading it into memory —
bounding disk use to roughly one case at a time. This mattered for real:
a hosted deploy kept crashing after some active use with zero log
output (an OS-level resource kill, not a Python exception), and an
ever-growing download cache on a container whose writable filesystem
may be memory-backed is a very plausible contributor on top of the
per-case in-memory footprint (see app.py's _load_case_volumes docstring
for that half of the fix).
"""
from __future__ import annotations

import os
import re
import tempfile

import streamlit as st
from huggingface_hub import HfApi, hf_hub_download

from .data_loader import Case

# Not the default ~/.cache/huggingface/hub — a dedicated directory we
# fully control the lifecycle of, so cleanup_local_path can just
# os.remove() without touching huggingface_hub's shared content-store.
_LOCAL_DIR = os.path.join(tempfile.gettempdir(), "mamba_dent_hitl_hub_download")

SECRETS_KEY = "huggingface"  # [huggingface] section in secrets.toml


def is_configured() -> bool:
    try:
        sec = st.secrets[SECRETS_KEY]
        return "token" in sec and "repo_id" in sec
    except Exception:
        return False


def _repo_and_token() -> tuple[str, str]:
    sec = st.secrets[SECRETS_KEY]
    return sec["repo_id"], sec["token"]


@st.cache_resource(show_spinner=False)
def _api() -> HfApi:
    _repo_id, token = _repo_and_token()
    return HfApi(token=token)


@st.cache_data(show_spinner="Listing dataset on Hugging Face Hub...")
def discover_cases(cfg: dict) -> list[Case]:
    repo_id, _token = _repo_and_token()
    files = set(_api().list_repo_files(repo_id, repo_type="dataset"))

    images_prefix = cfg["data"]["images_dir"] + "/"
    pattern = re.compile(cfg["data"]["image_pattern"])

    cases: list[Case] = []
    for f in sorted(files):
        if not f.startswith(images_prefix):
            continue
        fname = f[len(images_prefix):]
        m = pattern.match(fname)
        if not m:
            continue
        case_id = m.group("case_id")

        layer_paths = {}
        for layer in cfg["layers"]:
            layer_dir = layer.get("dir")
            if not layer_dir:
                continue
            candidate = f"{layer_dir}/{case_id}{layer['suffix']}"
            if candidate in files:
                layer_paths[layer["key"]] = candidate  # repo-relative, resolved lazily

        cases.append(Case(case_id=case_id, image_path=f, layer_paths=layer_paths))
    return cases


def resolve_local_path(repo_relative_path: str) -> str:
    """Downloads and returns a real local path nibabel can open. Caller
    should call cleanup_local_path() on the result once its content has
    been read into memory — see module docstring for why."""
    repo_id, token = _repo_and_token()
    return hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=repo_relative_path,
        token=token, local_dir=_LOCAL_DIR,
    )


def cleanup_local_path(local_path: str) -> None:
    """Delete a file resolve_local_path returned, once its content is no
    longer needed on disk (i.e. already loaded into memory). Best-effort
    — a failed cleanup shouldn't break the app, just leave a bit of
    disk around."""
    try:
        os.remove(local_path)
    except OSError:
        pass


def load_label_names(cfg: dict) -> dict[int, str]:
    import json
    path = resolve_local_path(cfg["data"]["dataset_json_rel"])
    with open(path, "r", encoding="utf-8") as f:
        ds = json.load(f)
    return {int(v): k for k, v in ds["labels"].items()}
