"""Hugging Face Hub-backed dataset access — the remote alternative to
data_loader.py's local-disk reads, for when this app runs somewhere that
doesn't have the 28GB ToothFairy3 dataset on local disk (a hosted
deployment). Mirrors db_backend.py's shape: is_configured() gates
whether app.py uses this module or the local-disk one, auto-detected
from secrets, never erroring when secrets are simply absent.

Case files are downloaded on first access via hf_hub_download, which
caches them locally (~/.cache/huggingface/hub by default) so repeat
views of the same case in one session don't re-download. Known
limitation: that cache isn't bounded here, so a long session touching
many different cases could accumulate a lot of local disk use on a
hosted container — worth watching if disk space becomes an issue (see
README).
"""
from __future__ import annotations

import re

import streamlit as st
from huggingface_hub import HfApi, hf_hub_download

from .data_loader import Case

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
    """Downloads (cached) and returns a real local path nibabel can open."""
    repo_id, token = _repo_and_token()
    return hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=repo_relative_path, token=token)


def load_label_names(cfg: dict) -> dict[int, str]:
    import json
    path = resolve_local_path(cfg["data"]["dataset_json_rel"])
    with open(path, "r", encoding="utf-8") as f:
        ds = json.load(f)
    return {int(v): k for k, v in ds["labels"].items()}
