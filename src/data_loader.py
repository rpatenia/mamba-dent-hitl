"""Config loading, case discovery, and label-name/color lookup.

Case discovery and layer file paths are driven entirely by config.yaml —
nothing here hardcodes "cbct.nii.gz" / "apex.nii.gz" style filenames, so
pointing this at a differently-laid-out dataset is a config edit, not a
code change.
"""
from __future__ import annotations

import colorsys
import json
import os
import re
from dataclasses import dataclass, field

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # results.csv_path is relative to the project root (this file's
    # grandparent), not the CWD streamlit happens to be launched from.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    cfg["_project_root"] = project_root
    csv_rel = cfg["results"]["csv_path"]
    if not os.path.isabs(csv_rel):
        cfg["results"]["csv_path"] = os.path.join(project_root, csv_rel)
    return cfg


@dataclass(frozen=True)
class Case:
    case_id: str
    image_path: str
    # layer key -> resolved file path, only for layers whose file actually
    # exists for this case (a layer configured but not yet generated for
    # this case is simply absent, not an error)
    layer_paths: dict[str, str] = field(default_factory=dict)


def discover_cases(cfg: dict) -> list[Case]:
    data_root = cfg["data"]["root"]
    images_dir = os.path.join(data_root, cfg["data"]["images_dir"])
    pattern = re.compile(cfg["data"]["image_pattern"])

    cases: list[Case] = []
    for fname in sorted(os.listdir(images_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        case_id = m.group("case_id")
        image_path = os.path.join(images_dir, fname)

        layer_paths = {}
        for layer in cfg["layers"]:
            layer_dir = layer.get("dir")
            if not layer_dir:
                continue  # layer not wired up yet (e.g. apex, nerve_centerline)
            candidate = os.path.join(data_root, layer_dir, f"{case_id}{layer['suffix']}")
            if os.path.exists(candidate):
                layer_paths[layer["key"]] = candidate

        cases.append(Case(case_id=case_id, image_path=image_path, layer_paths=layer_paths))
    return cases


def reviewable_layer_keys(cfg: dict) -> list[str]:
    return [l["key"] for l in cfg["layers"] if l.get("reviewable")]


def load_label_names(cfg: dict) -> dict[int, str]:
    """id -> anatomical name, straight from ToothFairy3's own dataset.json
    (the authoritative source — not a derived/cached copy)."""
    with open(cfg["data"]["dataset_json"], "r", encoding="utf-8") as f:
        ds = json.load(f)
    return {int(v): k for k, v in ds["labels"].items()}


def build_label_colors(label_ids: list[int]) -> dict[int, str]:
    """Deterministic per-ID color (stable across cases/sessions) so a
    reviewer learns "this color = this tooth" over a session, rather than
    colors reshuffling case to case. 0 (background) is left uncolored.
    """
    ids = sorted(i for i in label_ids if i != 0)
    colors = {0: "rgba(0,0,0,0)"}
    n = max(len(ids), 1)
    for idx, label_id in enumerate(ids):
        hue = idx / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        colors[label_id] = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
    return colors
