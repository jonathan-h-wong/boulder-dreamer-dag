"""
Stage registry + content-addressed cache keys.

Every stage declares what its output depends on:
  - code:        source paths whose *semantics* feed the output (AST-normalized, so
                 comments, formatting and docstrings don't invalidate)
  - config_keys: dotted paths into pipeline.yaml that the stage reads
  - pins:        upstream model-code commits from common/pins.py
  - deps:        upstream stages, or external inputs:
                   raw_video     — ETag of raw/videos/{video_id}.mp4
                   raw_metadata  — hash of the climber fields (height_cm, ape_index_cm) from raw/metadata CSVs
                   hold_catalog  — cache key of the current global hold catalog

cache_key = sha256(stage, code hash, config subset, pins, {dep: upstream cache_key})

Because each upstream key already covers *its* code/config/upstreams, keys chain
(Merkle-style): changing anything invalidates that stage and everything downstream,
and the final stage's key identifies a sample's entire lineage.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal

from common import pins as _pins

REPO_ROOT = Path(__file__).resolve().parents[1]

EXTERNAL_INPUTS = ("raw_video", "raw_metadata", "hold_catalog")


@dataclass(frozen=True)
class Stage:
    name: str
    runner: Literal["modal", "agent"]
    scope: Literal["video", "global"]
    root: Literal["filtered_videos", "processed", "hold_catalog"]  # s3.prefixes key for outputs
    deps: tuple[str, ...] = ()
    config_keys: tuple[str, ...] = ()
    code: tuple[str, ...] = ()
    pins: tuple[str, ...] = ()


def _video(name, runner, root="processed", **kw) -> Stage:
    # Agent outputs are shaped by the shared request/validation code and the output schemas.
    default = ((f"agents/{name}", "agents/base.py", "common/schemas.py") if runner == "agent"
               else (f"jobs/{name}",))
    code = kw.pop("code", default)
    return Stage(name=name, runner=runner, scope="video", root=root, code=code, **kw)


_AGENT_CFG = ("agents.model", "agents.max_tokens")

STAGES: dict[str, Stage] = {s.name: s for s in [
    # --- filter flow -----------------------------------------------------------
    _video("extract_filter_frames", "modal", root="filtered_videos",
           deps=("raw_video",), config_keys=("filter.n_frames",)),
    _video("filter_video", "agent", root="filtered_videos",
           deps=("extract_filter_frames",), config_keys=(*_AGENT_CFG, "agents.effort")),

    # --- dataset flow: scene perception ------------------------------------------
    _video("extract_bkg_plate", "modal",
           deps=("raw_video",), config_keys=("downsampling.bkg_plate_rate",)),
    _video("extract_depth", "modal",
           deps=("raw_video",), config_keys=("downsampling.depth_rate", "models.moge3"),
           pins=("MOGE_REF",)),
    _video("extract_poses", "modal",
           deps=("raw_video", "extract_depth"),   # GVHMR uses depth's static-camera K
           config_keys=("downsampling.keypose_rate", "models.gvhmr"), pins=("GVHMR_REF",)),

    # --- dataset flow: hold perception (set-of-marks + geometric contacts) --------
    _video("extract_hold_proposals", "modal",
           deps=("extract_bkg_plate",),
           config_keys=("hold_proposals", "models.sam2"), pins=("SAM2_REF",)),
    _video("extract_hold_descriptions", "agent",
           deps=("extract_bkg_plate", "extract_hold_proposals"),
           config_keys=(*_AGENT_CFG, "agents.effort")),
    _video("extract_hold_crops", "modal",
           deps=("extract_bkg_plate", "extract_hold_proposals", "extract_hold_descriptions"),
           config_keys=("models.marigold_v2",), pins=("MARIGOLD_V2_REF",)),
    _video("extract_hold_contacts", "modal",
           deps=("extract_poses", "extract_depth", "extract_hold_proposals",
                 "extract_hold_descriptions"),
           config_keys=("hold_contacts", "models.gvhmr")),

    # --- dataset flow: geometry shared by the catalog and the scenegraph ------------
    _video("extract_grounded_poses", "modal",
           deps=("extract_poses", "extract_depth"), config_keys=("models.gvhmr",)),
    _video("extract_wall_mesh", "modal",   # also writes bare-wall depth + hold_frames.json
           deps=("extract_depth", "extract_hold_crops", "extract_hold_contacts")),

    # --- global hold catalog (deps = every member video's extract_wall_mesh, which
    #     transitively covers its hold descriptions, crops and contacts) --------------
    Stage(name="curate_hold_catalog", runner="modal", scope="global", root="hold_catalog",
          config_keys=("hold_catalog", "hold_fit", *_AGENT_CFG),
          code=("jobs/curate_hold_catalog", "common/hold_fit.py")),

    # --- dataset flow: reconstruction ---------------------------------------------
    _video("extract_scenegraph", "modal",
           deps=("extract_bkg_plate", "extract_depth", "extract_hold_descriptions",
                 "extract_hold_crops", "extract_hold_contacts", "extract_wall_mesh",
                 "hold_catalog"),   # normals come from extract_hold_crops' bkg_normals
           config_keys=("hold_fit",),
           code=("jobs/extract_scenegraph", "common/hold_fit.py")),

    # --- dataset flow: grounding --------------------------------------------------
    _video("post_processing", "modal",
           deps=("extract_wall_mesh", "extract_grounded_poses", "extract_scenegraph",
                 "raw_metadata")),
]}


# ---------------------------------------------------------------------------
# Code hashing
# ---------------------------------------------------------------------------

def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def _py_files(rel: str) -> list[Path]:
    path = REPO_ROOT / rel
    files = [path] if path.is_file() else sorted(path.rglob("*.py"))
    # job.py is Modal plumbing (resources, image choice) — not output-affecting logic.
    return [f for f in files if f.name != "job.py"]


@lru_cache(maxsize=None)
def code_hash(stage_name: str) -> str:
    h = hashlib.sha256()
    for rel in STAGES[stage_name].code:
        files = _py_files(rel)
        if not files:
            raise FileNotFoundError(f"{stage_name}: no source files under {rel}")
        for f in files:
            tree = _strip_docstrings(ast.parse(f.read_text(encoding="utf-8")))
            h.update(f.relative_to(REPO_ROOT).as_posix().encode())
            h.update(ast.dump(tree, include_attributes=False).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------

def config_subset(stage_name: str, config: dict) -> dict:
    out = {}
    for dotted in STAGES[stage_name].config_keys:
        node = config
        for part in dotted.split("."):
            node = node[part]
        out[dotted] = node
    return out


def lineage(stage_name: str, config: dict, upstream: dict[str, str]) -> dict:
    """Everything the stage's output is a function of. Stored in the marker for debugging."""
    stage = STAGES[stage_name]
    return {
        "stage": stage_name,
        "code": code_hash(stage_name),
        "config": config_subset(stage_name, config),
        "pins": {p: getattr(_pins, p) for p in stage.pins},
        "upstream": dict(sorted(upstream.items())),
    }


def cache_key(lin: dict) -> str:
    blob = json.dumps(lin, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


class KeyResolver:
    """
    Lazily computes (and memoizes) cache keys for per-video stages.
    `external(name, video_id)` must return the identity of raw_video / raw_metadata /
    hold_catalog (an ETag or cache key; None if missing).
    """

    def __init__(self, config: dict, external: Callable[[str, str], str | None]):
        self.config = config
        self.external = external
        self._memo: dict[tuple[str, str], tuple[str, dict]] = {}

    def lineage(self, stage_name: str, video_id: str) -> dict:
        return self._resolve(stage_name, video_id)[1]

    def key(self, stage_name: str, video_id: str) -> str:
        return self._resolve(stage_name, video_id)[0]

    def _resolve(self, stage_name: str, video_id: str) -> tuple[str, dict]:
        memo_key = (stage_name, video_id)
        if memo_key not in self._memo:
            upstream = {
                dep: (self.external(dep, video_id) if dep in EXTERNAL_INPUTS
                      else self.key(dep, video_id))
                for dep in STAGES[stage_name].deps
            }
            lin = lineage(stage_name, self.config, upstream)
            self._memo[memo_key] = (cache_key(lin), lin)
        return self._memo[memo_key]

    def invalidate_external(self, name: str):
        """Drop memoized keys that (transitively) depend on an external input, e.g. after
        the hold catalog is rebuilt."""
        def depends(stage_name: str) -> bool:
            return any(d == name or (d in STAGES and depends(d)) for d in STAGES[stage_name].deps)
        self._memo = {k: v for k, v in self._memo.items() if not depends(k[0])}
