"""
Shared hold-fitting procedure.

Used by BOTH curate_hold_catalog (fitting catalog exemplars, then every member hold) and
extract_scenegraph (fitting a video's on-route holds), so a catalog parameter vector and
a scenegraph parameter vector mean exactly the same thing. Both stages hash this file
into their cache keys: changing the procedure invalidates the catalog and every
scenegraph together.

Parameterization (per hold)
---------------------------
frame       FIXED, from extract_wall_mesh's hold_frames.json (same for catalog + scenegraph)
            origin  mask centroid lifted onto the bare-wall surface (all holds inpainted)
            normal  local plane fit of the bare wall around the hold, or of the volume
                    face for holds with mounted_on set
            tangent mask principal axis (2D PCA) projected into that plane
extrinsics  FITTED, small residuals relative to the frame
            theta   in-plane rotation about the normal
            tilt    2-dof rotation of the hold's base normal away from the frame normal
            offset  3-dof translation of the base
intrinsics  FITTED, the cluster's procedural shape program parameters (parametric_holds.py)

Expressing extrinsics relative to the local frame keeps them small at init and keeps
intrinsics comparable across holds, which is what makes the catalog kNN warm start
meaningful.

Initialization
--------------
extrinsics  theta multi-start by symmetry (radial: 1 start, bilateral: 2, none: 4);
            tilt = 0; offset = 0. Never copied from catalog neighbors: setters mount the
            same hold at any rotation, so a neighbor's orientation is not a prior.
intrinsics  catalog:    program defaults (or Claude-proposed values for exemplars)
            scenegraph: kNN-weighted mean of the neighbors' fitted intrinsics
The best start by final loss wins.

Rendering, target, loss
-----------------------
render  bare wall (bare_wall_depth, every hold inpainted) + SDFs of the holds being
        fitted, differentiable SDF ray marching with the shared intrinsics.json K
        -> silhouette, depth, normals
target  the REAL observations: bkg_depth, bkg_normals.png, hold masks. The bumps of the
        fitted holds are what the SDFs must explain.
window  each fitted hold's mask dilated by window_margin_px, MINUS the masks of holds
        not being fitted. Inside those masks this is equivalent to inpainting them, with
        no second depth map, and the same rule covers both callers:
          scenegraph: fitted = on-route holds (with the volumes they're mounted on)
          catalog:    fitted = the single hold; every neighbor is excluded
  loss = w_iou    * (1 - soft IoU(silhouette, hold mask))
       + w_normal * (1 - mean cos(rendered normal, bkg_normals))   in the window
       + w_depth  * L1(rendered depth, bkg_depth)                  in the window
       + w_points * mean |SDF(p)|  for p = observed depth inside the hold mask lifted to
                    3D with K (zero on the surface; no ray marching, and direct 3D
                    gradients even from a poor init)
       + w_reg    * ||extrinsic residuals||^2
Accepted when the fit score (IoU / normal agreement) >= hold_fit.min_score.
Weights, iterations, learning rate and the threshold come from config `hold_fit`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class HoldFrame:
    hold_id: str
    origin: np.ndarray          # (3,) camera space, metres
    normal: np.ndarray          # (3,) unit, pointing out of the wall toward the camera
    tangent: np.ndarray         # (3,) unit, in-plane initial "up" axis of the hold
    source: str                 # "wall" | "volume"


@dataclass
class FitParams:
    theta: float = 0.0
    tilt: np.ndarray = field(default_factory=lambda: np.zeros(2))
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    intrinsics: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class FitResult:
    params: FitParams
    score: float                # IoU / normal-agreement score compared to min_score
    loss: float
    accepted: bool


THETA_STARTS = {
    "radial": (0.0,),
    "bilateral": (0.0, np.pi),
    "none": (0.0, np.pi / 2, np.pi, 3 * np.pi / 2),
}


def initial_params(symmetry: str, intrinsics_init: np.ndarray) -> list[FitParams]:
    """One FitParams per theta start; tilt and offset start at zero."""
    return [FitParams(theta=t, intrinsics=intrinsics_init.copy()) for t in THETA_STARTS[symmetry]]


def fit_hold(program, frame: HoldFrame, starts: list[FitParams], observations: dict,
             K: np.ndarray, config: dict) -> FitResult:
    """
    program:      the cluster's procedural SDF program (from parametric_holds.py)
    observations: {"mask", "normals", "depth", "bare_depth", "window", "points"} cropped
                  to the hold; window already excludes holds not being fitted
    """
    # TODO: implement differentiable SDF ray marching + optimization (torch)
    raise NotImplementedError
