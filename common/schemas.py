"""
Pydantic schemas for inter-job data contracts.
These define the shape of metadata written/read between pipeline stages.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Hold Catalog Schemas
# ---------------------------------------------------------------------------

class HoldType(str, Enum):
    jug = "jug"
    crimp_edge = "crimp_edge"
    rail = "rail"
    sloper = "sloper"
    pinch = "pinch"
    pocket = "pocket"
    horn_knob = "horn_knob"
    foot_chip = "foot_chip"
    volume_pyramid = "volume_pyramid"
    volume_dome = "volume_dome"
    volume_other = "volume_other"
    macro_slab = "macro_slab"
    wedge_chip = "wedge_chip"
    flake = "flake"
    other = "other"


class HoldOutline(str, Enum):
    round = "round"
    oval = "oval"
    elongated = "elongated"
    rectangular = "rectangular"
    triangular = "triangular"
    irregular = "irregular"
    crescent = "crescent"


class HoldProfile(str, Enum):
    convex_dome = "convex_dome"
    flat_top = "flat_top"
    dished = "dished"
    hollowed = "hollowed"
    ridge = "ridge"
    faceted = "faceted"
    mixed = "mixed"


class HoldLip(str, Enum):
    none = "none"
    rounded = "rounded"
    sharp = "sharp"
    incut = "incut"


class HoldOpening(str, Enum):
    none = "none"
    single_finger = "single_finger"
    two_finger = "two_finger"
    three_finger = "three_finger"
    hand_scoop = "hand_scoop"
    slot = "slot"


class HoldSymmetry(str, Enum):
    bilateral = "bilateral"
    radial = "radial"
    none = "none"


class HoldRelativeSize(str, Enum):
    tiny = "tiny"
    small = "small"
    medium = "medium"
    large = "large"
    volume = "volume"


class HoldDepthClass(str, Enum):
    shallow = "shallow"
    medium = "medium"
    deep = "deep"
    very_deep = "very_deep"


class HoldDescription(BaseModel):
    """Schema for a single detected hold, output by extract_hold_descriptions agent."""
    hold_id: str
    hold_type: HoldType
    outline: HoldOutline
    profile: HoldProfile
    lip: HoldLip
    opening: HoldOpening
    symmetry: HoldSymmetry
    relative_size: HoldRelativeSize
    depth_class: HoldDepthClass
    description: str = Field(..., max_length=500, description="Free text, max two sentences.")
    # Bounding box in pixel coords: [x_min, y_min, x_max, y_max]
    bbox: list[int] = Field(..., min_length=4, max_length=4)


class HoldDescriptionWithRoute(HoldDescription):
    """Extension for per-video hold detection — adds on-route flag."""
    on_route: bool = Field(..., description="True if touched by climber hands or feet.")


# ---------------------------------------------------------------------------
# Video Filtering Schemas
# ---------------------------------------------------------------------------

class VideoFilterResult(BaseModel):
    video_id: str
    passed: bool
    too_many_climbers: bool = False
    camera_blur: bool = False
    moving_camera: bool = False
    no_floor_plane: bool = False
    missing_climber_metadata: bool = False
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Depth / Intrinsics Schemas
# ---------------------------------------------------------------------------

class CameraIntrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
