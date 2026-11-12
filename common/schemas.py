"""
Pydantic schemas for inter-job data contracts.
These define the shape of metadata written/read between pipeline stages.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


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


# --- Agent output: what Claude judges. These models generate the structured-output JSON
# schema (agents.base.output_format) and re-validate the response. They contain no ids or
# coordinates: those come from SAM2 proposals and are added in code.

class HoldLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mark_ids: list[int] = Field(
        ..., min_length=1,
        description="Numbers of every overlay region that together make up this one hold or "
                    "volume (usually one; several if the segmenter split it).",
    )
    hold_type: HoldType
    outline: HoldOutline
    profile: HoldProfile
    lip: HoldLip
    opening: HoldOpening
    symmetry: HoldSymmetry
    relative_size: HoldRelativeSize
    depth_class: HoldDepthClass
    description: str = Field(..., max_length=300, description="Free text, at most two sentences.")


class HoldLabelResponse(BaseModel):
    """Agent output for extract_hold_descriptions."""
    model_config = ConfigDict(extra="forbid")

    holds: list[HoldLabel]
    non_hold_marks: list[int] = Field(
        ..., description="Numbers of regions that are not holds or volumes on the target wall "
                         "(bare wall, tape, labels, people, holds on further walls).",
    )

    def check_coverage(self, mark_ids: set[int]) -> None:
        """Every proposal mark must be accounted for exactly once."""
        seen = [m for h in self.holds for m in h.mark_ids] + list(self.non_hold_marks)
        dupes = sorted({m for m in seen if seen.count(m) > 1})
        unknown = sorted(set(seen) - mark_ids)
        missing = sorted(mark_ids - set(seen))
        if dupes or unknown or missing:
            raise ValueError(f"mark coverage: duplicated={dupes} unknown={unknown} missing={missing}")


# --- Records written to S3 (agent judgment + code-derived fields).

class HoldRecord(BaseModel):
    """One row of processed/{video_id}/holds/hold_descriptions.csv."""
    hold_id: str
    mark_ids: list[int]
    bbox: list[int] = Field(..., min_length=4, max_length=4, description="[x_min, y_min, x_max, y_max] px")
    hold_type: HoldType
    outline: HoldOutline
    profile: HoldProfile
    lip: HoldLip
    opening: HoldOpening
    symmetry: HoldSymmetry
    relative_size: HoldRelativeSize
    depth_class: HoldDepthClass
    description: str
    # Filled by extract_hold_contacts (geometry), None until then.
    on_route: Optional[bool] = None
    mounted_on: Optional[str] = Field(None, description="hold_id of the volume this hold is mounted on")

    @property
    def is_volume(self) -> bool:
        return self.hold_type in (HoldType.volume_pyramid, HoldType.volume_dome, HoldType.volume_other)


# ---------------------------------------------------------------------------
# Video Filtering Schemas
# ---------------------------------------------------------------------------

class FilterJudgment(BaseModel):
    """Agent output for filter_video. True means the video FAILS that criterion."""
    model_config = ConfigDict(extra="forbid")

    too_many_climbers: bool = Field(..., description="Two or more people climbing or on the wall at the same time in any frame.")
    camera_blur: bool = Field(..., description="Frames are significantly blurry or out of focus.")
    moving_camera: bool = Field(..., description="The camera pans, tilts, zooms or is handheld rather than static.")
    no_floor_plane: bool = Field(..., description="The gym floor is not visible in any frame.")
    notes: str = Field(..., max_length=300, description="One short sentence explaining any failure; empty if none.")


class VideoFilterResult(BaseModel):
    """filtered_videos/{video_id}/filter_result.json — judgment + code-derived fields."""
    video_id: str
    passed: bool
    too_many_climbers: bool = False
    camera_blur: bool = False
    moving_camera: bool = False
    no_floor_plane: bool = False
    missing_climber_metadata: bool = False
    notes: Optional[str] = None

    @classmethod
    def from_judgment(cls, video_id: str, j: FilterJudgment) -> "VideoFilterResult":
        failed = j.too_many_climbers or j.camera_blur or j.moving_camera or j.no_floor_plane
        return cls(video_id=video_id, passed=not failed, notes=j.notes or None,
                   **j.model_dump(exclude={"notes"}))


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
