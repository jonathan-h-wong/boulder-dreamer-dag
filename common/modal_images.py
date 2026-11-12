"""
Reusable Modal Image definitions.
Import these into job/job.py files instead of re-defining images per job.

Usage:
    from common.modal_images import cpu_base, gpu_moge, gpu_gvhmr
"""

import modal

# ---------------------------------------------------------------------------
# CPU base — lightweight, used by most CPU batch jobs
# ---------------------------------------------------------------------------
cpu_base = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "opencv-python-headless",
        "Pillow",
        "scipy",
        "scikit-learn",
        "boto3",
        "pydantic>=2.0",
        "anthropic>=0.30",
    )
)

# ---------------------------------------------------------------------------
# GPU base — CUDA 12.1, shared across GPU jobs
# ---------------------------------------------------------------------------
gpu_base = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "numpy",
        "opencv-python-headless",
        "Pillow",
        "scipy",
        "boto3",
        "pydantic>=2.0",
    )
)

# ---------------------------------------------------------------------------
# MoGe-2 depth model image
# ---------------------------------------------------------------------------
gpu_moge = gpu_base.pip_install("moge")  # TODO: pin to exact release

# ---------------------------------------------------------------------------
# GVHMR pose estimation image
# ---------------------------------------------------------------------------
gpu_gvhmr = (
    gpu_base
    .pip_install("gvhmr")  # TODO: pin to exact release
    .run_commands("pip install smplx")
)

# ---------------------------------------------------------------------------
# SAM2 + Marigold V2 image (hold crops, hold maps)
# ---------------------------------------------------------------------------
gpu_sam2_marigold = (
    gpu_base
    .pip_install(
        "segment-anything-2",
        "diffusers>=0.27",
        "transformers>=4.40",
        "accelerate",
    )
)

# ---------------------------------------------------------------------------
# Scenegraph / differentiable rendering image
# ---------------------------------------------------------------------------
gpu_scenegraph = (
    gpu_base
    .pip_install(
        "gsplat",         # gaussian splatting
        "trimesh",
        "open3d",
        "diffusers>=0.27",
        "transformers>=4.40",
    )
)
