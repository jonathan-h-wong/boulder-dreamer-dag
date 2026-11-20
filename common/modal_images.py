"""
Reusable Modal Image + Volume definitions. This is the single source of truth for
job dependencies — there are no per-job requirements.txt files.

Usage (in jobs/<name>/job.py):
    from common.modal_images import gpu_moge, WEIGHTS_MOUNT, weights_volume

Every exported image ends with `_with_src(...)`, which ships the repo's own packages
(common/, jobs/, agents/) into the container. Modal requires add_local_* to be the
final build step, so derive new images from the private `_` bases, never from an
exported image.

Model weights live on the `boulderdreamer-weights` Volume mounted at WEIGHTS_MOUNT,
not in the images. HF_HOME / TORCH_HOME point there so hub downloads are cached
across runs. Licensed assets (SMPL / SMPL-X body models, GVHMR checkpoints) must be
uploaded once by hand (create the volume first: `modal volume create boulderdreamer-weights`):
    /body_models/smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.npz
    /gvhmr/{gvhmr,hmr2,vitpose,yolo}/...   (GVHMR's inputs/checkpoints layout)
"""

from pathlib import Path

import modal

from common.pins import GVHMR_REF, MARIGOLD_V2_REF, MOGE_REF, SAM2_REF

WEIGHTS_MOUNT = "/weights"
weights_volume = modal.Volume.from_name("boulderdreamer-weights", create_if_missing=True)

# Credentials, one Modal Secret per provider. Every job attaches all of them:
# aws-secret          AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
# huggingface-secret  HF_TOKEN
# anthropic-secret    ANTHROPIC_API_KEY
SECRET_NAMES = ("aws-secret", "huggingface-secret", "anthropic-secret")
SECRETS = [modal.Secret.from_name(n) for n in SECRET_NAMES]

_CACHE_ENV = {
    "HF_HOME": f"{WEIGHTS_MOUNT}/hf",
    "TORCH_HOME": f"{WEIGHTS_MOUNT}/torch",
}

_COMMON_PY = [
    "numpy",
    "opencv-python-headless",
    "Pillow",
    "scipy",
    "boto3",
    "pydantic>=2.0",
]


def _with_src(image: modal.Image) -> modal.Image:
    return image.add_local_python_source("common", "jobs", "agents")


# ---------------------------------------------------------------------------
# CPU images
# ---------------------------------------------------------------------------
_cpu = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_COMMON_PY, "scikit-learn", "anthropic>=0.40")
)

# Lightweight: filter_video, extract_bkg_plate
cpu_base = _with_src(_cpu)

# Geometry: extract_grounded_poses, extract_wall_mesh, post_processing
cpu_geometry = _with_src(
    _cpu.apt_install("libgl1", "libglib2.0-0")  # open3d runtime deps
    .pip_install("trimesh", "open3d", "smplx")
    .env(_CACHE_ENV)
)

# ---------------------------------------------------------------------------
# GPU images
# The devel image provides nvcc/gcc for packages that compile extensions.
# nvidia/cuda images ship without Python, hence add_python.
# ---------------------------------------------------------------------------
_CUDA_IMAGE = "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04"


def _gpu_root(python_version: str) -> modal.Image:
    return (
        modal.Image.from_registry(_CUDA_IMAGE, add_python=python_version)
        .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
        .env(_CACHE_ENV)
    )


# torch 2.10 / cu128: what Marigold V2 pins, and new enough for MoGe-3 (FlexGEMM/Triton)
# and SAM2 (>= 2.5.1). Each model still gets its own image so their pins can't collide.
_gpu_torch = (
    _gpu_root("3.11")
    .pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(*_COMMON_PY)
)

# MoGe-3 depth + intrinsics. FlexGEMM (a MoGe-3 dependency) builds on Triton.
gpu_moge = _with_src(_gpu_torch.pip_install(f"git+https://github.com/microsoft/MoGe.git@{MOGE_REF}"))

# SAM2 automatic masks (hold proposals). Its optional CUDA extension (mask
# post-processing) is skipped to avoid build failures.
gpu_sam2 = _with_src(
    _gpu_torch.env({"SAM2_BUILD_CUDA": "0"}).pip_install(
        f"git+https://github.com/facebookresearch/sam2.git@{SAM2_REF}",
        "huggingface_hub",   # SAM2's from_pretrained; not in its install_requires
    )
)

# Marigold V2 normals (hold crops): a LoRA on Qwen-Image-Edit-2509, installed from its repo
# (not on PyPI). Its scripts (download_assets.py, infer.py) read and write assets under
# $DEPTH_ASSETS_DIR, pointed at the weights volume so the ~71 GB download (Qwen base 54 GB +
# Marigold checkpoints 17 GB) happens once; already populated there. The repo ships its own
# assets/ folder, so a symlink in its place would land inside it instead.
# ~17 GB VRAM at 1024^2, ~29 GB at 2048^2.
gpu_marigold = _with_src(
    _gpu_torch.run_commands(
        "git clone https://github.com/huawei-bayerlab/marigold-v2.git /opt/marigold-v2"
        f" && cd /opt/marigold-v2 && git checkout {MARIGOLD_V2_REF}",
        "cd /opt/marigold-v2 && pip install -e .",
    ).env({"DEPTH_ASSETS_DIR": f"{WEIGHTS_MOUNT}/marigold-v2-assets"})
)

# Hold fitting (jobs/hold_fit.py): differentiable SDF ray marching in plain torch.
# Shared by curate_hold_catalog and extract_scenegraph so both fit with identical code;
# the catalog also needs clustering, the text encoder and the Claude SDK. Normal
# supervision reuses extract_hold_crops' bkg_normals.png, so no Marigold here.
gpu_fit = _with_src(
    _gpu_torch.pip_install(
        "trimesh", "open3d", "scikit-learn", "hdbscan", "sentence-transformers",
        "pyarrow", "anthropic>=0.40",
    )
)

# GVHMR: not on PyPI. Its requirements pin torch 2.3.0+cu121 and a pytorch3d wheel
# built for CPython 3.10, so it gets its own Python 3.10 root instead of _gpu_torch.
# Checkpoints + SMPL/SMPL-X body models are expected on the weights volume.
gpu_gvhmr = _with_src(
    _gpu_root("3.10")
    .run_commands(
        f"git clone https://github.com/zju3dv/GVHMR.git /opt/GVHMR"
        f" && cd /opt/GVHMR && git checkout {GVHMR_REF}",
        "cd /opt/GVHMR && pip install -r requirements.txt && pip install -e .",
    )
    .pip_install("boto3", "pydantic>=2.0")
)

# ---------------------------------------------------------------------------
# Orchestrator image (scripts/run_flow_on_modal.py): runs a Prefect flow, which only
# dispatches work (S3 reads, Modal calls, Claude batches), so no ML packages. Python 3.12
# must match the local .venv: cache keys hash Python ASTs, which can differ across
# Python versions. Also ships flows/ and the config, which job images don't need.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]

orchestrator = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("prefect==3.9.3", "anthropic==2.8.0", "modal==1.6.4", "pyyaml", "boto3",
                 "pydantic>=2.0", "numpy",
                 "opencv-python-headless")   # benchmark renders (scripts/benchmark_set_of_marks.py)
    .env({"PREFECT_HOME": "/tmp/prefect", "PREFECT_SERVER_ALLOW_EPHEMERAL_MODE": "true"})
    .add_local_file(_REPO_ROOT / "configs" / "pipeline.yaml", "/root/configs/pipeline.yaml")
    .add_local_python_source("common", "jobs", "agents", "flows")
)
