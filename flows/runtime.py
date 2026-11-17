"""
Orchestration runtime shared by all flows.

Flows are stage-major: each stage is dispatched for *all* eligible videos at once —
Modal stages via Function.map (one container per video, run concurrently), agent
stages via a single Claude batch submission (agents.base.run_batch_stage). A video
that fails a stage simply drops out of downstream stages for this run; rerunning the
flow retries only what isn't current.

Skip logic: a (stage, video) is skipped when its marker's cache_key equals the key
computed now (flows/stages.py). Keys are computed here, by the orchestrator, because
only it sees the whole DAG; Modal jobs write their own marker after their outputs.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import io
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

import yaml
from prefect import get_run_logger, task
from prefect.artifacts import create_table_artifact
from prefect.cache_policies import NO_CACHE

from common.idempotency import marker_key, marker_spec, read_marker
from common.modal_jobs import job_fn
from common.s3 import get_client
from flows.stages import EXTERNAL_INPUTS, REPO_ROOT, STAGES, KeyResolver, cache_key, lineage

S3_WORKERS = 32  # parallel marker reads / HEADs


def load_dotenv(path=None) -> None:
    """KEY=VALUE lines from the git-ignored repo-root .env, for local orchestrator runs
    (e.g. ANTHROPIC_API_KEY). Never overrides variables already set in the environment."""
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_config(config_path: str) -> dict:
    load_dotenv()
    with open(REPO_ROOT / config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def git_sha() -> str:
    if os.environ.get("GIT_SHA"):
        return os.environ["GIT_SHA"]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
        return sha + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass
class StageReport:
    stage: str
    cached: int = 0
    ran: int = 0
    failed: dict[str, str] = field(default_factory=dict)


@dataclass
class Context:
    config: dict
    s3: object
    git_sha: str
    inputs: dict[str, dict]                      # video_id -> {"video": etag, "metadata": etag|None}
    catalog_key: str | None = None
    ok: dict[str, set[str]] = field(default_factory=dict)   # stage -> videos current after this run
    reports: list[StageReport] = field(default_factory=list)

    def __post_init__(self):
        self.keys = KeyResolver(self.config, self._external)

    @property
    def bucket(self) -> str:
        return self.config["s3"]["bucket"]

    @property
    def video_ids(self) -> list[str]:
        return sorted(self.inputs)

    def prefix(self, root: str) -> str:
        return self.config["s3"]["prefixes"][root]

    def _external(self, name: str, video_id: str) -> str | None:
        if name == "raw_video":
            return self.inputs[video_id]["video"]
        if name == "raw_metadata":
            return self.inputs[video_id]["metadata"]
        if name == "hold_catalog":
            if self.catalog_key is None:
                raise RuntimeError("hold catalog key requested before ensure_hold_catalog ran")
            return self.catalog_key
        raise KeyError(name)

    def marker_spec(self, stage: str, video_id: str) -> dict:
        return marker_spec(
            bucket=self.bucket,
            region=self.config["s3"]["region"],
            key=marker_key(self.prefix(STAGES[stage].root), stage, video_id),
            cache_key=self.keys.key(stage, video_id),
            body={"stage": stage, "video_id": video_id, "git_sha": self.git_sha,
                  "lineage": self.keys.lineage(stage, video_id)},
        )

    def eligible(self, stage: str, video_ids: list[str] | None = None) -> list[str]:
        """Videos whose upstream stages are all current."""
        deps = [d for d in STAGES[stage].deps if d not in EXTERNAL_INPUTS]
        return [v for v in (video_ids or self.video_ids)
                if all(v in self.ok.get(d, ()) for d in deps)]

    def partition(self, stage: str, video_ids: list[str]) -> tuple[list[str], list[str]]:
        """-> (current, todo) by comparing stored marker keys against computed keys."""
        prefix = self.prefix(STAGES[stage].root)

        def stored(vid):
            m = read_marker(self.s3, self.bucket, marker_key(prefix, stage, vid))
            return m["cache_key"] if m else None

        with ThreadPoolExecutor(S3_WORKERS) as pool:
            stored_keys = list(pool.map(stored, video_ids))
        current, todo = [], []
        for vid, k in zip(video_ids, stored_keys):
            (current if k == self.keys.key(stage, vid) else todo).append(vid)
        return current, todo


CLIMBER_FIELDS = ("height_cm", "ape_index_cm")   # the metadata the pipeline actually uses


def load_climber_metadata(config: dict, s3) -> dict[str, dict]:
    """
    raw/metadata/{scrape_ts}_{gym_slug}.csv -> {video_id: row}. Files are applied in name
    (= scrape timestamp) order, so a later scrape's row for the same video wins.
    """
    bucket, prefix = config["s3"]["bucket"], config["s3"]["prefixes"]["raw_metadata"]
    keys = sorted(
        obj["Key"]
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", []) if obj["Key"].endswith(".csv")
    )

    def read(key):
        text = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        return list(csv.DictReader(io.StringIO(text)))

    with ThreadPoolExecutor(S3_WORKERS) as pool:
        tables = list(pool.map(read, keys))
    rows: dict[str, dict] = {}
    for table in tables:                       # same order as keys: oldest scrape first
        for row in table:
            vid = row["video_key"].rsplit("/", 1)[-1].removesuffix(".mp4")
            rows[vid] = row
    return rows


def climber_info(row: dict | None) -> dict | None:
    """Climber fields used downstream, or None when height is missing (needed for metric scale)."""
    if not row or not row.get("height_cm", "").strip():
        return None
    return {f: (float(row[f]) if row.get(f, "").strip() else None) for f in CLIMBER_FIELDS}


def list_raw_inputs(config: dict, s3) -> dict[str, dict]:
    """
    raw/videos/{video_id}.mp4 + climber metadata ->
      {video_id: {"video": mp4 ETag, "metadata": hash of climber fields | None, "climber": dict | None}}
    The metadata identity hashes only CLIMBER_FIELDS, so unrelated CSV changes (new scrape
    columns, other videos' rows) don't invalidate anything.
    """
    prefix = config["s3"]["prefixes"]["raw_videos"]
    inputs: dict[str, dict] = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=config["s3"]["bucket"], Prefix=prefix):
        for obj in page.get("Contents", []):
            name = obj["Key"][len(prefix):]
            if "/" not in name and name.endswith(".mp4"):
                inputs[name.removesuffix(".mp4")] = {"video": obj["ETag"].strip('"')}
    rows = load_climber_metadata(config, s3)
    for vid, d in inputs.items():
        info = climber_info(rows.get(vid))
        row = rows.get(vid)
        d["climber"] = info
        d["gym"] = row.get("gym_slug") if row else None   # reporting only; not in any key
        d["has_metadata_row"] = row is not None
        d["metadata"] = (hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()[:16]
                         if info else None)
    return inputs


def build_context(config: dict, video_ids: list[str] | None = None) -> Context:
    s3 = get_client(config["s3"]["region"])
    inputs = list_raw_inputs(config, s3)
    if video_ids is not None:
        missing = set(video_ids) - set(inputs)
        if missing:
            raise FileNotFoundError(f"raw videos missing for: {sorted(missing)[:10]}")
        inputs = {v: inputs[v] for v in video_ids}
    return Context(config=config, s3=s3, git_sha=git_sha(), inputs=inputs)


# ---------------------------------------------------------------------------
# Stage runners (Prefect tasks, so each stage shows up on the dashboard)
# ---------------------------------------------------------------------------

def _finish(ctx: Context, report: StageReport, current: set[str]):
    ctx.ok[report.stage] = current
    ctx.reports.append(report)
    log = get_run_logger()
    log.info(f"[{report.stage}] cached={report.cached} ran={report.ran} failed={len(report.failed)}")
    for vid, reason in sorted(report.failed.items())[:20]:
        log.warning(f"[{report.stage}] {vid}: {reason}")


@task(task_run_name="{stage}", cache_policy=NO_CACHE)
def refresh_stage(ctx: Context, stage: str, video_ids: list[str] | None = None) -> set[str]:
    """Record which videos are current for a stage without running anything. Upstream
    markers needn't be checked: the stage's key already encodes its upstream keys."""
    current, _ = ctx.partition(stage, video_ids or ctx.video_ids)
    _finish(ctx, StageReport(stage, cached=len(current)), set(current))
    return set(current)


@task(task_run_name="{stage}", cache_policy=NO_CACHE)
def run_modal_stage(ctx: Context, stage: str, video_ids: list[str] | None = None,
                    before_dispatch=None) -> set[str]:
    """before_dispatch(ctx, todo): optional hook to stage inputs for exactly the videos that
    are about to run (e.g. climber.json for post_processing)."""
    assert STAGES[stage].runner == "modal" and STAGES[stage].scope == "video", stage
    current, todo = ctx.partition(stage, ctx.eligible(stage, video_ids))
    report = StageReport(stage, cached=len(current))
    ok = set(current)
    if todo:
        if before_dispatch:
            before_dispatch(ctx, todo)
        get_run_logger().info(f"[{stage}] dispatching {len(todo)} videos to Modal")
        specs = [ctx.marker_spec(stage, v) for v in todo]
        results = job_fn(stage).map(
            todo, specs, kwargs={"config": ctx.config},
            return_exceptions=True,   # returned as-is (Modal >=1.6 no longer wraps them)
        )
        for vid, res in zip(todo, results):
            if isinstance(res, BaseException):
                report.failed[vid] = repr(res)
            else:
                ok.add(vid)
                report.ran += 1
    _finish(ctx, report, ok)
    return ok


@task(task_run_name="{stage}", cache_policy=NO_CACHE)
def run_agent_stage(ctx: Context, stage: str, video_ids: list[str] | None = None) -> set[str]:
    from agents.base import run_batch_stage

    assert STAGES[stage].runner == "agent", stage
    current, todo = ctx.partition(stage, ctx.eligible(stage, video_ids))
    report = StageReport(stage, cached=len(current))
    ok = set(current)
    if todo:
        agent = importlib.import_module(f"agents.{stage}.agent")
        outcome = run_batch_stage(
            stage=stage,
            items={v: ctx.marker_spec(stage, v) for v in todo},
            build_params=agent.build_params,
            handle_result=agent.handle_result,
            config=ctx.config,
            s3=ctx.s3,
            log=get_run_logger().info,
        )
        ok |= outcome.succeeded
        report.ran = len(outcome.succeeded)
        report.failed = outcome.failed
    _finish(ctx, report, ok)
    return ok


def write_climber_info(ctx: Context, video_ids: list[str]) -> None:
    """processed/{video_id}/climber.json from the same values that feed the raw_metadata
    cache-key input, so post_processing never reads metadata older than its key."""
    prefix = ctx.prefix("processed")

    def put(vid):
        ctx.s3.put_object(Bucket=ctx.bucket, Key=f"{prefix}{vid}/climber.json",
                          Body=json.dumps(ctx.inputs[vid]["climber"]).encode(),
                          ContentType="application/json")

    with ThreadPoolExecutor(S3_WORKERS) as pool:
        list(pool.map(put, video_ids))


# ---------------------------------------------------------------------------
# Global hold catalog
# ---------------------------------------------------------------------------

@task(cache_policy=NO_CACHE)
def ensure_hold_catalog(ctx: Context, rebuild: bool = False,
                        build_if_missing: bool = True) -> str | None:
    """
    Make ctx.catalog_key available to downstream stages.

    The catalog is an explicit snapshot: without `rebuild`, the existing catalog is used
    as-is even if new videos have arrived (they query it, they aren't in it). With
    `rebuild` (or if none exists and build_if_missing), it is rebuilt from every video
    whose extract_wall_mesh is current (its frames are the fitting init; its key covers
    holds, crops, contacts). The catalog key covers those members' keys, so any change in
    membership or upstream holds yields a new key — and invalidates every downstream
    scenegraph. Returns None when there is no catalog and building one wasn't allowed.
    """
    log = get_run_logger()
    stage = "curate_hold_catalog"
    key = marker_key(ctx.prefix("hold_catalog"), stage)
    existing = read_marker(ctx.s3, ctx.bucket, key)

    if existing is None and not rebuild and not build_if_missing:
        log.info(f"[{stage}] no catalog exists and building one was not requested")
        return None

    if existing and not rebuild:
        ctx.catalog_key = existing["cache_key"]
        log.info(f"[{stage}] using existing catalog {ctx.catalog_key} "
                 f"({len(existing.get('members', []))} member videos)")
        # Membership growing is expected (snapshot semantics), but a changed catalog
        # config/code means the snapshot no longer reflects how you'd build it today.
        now = lineage(stage, ctx.config, {})
        then = existing.get("lineage", {})
        if (then.get("code"), then.get("config")) != (now["code"], now["config"]):
            log.warning(f"[{stage}] catalog code/config changed since it was built; "
                        "run with rebuild_catalog=True to apply")
        return ctx.catalog_key

    members = sorted(ctx.ok.get("extract_wall_mesh", ()))
    if not members:
        raise RuntimeError("cannot build hold catalog: no videos with a current wall mesh")
    lin = lineage(stage, ctx.config, {v: ctx.keys.key("extract_wall_mesh", v) for v in members})
    new_key = cache_key(lin)

    if existing and existing["cache_key"] == new_key:
        log.info(f"[{stage}] catalog {new_key} already current")
    else:
        log.info(f"[{stage}] building catalog from {len(members)} videos")
        spec = marker_spec(
            bucket=ctx.bucket, region=ctx.config["s3"]["region"], key=key, cache_key=new_key,
            body={"stage": stage, "git_sha": ctx.git_sha, "members": members, "lineage": lin},
        )
        job_fn(stage).remote(members, spec, ctx.config)

    ctx.catalog_key = new_key
    ctx.keys.invalidate_external("hold_catalog")
    return new_key


# ---------------------------------------------------------------------------
# Reporting + dataset releases
# ---------------------------------------------------------------------------

def publish_report(ctx: Context, artifact_key: str):
    create_table_artifact(
        key=artifact_key,
        table=[{"stage": r.stage, "cached": r.cached, "ran": r.ran, "failed": len(r.failed),
                "failed_examples": ", ".join(sorted(r.failed)[:5])} for r in ctx.reports],
        description=f"Per-stage outcome (git {ctx.git_sha})",
    )


@task(cache_policy=NO_CACHE)
def write_release(ctx: Context, name: str) -> str:
    """
    Snapshot the current dataset as manifests/{name}.json: one row per sample (video) —
    its final cache key (which covers its whole lineage) plus the S3 VersionId of every
    file under input/ and output/. With bucket versioning on, any release can be
    re-materialized exactly even after processed/ is overwritten by later runs.
    """
    log = get_run_logger()
    out_key = f"{ctx.prefix('manifests')}{name}.json"
    if read_marker(ctx.s3, ctx.bucket, out_key) is not None:
        raise FileExistsError(f"release {name!r} already exists — releases are immutable")

    processed = ctx.prefix("processed")

    def sample(vid: str) -> dict:
        objects = {}
        for sub in ("input/", "output/"):
            pages = ctx.s3.get_paginator("list_objects_v2").paginate(
                Bucket=ctx.bucket, Prefix=f"{processed}{vid}/{sub}")
            for page in pages:
                for obj in page.get("Contents", []):
                    head = ctx.s3.head_object(Bucket=ctx.bucket, Key=obj["Key"])
                    objects[obj["Key"]] = head.get("VersionId")
        return {"video_id": vid, "cache_key": ctx.keys.key("post_processing", vid),
                "objects": objects}

    vids = sorted(ctx.ok.get("post_processing", ()))
    with ThreadPoolExecutor(S3_WORKERS) as pool:
        samples = list(pool.map(sample, vids))
    if any(v in (None, "null") for s in samples for v in s["objects"].values()):
        log.warning("bucket versioning appears disabled: release records keys but cannot pin "
                    "versions, so a later run may overwrite its files")

    manifest = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": ctx.git_sha,
        "pipeline_version": ctx.config["pipeline"]["version"],
        "hold_catalog_key": ctx.catalog_key,
        "samples": samples,
    }
    ctx.s3.put_object(Bucket=ctx.bucket, Key=out_key, Body=json.dumps(manifest, indent=2).encode(),
                      ContentType="application/json")
    log.info(f"release {name}: {len(samples)} samples -> s3://{ctx.bucket}/{out_key}")
    return out_key
