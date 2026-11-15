"""
End-to-end test of the orchestration layer with S3 mocked by moto, Modal replaced by a
local fake that writes the marker exactly like the real job wrapper, and Claude batches
replaced by a fake client. No network, no Modal, no Claude.

    pip install -e ".[dev]"
    python tests/test_orchestration.py
"""
import json
import os
import sys
import types
from collections import Counter
from pathlib import Path

import tempfile

REPO = Path(__file__).resolve().parents[1]
SCRATCH = Path(tempfile.mkdtemp(prefix="bd-orch-test-"))
sys.path.insert(0, str(REPO))
os.environ.update(AWS_ACCESS_KEY_ID="x", AWS_SECRET_ACCESS_KEY="x", AWS_DEFAULT_REGION="us-east-1",
                  PREFECT_HOME=str(SCRATCH / "prefect_home"), GIT_SHA="testsha",
                  PREFECT_LOGGING_LEVEL="WARNING", ANTHROPIC_API_KEY="x")

import yaml
from moto import mock_aws

mock = mock_aws()
mock.start()

import boto3
from prefect.testing.utilities import prefect_test_harness

import agents.base as base
import agents.extract_hold_descriptions.agent as hold_agent
import flows.runtime as runtime
from common.idempotency import write_marker
from flows.dataset_flow import dataset_flow
from flows.filter_flow import filter_flow

config = yaml.safe_load((REPO / "configs/pipeline.yaml").read_text(encoding="utf-8"))
BUCKET, REGION = config["s3"]["bucket"], config["s3"]["region"]
s3 = boto3.client("s3", region_name=REGION)
s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION})
s3.put_bucket_versioning(Bucket=BUCKET, VersioningConfiguration={"Status": "Enabled"})
HEADER = ("post_id,item_id,video_key,stored_this_run,scraped_at,gym_id,gym_slug,label,color,grade,"
          "grade_source,climb_grade_raw,poster_ascent_grade,consensus_grade,consensus_counts,climb_name,"
          "climb_slug,climb_id,gym,username,user_private,height_cm,ape_index_cm,height_reach,"
          "poster_ascent_feel,poster_ascent_comment,ascent_date,date_created,caption,embed_url,video_url")


def meta_row(vid, height, ape=""):
    return f'1,2,videos/{vid}.mp4,True,t,1,gym,v3,null,v3,climb,v3,v3,,,,c,1,Gym,u,False,{height},{ape},"",Solid,,d,d,,e,f'


for vid in ("v1", "v2", "v3"):
    s3.put_object(Bucket=BUCKET, Key=f"raw/videos/{vid}.mp4", Body=f"video-{vid}".encode())
# Older scrape has v2 without height; newer scrape fills it in (latest wins). v3 never has one.
s3.put_object(Bucket=BUCKET, Key="raw/metadata/20261001T000000Z_gym.csv",
              Body="\n".join([HEADER, meta_row("v1", "175.0", "2.5"), meta_row("v2", ""), meta_row("v3", "")]).encode())
s3.put_object(Bucket=BUCKET, Key="raw/metadata/20261102T000000Z_gym.csv",
              Body="\n".join([HEADER, meta_row("v2", "180.3")]).encode())
# Scraper-owned state must never be touched
s3.put_object(Bucket=BUCKET, Key="state/kaya/gyms.csv", Body=b"x")
CFG = SCRATCH / "cfg.yaml"


def write_cfg():
    CFG.write_text(yaml.safe_dump(config))


write_cfg()

# ---------------------------------------------------------------- fake Modal
calls = Counter()          # (stage, video) -> times run
FAIL = set()               # (stage, video) to fail


class FakeFn:
    def __init__(self, stage):
        self.stage = stage

    def _run(self, vid, spec):
        calls[(self.stage, vid)] += 1
        if (self.stage, vid) in FAIL:
            raise RuntimeError(f"boom {self.stage} {vid}")
        if self.stage == "extract_filter_frames":
            s3.put_object(Bucket=BUCKET, Key=f"filtered_videos/{vid}/frames/0000.jpg", Body=b"jpg")
        if self.stage == "extract_bkg_plate":
            s3.put_object(Bucket=BUCKET, Key=f"processed/{vid}/bkg_rgb.png", Body=b"png")
        if self.stage == "extract_hold_proposals":
            marks = {"image_size": [100, 100], "marks": {
                "1": {"bbox": [0, 0, 10, 10], "area": 80, "centroid": [5, 5]},
                "2": {"bbox": [5, 5, 30, 20], "area": 200, "centroid": [15, 12]},
                "3": {"bbox": [50, 50, 60, 60], "area": 90, "centroid": [55, 55]}}}
            s3.put_object(Bucket=BUCKET, Key=f"processed/{vid}/holds/proposals/marks.json",
                          Body=json.dumps(marks).encode())
            s3.put_object(Bucket=BUCKET, Key=f"processed/{vid}/holds/proposals/tiles/00.png", Body=b"png")
        if self.stage == "post_processing":
            for sub in ("input/wall_mesh.glb", "output/scenegraph.json", "output/poses.npy"):
                s3.put_object(Bucket=BUCKET, Key=f"processed/{vid}/{sub}", Body=b"x")
        write_marker(spec)  # what the real job wrapper does last

    def map(self, vids, specs, kwargs=None, return_exceptions=False, wrap_returned_exceptions=None):
        out = []
        for vid, spec in zip(vids, specs):
            try:
                self._run(vid, spec)
                out.append(None)
            except Exception as e:
                if not return_exceptions:
                    raise
                out.append(e)
        return out

    def remote(self, members, spec, config):
        calls[(self.stage, "global")] += 1
        write_marker(spec)


runtime.job_fn = FakeFn

# ---------------------------------------------------------------- fake Claude
class FakeBatches:
    def __init__(self):
        self.batches, self.created, self.cancelled = {}, 0, []
        self.hang = False   # simulate orchestrator crash while polling

    def create(self, requests):
        self.created += 1
        bid = f"msgbatch_{self.created}"
        self.batches[bid] = requests
        return types.SimpleNamespace(id=bid)

    def retrieve(self, bid):
        if self.hang:
            raise RuntimeError("orchestrator died")
        return types.SimpleNamespace(processing_status="ended")

    def cancel(self, bid):
        self.cancelled.append(bid)

    def results(self, bid):
        for r in self.batches[bid]:
            schema = r["params"]["output_config"]["format"]["schema"]
            if "holds" in schema["properties"]:   # HoldLabelResponse
                hold = {"mark_ids": [2, 1], "hold_type": "jug", "outline": "oval",
                        "profile": "convex_dome", "lip": "incut", "opening": "hand_scoop",
                        "symmetry": "bilateral", "relative_size": "medium",
                        "depth_class": "deep", "description": "Big jug."}
                text = json.dumps({"holds": [hold], "non_hold_marks": [3]})
            else:                                  # FilterJudgment
                text = json.dumps({"too_many_climbers": False, "camera_blur": False,
                                   "moving_camera": False, "no_floor_plane": False, "notes": ""})
            msg = types.SimpleNamespace(stop_reason="end_turn",
                                        content=[types.SimpleNamespace(type="text", text=text)])
            yield types.SimpleNamespace(custom_id=r["custom_id"],
                                        result=types.SimpleNamespace(type="succeeded", message=msg))


fake = types.SimpleNamespace(messages=types.SimpleNamespace(batches=FakeBatches()))
base.get_client = lambda: fake


def ran(stage):
    return sorted(v for (s, v), n in calls.items() if s == stage)


def check(label, cond):
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        global failures
        failures += 1


failures = 0
# extract_depth and everything that (transitively) depends on it
DEPTH_DOWNSTREAM = {"extract_depth", "extract_poses", "extract_hold_contacts",
                    "extract_grounded_poses", "extract_wall_mesh", "extract_scenegraph",
                    "post_processing"}
with prefect_test_harness():
    # 1. filter
    passing = filter_flow(config_path=str(CFG))
    check("filter: v3 rejected for missing metadata, v1/v2 pass", passing == ["v1", "v2"])
    check("filter: one Claude batch for all videos", fake.messages.batches.created == 1
          and len(fake.messages.batches.batches["msgbatch_1"]) == 2)

    # 2. first dataset run, with depth failing for v2
    FAIL.add(("extract_depth", "v2"))
    dataset_flow(config_path=str(CFG))
    check("failure isolation: v2 stops after depth, v1 completes",
          ran("post_processing") == ["v1"] and ran("extract_grounded_poses") == ["v1"])
    check("independent stages still ran for v2", ran("extract_hold_crops") == ["v1", "v2"])
    check("catalog built once", calls[("curate_hold_catalog", "global")] == 1)

    # 3. rerun after fixing: only v2's depth-dependent stages run
    FAIL.clear(); calls.clear()
    dataset_flow(config_path=str(CFG))
    check("retry: only v2 runs, only depth-dependent stages",
          set(s for s, _ in calls) == DEPTH_DOWNSTREAM
          and all(v == "v2" for _, v in calls))

    # 4. full no-op rerun
    calls.clear(); created_before = fake.messages.batches.created
    dataset_flow(config_path=str(CFG))
    check("no-op rerun: nothing dispatched",
          not calls and fake.messages.batches.created == created_before)

    # 5. config change propagates downstream only
    config["downsampling"]["depth_rate"] = 3; write_cfg(); calls.clear()
    dataset_flow(config_path=str(CFG))
    stages = set(s for s, _ in calls)
    check("depth_rate change reruns depth + descendants only",
          stages == DEPTH_DOWNSTREAM)
    check("existing catalog reused (snapshot semantics)", ("curate_hold_catalog", "global") not in calls)

    # 6. catalog rebuild invalidates only scenegraph + post_processing
    calls.clear()
    dataset_flow(config_path=str(CFG), rebuild_catalog=True)
    # The depth_rate change above flowed into the catalog through extract_wall_mesh
    # (catalog fits use its hold frames), so this rebuild is real...
    check("rebuild picks up upstream (wall mesh) changes",
          set(s for s, _ in calls) == {"curate_hold_catalog", "extract_scenegraph", "post_processing"})
    calls.clear()
    dataset_flow(config_path=str(CFG), rebuild_catalog=True)
    # ...and a second rebuild with nothing changed is a no-op.
    check("catalog rebuild with same members/inputs is a no-op",
          not calls)
    config["hold_catalog"]["pca_components"] = 32; write_cfg(); calls.clear()
    dataset_flow(config_path=str(CFG), rebuild_catalog=True)
    check("catalog config change + rebuild reruns catalog, scenegraph, post_processing",
          set(s for s, _ in calls) == {"curate_hold_catalog", "extract_scenegraph", "post_processing"})

    # 7. code change (comments-only vs semantic)
    from common import stages as st
    p = REPO / "jobs/extract_wall_mesh/logic.py"
    orig = p.read_text()
    try:
        p.write_text(orig + "\n# just a comment\n"); st.code_hash.cache_clear(); calls.clear()
        dataset_flow(config_path=str(CFG))
        check("comment-only edit does not invalidate", not calls)
        p.write_text(orig + "\nX = 1\n"); st.code_hash.cache_clear(); calls.clear()
        dataset_flow(config_path=str(CFG))
        check("semantic edit reruns wall_mesh + descendants",
              set(s for s, _ in calls) == {"extract_wall_mesh", "extract_scenegraph", "post_processing"})
    finally:
        p.write_text(orig); st.code_hash.cache_clear()

    # 7b. per-stage agent config: changing the filter model touches nothing in the dataset
    dataset_flow(config_path=str(CFG))   # settle the rerun from restoring the code in 7
    config["agents"]["filter_video"]["model"] = "claude-sonnet-5-5"; write_cfg(); calls.clear()
    dataset_flow(config_path=str(CFG))
    check("filter model change leaves dataset stages cached", not calls)

    # 8. batch resume: crash while polling, rerun re-attaches without resubmitting
    config["agents"]["extract_hold_descriptions"]["max_tokens"] = 8000; write_cfg()   # invalidates hold descriptions
    fb = fake.messages.batches
    fb.hang = True
    created_before = fb.created
    try:
        dataset_flow(config_path=str(CFG))
    except BaseException as e:
        pass
    pending = s3.list_objects_v2(Bucket=BUCKET, Prefix="_batches/").get("KeyCount", 0)
    check("crash mid-poll leaves batch record in S3", pending == 1 and fb.created == created_before + 1)
    fb.hang = False; calls.clear()
    dataset_flow(config_path=str(CFG))
    check("rerun re-attaches: no new batch submitted", fb.created == created_before + 1)
    check("batch record cleaned up", s3.list_objects_v2(Bucket=BUCKET, Prefix="_batches/").get("KeyCount", 0) == 0)
    check("downstream of hold descriptions reran", "extract_hold_crops" in set(s for s, _ in calls))

    # 9. stale in-flight batch is cancelled, not collected
    config["agents"]["extract_hold_descriptions"]["max_tokens"] = 9000; write_cfg()
    fb.hang = True
    try:
        dataset_flow(config_path=str(CFG))
    except BaseException:
        pass
    stale_id = f"msgbatch_{fb.created}"
    config["agents"]["extract_hold_descriptions"]["max_tokens"] = 9500; write_cfg(); fb.hang = False
    dataset_flow(config_path=str(CFG))
    check("stale in-flight batch cancelled", stale_id in fb.cancelled)

    # 10. release
    dataset_flow(config_path=str(CFG), release="r1")
    m = json.loads(s3.get_object(Bucket=BUCKET, Key="manifests/r1.json")["Body"].read())
    check("release: one row per sample with version ids",
          [s["video_id"] for s in m["samples"]] == ["v1", "v2"]
          and all(v not in (None, "null") for s in m["samples"] for v in s["objects"].values()))
    try:
        dataset_flow(config_path=str(CFG), release="r1")
        check("release names are immutable", False)
    except Exception as e:
        check("release names are immutable", "already exists" in str(e))

    import csv as _csv, io as _io
    rows = list(_csv.DictReader(_io.StringIO(s3.get_object(
        Bucket=BUCKET, Key="processed/v1/holds/hold_descriptions.csv")["Body"].read().decode())))
    check("hold records: code-assigned id, merged marks, union bbox",
          len(rows) == 1 and rows[0]["hold_id"] == "h000" and json.loads(rows[0]["mark_ids"]) == [1, 2]
          and json.loads(rows[0]["bbox"]) == [0, 0, 30, 20] and rows[0]["on_route"] == "")
    req = fake.messages.batches.batches["msgbatch_1"][0]["params"]
    check("filter request carries structured-output schema + effort",
          req["output_config"]["format"]["type"] == "json_schema"
          and req["output_config"]["format"]["schema"].get("additionalProperties") is False
          and req["output_config"]["effort"] == "low" and req["model"] == "claude-haiku-5-5")

    climber = json.loads(s3.get_object(Bucket=BUCKET, Key="processed/v2/climber.json")["Body"].read())
    check("climber.json from latest scrape", climber == {"height_cm": 180.3, "ape_index_cm": None})
    calls.clear()
    s3.put_object(Bucket=BUCKET, Key="raw/metadata/20261201T000000Z_gym.csv",
                  Body="\n".join([HEADER, meta_row("v1", "176.0", "2.5")]).encode())
    dataset_flow(config_path=str(CFG))
    check("height correction reruns post_processing for that video only",
          set(calls) == {("post_processing", "v1")})
    check("raw/ and state/ untouched",
          sorted(o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET, Prefix="state/")["Contents"]) == ["state/kaya/gyms.csv"]
          and all(o["Key"].startswith(("raw/videos/", "raw/metadata/"))
                  for o in s3.list_objects_v2(Bucket=BUCKET, Prefix="raw/")["Contents"]))

print(f"\n{failures} failure(s)")
sys.exit(1 if failures else 0)
