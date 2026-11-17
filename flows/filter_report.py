"""
Filter statistics: one table row per raw video, and a summary computed from the table.

Stored by the filter flow as
  filtered_videos/filter_results.csv       current status of every video ever filtered
                                           (a subset run updates only its own rows)
  filtered_videos/_runs/{utc_ts}.json      append-only history: summary of the whole table
                                           after each run, plus run scope, git sha, model
and published to Prefect as a markdown artifact.
"""

from __future__ import annotations

import csv
import io
from collections import Counter

CRITERIA = ("missing_climber_metadata", "too_many_climbers", "camera_blur",
            "moving_camera", "no_floor_plane", "camera_motion")
# moving_camera is Claude's judgment; camera_motion is the measured check
# (camera_motion_px > filter.max_camera_motion_px). Either one rejects.
COLUMNS = ("video_id", "gym", "status", *CRITERIA, "no_metadata_row", "not_evaluated_reason",
           "notes", "camera_motion_px", "duration_s", "width", "height")


def _b(v) -> bool:
    return v is True or str(v).lower() == "true"


def rows_to_csv(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="ignore")
    w.writeheader()
    for r in sorted(rows, key=lambda r: r["video_id"]):
        w.writerow({k: ("" if r.get(k) is None else str(r[k]).lower() if isinstance(r.get(k), bool)
                        else r[k]) for k in COLUMNS})
    return buf.getvalue().encode("utf-8")


def rows_from_csv(data: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def summarize(rows: list[dict]) -> dict:
    status = Counter(r["status"] for r in rows)
    evaluated = [r for r in rows if r["status"] != "not_evaluated"]
    rejected = [r for r in rows if r["status"] == "rejected"]
    any_reason = {c: sum(_b(r[c]) for r in rejected) for c in CRITERIA}
    sole_reason = Counter()
    combos = Counter()
    for r in rejected:
        failed = [c for c in CRITERIA if _b(r[c])]
        combos[" + ".join(failed)] += 1
        if len(failed) == 1:
            sole_reason[failed[0]] += 1
    no_meta = [r for r in rows if _b(r["missing_climber_metadata"])]
    gyms: dict[str, Counter] = {}
    for r in rows:
        gyms.setdefault(r.get("gym") or "(unknown)", Counter())[r["status"]] += 1
    by_gym = sorted(
        ({"gym": g, "total": sum(c.values()), "passed": c["passed"], "rejected": c["rejected"],
          "not_evaluated": c["not_evaluated"]} for g, c in gyms.items()),
        key=lambda d: (-d["total"], d["gym"]))
    n = len(rows)
    return {
        "total": n,
        "passed": status["passed"],
        "rejected": status["rejected"],
        "not_evaluated": status["not_evaluated"],
        "pass_rate": round(status["passed"] / n, 4) if n else None,
        "pass_rate_of_evaluated": round(status["passed"] / len(evaluated), 4) if evaluated else None,
        "rejections_by_criterion": any_reason,          # a video can count under several
        "rejections_sole_criterion": dict(sole_reason),
        "rejection_combinations": dict(combos.most_common()),
        "missing_metadata": {
            "no_metadata_row": sum(_b(r["no_metadata_row"]) for r in no_meta),
            "row_without_height": sum(not _b(r["no_metadata_row"]) for r in no_meta),
        },
        "not_evaluated_reasons": dict(Counter(
            (r.get("not_evaluated_reason") or "")[:120] for r in rows
            if r["status"] == "not_evaluated").most_common(20)),
        "by_gym": by_gym,
    }


def to_markdown(s: dict, run: dict, top_gyms: int = 25) -> str:
    def pct(k, d=None):
        d = d or s["total"]
        return f"{k} ({k / d:.1%})" if d else str(k)

    lines = [
        f"# Video filter: {s['passed']} / {s['total']} passed",
        f"Run {run['run_ts']}: {run['scope']} videos in scope, git {run['git_sha']}, "
        f"model {run['model']}.",
        "",
        "| status | videos |", "|---|---|",
        f"| passed | {pct(s['passed'])} |",
        f"| rejected | {pct(s['rejected'])} |",
        f"| not evaluated | {pct(s['not_evaluated'])} |",
        "",
        "## Rejections by criterion",
        "A rejected video counts under every criterion it failed; 'sole' counts videos that "
        "failed only that one.",
        "", "| criterion | any | sole |", "|---|---|---|",
        *(f"| {c} | {s['rejections_by_criterion'][c]} | {s['rejections_sole_criterion'].get(c, 0)} |"
          for c in CRITERIA),
        "",
        f"Missing metadata: {s['missing_metadata']['no_metadata_row']} with no metadata row, "
        f"{s['missing_metadata']['row_without_height']} with a row but no height.",
    ]
    if s["not_evaluated_reasons"]:
        lines += ["", "## Not evaluated", "", "| reason | videos |", "|---|---|",
                  *(f"| {r or '(none)'} | {n} |" for r, n in s["not_evaluated_reasons"].items())]
    lines += ["", f"## Top {top_gyms} gyms by video count", "",
              "| gym | total | passed | pass rate |", "|---|---|---|---|",
              *(f"| {g['gym']} | {g['total']} | {g['passed']} | {g['passed'] / g['total']:.0%} |"
                for g in s["by_gym"][:top_gyms])]
    return "\n".join(lines)
