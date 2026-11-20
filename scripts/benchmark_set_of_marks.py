"""
Set-of-marks model benchmark: which Claude model labels hold proposals well enough?

Runs the exact two-pass hold perception (detect_holds' close-ups, then
extract_hold_descriptions on the detected holds; same images, prompts, schemas and
validation) for each video with several model/effort variants (one variant drives both
passes), synchronously (so latency is measured; the pipeline itself uses the Batches API
at half the price). One variant is the reference and every other one is scored against
it; or score everything against a saved earlier run (--reference-run, e.g. the Opus-high
answers of a previous benchmark) without paying for the reference again. Spot-check the
reference with the renders.

Needs extract_hold_proposals outputs. By default it uses the same seeded sample as
    modal run --detach scripts/run_flow_on_modal.py --flow dataset --sample 20 \
        --stages extract_bkg_plate,extract_hold_proposals
and skips videos without proposals. Run:

    modal run --detach scripts/benchmark_set_of_marks.py
    modal run --detach scripts/benchmark_set_of_marks.py --variants \
        claude-opus-5-5:high,claude-haiku-5-5:medium --sample 10
    modal run --detach scripts/benchmark_set_of_marks.py --variants claude-haiku-5-5:medium \
        --reference-run 20261117T031500Z --reference claude-opus-5-5:high

Writes _benchmarks/set_of_marks/{utc_ts}/
    summary.json, summary.md             per-variant scores, cost, latency
    responses/{video}/{variant}.json     labels (detections joined with descriptions), per-pass
                                         usage, stop reasons, errors
    renders/{video}/{variant}.jpg        holds filled (one color per hold, mark numbers),
                                         non-hold marks outlined red

Scores against the reference, per video then averaged:
    hold P/R/F1     per mark: is it part of a hold or volume?
    group F1        per pair of marks both call holds: are they the same hold?
    attribute acc.  per mark both call holds: does its hold get the same hold_type, ...?
    coverage        marks missing/duplicated/unknown in the answer (scored as non-holds)
"""

from __future__ import annotations

import modal

from common.modal_images import SECRETS, orchestrator

app = modal.App("boulderdreamer-benchmark-som", image=orchestrator)

CONFIG = "configs/pipeline.yaml"
DEFAULT_VARIANTS = ("claude-opus-5-5:high,claude-opus-5-5:medium,"
                    "claude-sonnet-5-5:medium,claude-haiku-5-5:medium")
PRICES = {   # $ per MTok (input, output), synchronous; batches are half
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),   # prompts <= 100K tokens
}
ATTRS = ("hold_type", "outline", "profile", "lip", "opening", "symmetry", "relative_size",
         "depth_class")
WORKERS = 8


# ---------------------------------------------------------------------------
# Scoring (pure)
# ---------------------------------------------------------------------------

def mark_view(labels: dict, mark_ids: set[int]) -> dict[int, dict | None]:
    """mark -> its hold's fields plus "_hold" (index), or None if it isn't a hold."""
    view: dict[int, dict | None] = {m: None for m in mark_ids}
    for i, h in enumerate(labels.get("holds", [])):
        for m in h["mark_ids"]:
            if m in view and view[m] is None:
                view[m] = {**{a: h.get(a) for a in ATTRS}, "_hold": i}
    return view


def coverage(labels: dict, mark_ids: set[int]) -> dict:
    seen = [m for h in labels.get("holds", []) for m in h["mark_ids"]] + list(labels.get("non_hold_marks", []))
    return {"missing": len(mark_ids - set(seen)), "unknown": len(set(seen) - mark_ids),
            "duplicated": len({m for m in seen if seen.count(m) > 1})}


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None:
        return None
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def score(pred: dict, ref: dict, mark_ids: set[int]) -> dict:
    pv, rv = mark_view(pred, mark_ids), mark_view(ref, mark_ids)
    ph = {m for m, v in pv.items() if v}
    rh = {m for m, v in rv.items() if v}
    tp = len(ph & rh)
    prec = tp / len(ph) if ph else None
    rec = tp / len(rh) if rh else None
    both = sorted(ph & rh)
    pairs_p = {(a, b) for i, a in enumerate(both) for b in both[i + 1:]
               if pv[a]["_hold"] == pv[b]["_hold"]}
    pairs_r = {(a, b) for i, a in enumerate(both) for b in both[i + 1:]
               if rv[a]["_hold"] == rv[b]["_hold"]}
    if pairs_p or pairs_r:   # no multi-mark holds on either side: nothing to score (None)
        inter = len(pairs_p & pairs_r)
        gp = inter / len(pairs_p) if pairs_p else 0.0
        gr = inter / len(pairs_r) if pairs_r else 0.0
    else:
        gp = gr = None
    attrs = {a: (sum(pv[m][a] == rv[m][a] for m in both) / len(both) if both else None)
             for a in ATTRS}
    return {"hold_precision": prec, "hold_recall": rec, "hold_f1": _f1(prec, rec),
            "mark_agreement": sum((m in ph) == (m in rh) for m in mark_ids) / len(mark_ids),
            "group_f1": _f1(gp, gr), "attr_accuracy": attrs,
            "n_holds": len(pred.get("holds", [])), "n_hold_marks": len(ph)}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


# ---------------------------------------------------------------------------
# Rendering (spot checks)
# ---------------------------------------------------------------------------

def render(plate_jpg: bytes, marks: dict, masks: dict, labels: dict | None):
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(plate_jpg, np.uint8), cv2.IMREAD_COLOR)
    pw, _ = marks["image_size"]
    s = img.shape[1] / pw
    view = mark_view(labels or {}, set(masks))
    fill = img.copy()
    rng = np.random.default_rng(0)
    colors = [tuple(int(c) for c in rng.integers(60, 256, 3)) for _ in range(1000)]
    contours = {}
    for m, crop in masks.items():
        x0, y0 = marks["marks"][str(m)]["bbox"][:2]
        cs, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours[m] = [((c.astype(np.float32) + (x0, y0)) * s).astype(np.int32) for c in cs]
        if view[m]:
            cv2.drawContours(fill, contours[m], -1, colors[view[m]["_hold"] % 1000], -1)
    out = cv2.addWeighted(fill, 0.45, img, 0.55, 0)
    for m, cs in contours.items():
        hold = view[m]
        cv2.drawContours(out, cs, -1, colors[hold["_hold"] % 1000] if hold else (0, 0, 255),
                         2 if hold else 1)
    fs = max(0.35, 0.4 * max(img.shape[:2]) / 1280)
    lp = img.shape[1] / marks["overlay_size"][0]          # label points are in overlay px
    for m in masks:
        x, y = marks["marks"][str(m)]["label_point"]
        cv2.putText(out, str(m), (int(x * lp), int(y * lp)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255),
                    1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------

@app.function(secrets=SECRETS, timeout=4 * 3600, cpu=4, memory=8192)
def run_benchmark(variants: list[str], reference: str, video_ids: list[str] | None,
                  sample: int, seed: int, max_tokens: int, git_sha: str,
                  reference_run: str = "") -> dict:
    import copy
    import json
    import random
    import time
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timezone

    import anthropic

    from agents.base import message_text
    from agents.detect_holds import agent as detect
    from agents.extract_hold_descriptions import agent as describe
    from common.proposals import load_proposals
    from common.s3 import download_bytes, get_client
    from common.schemas import HoldDescriptionResponse, HoldDetectionResponse
    from flows.runtime import load_config

    config = load_config(CONFIG)
    s3, bucket = get_client(config["s3"]["region"]), config["s3"]["bucket"]
    bench = f"{config['s3']['prefixes']['benchmarks']}set_of_marks/"
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = f"{bench}{run_ts}/"
    client = anthropic.Anthropic()

    if not video_ids:
        ids = json.loads(download_bytes(s3, bucket, config["s3"]["prefixes"]["filtered_videos"]
                                        + "passing_video_ids.json"))
        video_ids = random.Random(seed).sample(sorted(ids), min(sample, len(ids)))

    def load(vid):
        try:
            marks, masks, plate = load_proposals(vid, config)
            return vid, {"marks": marks, "masks": masks, "plate": plate}
        except Exception as e:   # no proposals for this video yet
            print(f"skip {vid}: {e!r}")
            return vid, None

    with ThreadPoolExecutor(WORKERS) as pool:
        videos = {v: d for v, d in pool.map(load, video_ids) if d}
    if not videos:
        raise RuntimeError("no sampled video has hold proposals; run the proposals stage first")
    to_run = [v for v in variants if not (reference_run and v == reference)]
    print(f"{len(videos)} videos x {len(to_run)} variants, reference {reference}"
          + (f" from run {reference_run}" if reference_run else ""))

    def stream(params):
        with client.messages.stream(**params) as st:
            return st.get_final_message()

    def call(vid, variant):
        """Both passes with this variant -> record with labels (HoldLabelResponse JSON)."""
        model, effort = variant.split(":")
        cfg = copy.deepcopy(config)
        for stage in ("detect_holds", "extract_hold_descriptions"):
            cfg["agents"][stage].update(model=model, effort=effort, max_tokens=max_tokens)
        d = videos[vid]
        rec = {"video_id": vid, "variant": variant, "passes": {}, "labels": None}
        pin, pout = PRICES.get(model, (0.0, 0.0))
        t0 = time.time()
        try:
            msg = stream(detect.build_request(vid, d["marks"], d["masks"], d["plate"], cfg))
            rec["passes"]["detect"] = {"stop_reason": msg.stop_reason, "input_tokens": msg.usage.input_tokens,
                                       "output_tokens": msg.usage.output_tokens}
            if msg.stop_reason != "end_turn":
                raise ValueError(f"detect stop_reason={msg.stop_reason}")
            dets = detect.resolve(HoldDetectionResponse.model_validate_json(message_text(msg)),
                                  {int(k) for k in d["marks"]["marks"]},
                                  cfg["agents"]["detect_holds"]["max_missing_frac"])
            rec["unaccounted"] = dets.unaccounted
            msg = stream(describe.build_request(vid, d["marks"], d["masks"], d["plate"], dets, cfg))
            rec["passes"]["describe"] = {"stop_reason": msg.stop_reason, "input_tokens": msg.usage.input_tokens,
                                         "output_tokens": msg.usage.output_tokens}
            if msg.stop_reason != "end_turn":
                raise ValueError(f"describe stop_reason={msg.stop_reason}")
            labels = describe.join(dets, HoldDescriptionResponse.model_validate_json(message_text(msg)))
            rec["labels"] = labels.model_dump(mode="json")
        except Exception as e:
            rec["error"] = repr(e)[:500]
        rec["latency_s"] = round(time.time() - t0, 1)
        p = rec["passes"].values()
        rec["input_tokens"] = sum(x["input_tokens"] for x in p)
        rec["output_tokens"] = sum(x["output_tokens"] for x in p)
        rec["cost_usd"] = round((rec["input_tokens"] * pin + rec["output_tokens"] * pout) / 1e6, 4)
        s3.put_object(Bucket=bucket, Key=f"{out}responses/{vid}/{variant.replace(':', '_')}.json",
                      Body=json.dumps(rec, indent=2).encode(), ContentType="application/json")
        print(f"{vid} {variant}: {rec.get('error') or 'ok'} {rec['latency_s']}s")
        return rec

    jobs = [(v, var) for v in videos for var in to_run]
    with ThreadPoolExecutor(WORKERS) as pool:
        results = {(r["video_id"], r["variant"]): r for r in pool.map(lambda a: call(*a), jobs)}
    if reference_run:   # the saved reference answers (from either benchmark version)
        for vid in videos:
            key = f"{bench}{reference_run}/responses/{vid}/{reference.replace(':', '_')}.json"
            try:
                rec = json.loads(download_bytes(s3, bucket, key))
            except Exception:
                rec = {"labels": None, "error": "no saved reference answer"}
            results[(vid, reference)] = {**rec, "video_id": vid, "variant": reference}

    def render_one(key):
        vid, variant = key
        d, labels = videos[vid], results[key].get("labels")
        img = render(d["plate"], d["marks"], d["masks"], labels)
        s3.put_object(Bucket=bucket, Key=f"{out}renders/{vid}/{variant.replace(':', '_')}.jpg",
                      Body=img, ContentType="image/jpeg")
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(render_one, results))

    per_video, summary = {}, {}
    for variant in variants:
        rows = []
        for vid, d in videos.items():
            r, ref = results[(vid, variant)], results[(vid, reference)]
            mark_ids = {int(k) for k in d["marks"]["marks"]}
            row = {"video_id": vid, "n_marks": len(mark_ids), "ok": r.get("labels") is not None,
                   "error": r.get("error"), "latency_s": r.get("latency_s"),
                   "output_tokens": r.get("output_tokens"), "cost_usd": r.get("cost_usd")}
            if r.get("labels") is not None:
                row["coverage"] = coverage(r["labels"], mark_ids)
                if variant != reference and ref.get("labels") is not None:
                    row.update(score(r["labels"], ref["labels"], mark_ids))
                else:
                    row["n_holds"] = len(r["labels"].get("holds", []))
            rows.append(row)
        per_video[variant] = rows
        ok = [x for x in rows if x["ok"]]
        lat = sorted(x["latency_s"] for x in ok)
        summary[variant] = {
            "videos": len(rows), "ok": len(ok),
            "errors": sorted({x["error"][:80] for x in rows if x["error"]}),
            "complete_coverage": sum(1 for x in ok if not any(x["coverage"].values())),
            "hold_precision": mean(x.get("hold_precision") for x in ok),
            "hold_recall": mean(x.get("hold_recall") for x in ok),
            "hold_f1": mean(x.get("hold_f1") for x in ok),
            "mark_agreement": mean(x.get("mark_agreement") for x in ok),
            "group_f1": mean(x.get("group_f1") for x in ok),
            "attr_accuracy": {a: mean(x["attr_accuracy"][a] for x in ok if "attr_accuracy" in x)
                              for a in ATTRS},
            "mean_holds": mean(x.get("n_holds") for x in ok),
            "median_latency_s": lat[len(lat) // 2] if lat else None,
            "mean_output_tokens": mean(x.get("output_tokens") for x in ok),
            "cost_usd_total": round(sum(x["cost_usd"] or 0 for x in rows), 2),
            "cost_per_video_batch_usd": round(sum(x["cost_usd"] or 0 for x in rows) / len(rows) / 2, 4),
        }

    def f(x):
        return "—" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)
    md = [f"# Set-of-marks benchmark {run_ts}",
          f"{len(videos)} videos, reference **{reference}**"
          + (f" from run {reference_run}" if reference_run else "") + " (scores are agreement with it), "
          f"git {git_sha}, max_tokens {max_tokens}.", "",
          "| variant | ok | full coverage | hold P | hold R | hold F1 | group F1 | type acc | "
          "mean holds | median latency s | $/video (batch) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, s_ in summary.items():
        md.append(f"| {v} | {s_['ok']}/{s_['videos']} | {s_['complete_coverage']} | "
                  f"{f(s_['hold_precision'])} | {f(s_['hold_recall'])} | {f(s_['hold_f1'])} | "
                  f"{f(s_['group_f1'])} | {f(s_['attr_accuracy']['hold_type'])} | "
                  f"{f(s_['mean_holds'])} | {f(s_['median_latency_s'])} | "
                  f"{s_['cost_per_video_batch_usd']} |")
    md += ["", "Attribute accuracy (marks both call holds):", "",
           "| variant | " + " | ".join(ATTRS) + " |", "|---|" + "---|" * len(ATTRS)]
    for v, s_ in summary.items():
        if v != reference:
            md.append(f"| {v} | " + " | ".join(f(s_["attr_accuracy"][a]) for a in ATTRS) + " |")
    md_text = "\n".join(md)
    result = {"run_ts": run_ts, "prefix": out, "reference": reference, "reference_run": reference_run,
              "git_sha": git_sha,
              "videos": sorted(videos), "summary": summary, "per_video": per_video}
    s3.put_object(Bucket=bucket, Key=out + "summary.json", Body=json.dumps(result, indent=2).encode(),
                  ContentType="application/json")
    s3.put_object(Bucket=bucket, Key=out + "summary.md", Body=md_text.encode(), ContentType="text/markdown")
    result["markdown"] = md_text
    return result


@app.local_entrypoint()
def main(variants: str = DEFAULT_VARIANTS, reference: str = "", video_ids: str = "",
         sample: int = 20, seed: int = 0, max_tokens: int = 32000, reference_run: str = ""):
    from flows.runtime import git_sha

    vs = [v.strip() for v in variants.split(",") if v.strip()]
    ref = reference or vs[0]
    if ref not in vs:
        vs.insert(0, ref)
    ids = [v.strip() for v in video_ids.split(",") if v.strip()] or None
    result = run_benchmark.remote(vs, ref, ids, sample, seed, max_tokens, git_sha(), reference_run)
    print(result["markdown"])
    print(f"\nresults: s3://boulder-dreamer-dataset/{result['prefix']}")
