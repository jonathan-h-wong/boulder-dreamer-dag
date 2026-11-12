"""Schema-level checks for the structured agent outputs. Usage: python tests/test_schemas.py"""
import json
import sys
import types

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anthropic
from pydantic import ValidationError

from agents.base import output_config, parse_structured
from agents.extract_hold_descriptions.agent import build_records
from common.schemas import FilterJudgment, HoldLabelResponse

fails = 0


def check(label, cond):
    global fails
    print(("PASS " if cond else "FAIL ") + label)
    fails += not cond


def msg(obj, stop="end_turn"):
    text = obj if isinstance(obj, str) else json.dumps(obj)
    return types.SimpleNamespace(stop_reason=stop, content=[types.SimpleNamespace(type="text", text=text)])


# 1. Generated schemas are strict and enum-constrained
for model in (HoldLabelResponse, FilterJudgment):
    sch = output_config(model)["format"]["schema"]
    blob = json.dumps(sch)
    check(f"{model.__name__}: additionalProperties false", sch.get("additionalProperties") is False)
    check(f"{model.__name__}: all properties required",
          set(sch["required"]) == set(sch["properties"]))
hold_schema = json.dumps(output_config(HoldLabelResponse)["format"]["schema"])
check("hold schema enumerates hold_type values", "volume_pyramid" in hold_schema and "foot_chip" in hold_schema)
print(json.dumps(output_config(FilterJudgment)["format"]["schema"], indent=1)[:600])

HOLD = {"mark_ids": [1], "hold_type": "crimp_edge", "outline": "elongated", "profile": "flat_top",
        "lip": "sharp", "opening": "none", "symmetry": "bilateral", "relative_size": "small",
        "depth_class": "shallow", "description": "Thin edge."}
MARKS = {1: {"bbox": [0, 0, 5, 5]}, 2: {"bbox": [10, 10, 20, 20]}, 3: {"bbox": [1, 1, 2, 2]}}

# 2. Pydantic re-validation catches what the API-side schema can't express
try:
    parse_structured(msg({"holds": [{**HOLD, "description": "x" * 400}], "non_hold_marks": []}), HoldLabelResponse)
    check("description max_length enforced on re-validation", False)
except ValidationError:
    check("description max_length enforced on re-validation", True)
try:
    parse_structured(msg({"holds": [{**HOLD, "hold_type": "bucket"}], "non_hold_marks": []}), HoldLabelResponse)
    check("unknown enum value rejected", False)
except ValidationError:
    check("unknown enum value rejected", True)

# 3. Truncated / refused responses fail cleanly (retried next run)
for stop in ("max_tokens", "refusal"):
    try:
        parse_structured(msg('{"holds": [', stop=stop), HoldLabelResponse)
        check(f"stop_reason={stop} raises", False)
    except ValueError as e:
        check(f"stop_reason={stop} raises", "stop_reason" in str(e))

# 4. Mark coverage
ok = parse_structured(msg({"holds": [{**HOLD, "mark_ids": [3, 1]}], "non_hold_marks": [2]}), HoldLabelResponse)
recs = build_records(ok, MARKS)
check("records: id, sorted marks, union bbox",
      recs[0].hold_id == "h000" and recs[0].mark_ids == [1, 3] and recs[0].bbox == [0, 0, 5, 5])
for label, payload in [
    ("missing mark", {"holds": [HOLD], "non_hold_marks": []}),
    ("duplicate mark", {"holds": [HOLD], "non_hold_marks": [1, 2, 3]}),
    ("unknown mark", {"holds": [HOLD], "non_hold_marks": [2, 3, 9]}),
]:
    try:
        build_records(parse_structured(msg(payload), HoldLabelResponse), MARKS)
        check(f"coverage rejects {label}", False)
    except ValueError:
        check(f"coverage rejects {label}", True)

print(f"\n{fails} failure(s)")
sys.exit(1 if fails else 0)
