"""Build a tiny LoCoMo subset for pipeline smoke tests (not for reported results).

Takes conversation 0, keeps its first N sessions, and a small stratified QA subset.
Usage: python make_smoke_dataset.py
"""
import json

SRC = "data/locomo10.json"
DST = "data/locomo_smoke.json"
N_SESSIONS = 3
QA_PER_CATEGORY = 4

with open(SRC) as f:
    data = json.load(f)

sample = data[0]
conv = sample["conversation"]
keep = {}
for key, value in conv.items():
    if key.startswith("session_"):
        rest = key[len("session_"):]
        if rest.endswith("_date_time"):
            num = int(rest[: -len("_date_time")])
        else:
            num = int(rest)
        if num <= N_SESSIONS:
            keep[key] = value
    else:
        keep[key] = value

by_cat = {}
for qa in sample["qa"]:
    by_cat.setdefault(qa.get("category"), []).append(qa)
qa_subset = []
for cat, items in sorted(by_cat.items(), key=lambda kv: str(kv[0])):
    qa_subset.extend(items[:QA_PER_CATEGORY])

out = [{
    "sample_id": sample.get("sample_id", "conv0_smoke"),
    "conversation": keep,
    "qa": qa_subset,
    "event_summary": sample.get("event_summary", {}),
    "observation": sample.get("observation", {}),
    "session_summary": sample.get("session_summary", {}),
}]

with open(DST, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

n_turns = sum(len(v) for k, v in keep.items() if k.startswith("session_") and not k.endswith("_date_time"))
print(f"wrote {DST}: {N_SESSIONS} sessions, {n_turns} turns, {len(qa_subset)} QA pairs")
