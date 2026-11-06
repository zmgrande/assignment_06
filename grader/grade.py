"""Autograder for the graderthan sandbox.

Runs pytest once per rubric `auto` case (one per test file), scores each by pass
ratio, and writes the per-case results the harness reads. Case ids must match the
`auto` case ids in rubric.json. Requires pytest + pytest-json-report in the image.

`test_output` includes the pass count and, for any failing test, its name and a
concise error message — so the student (and the AI judge) see *which* tests failed
and *why*, not just a count.
"""
import json
import os
import subprocess

OUT = "/work/.grader"
os.makedirs(OUT, exist_ok=True)

# rubric.json `auto` case id  ->  test file to run for it
CASES = ["test_unit.py", "test_pipeline.py", "test_code.py"]

_MAX_FAILS = 5      # cap failing-test lines per case (keep test_output bounded)
_MAX_MSG = 200      # cap each failure message


def failures(report_path):
    """Concise `FAILED <test>: <error>` lines for non-passing tests."""
    out = []
    try:
        data = json.load(open(report_path))
    except Exception:  # noqa: BLE001 — no report / unreadable → no detail
        return out
    for t in data.get("tests", []):
        if t.get("outcome") == "passed":
            continue
        name = t.get("nodeid", "?").split("::")[-1]
        call = t.get("call") or {}
        crash = call.get("crash") or {}
        msg = crash.get("message") or call.get("longrepr") or t.get("outcome", "failed")
        msg = " ".join(str(msg).split())[:_MAX_MSG]     # one bounded line
        out.append(f"FAILED {name}: {msg}")
        if len(out) >= _MAX_FAILS:
            break
    return out


results = []
for name in CASES:
    report = os.path.join(OUT, name + ".report.json")
    subprocess.run(
        ["python3", "-m", "pytest", f"tests/{name}", "-q",
         "-p", "no:cacheprovider", "--json-report", f"--json-report-file={report}"],
        cwd="/work", capture_output=True, text=True,
    )
    total = passed = 0
    try:
        summary = json.load(open(report)).get("summary", {})
        total, passed = summary.get("total", 0), summary.get("passed", 0)
    except Exception:  # noqa: BLE001 — a collection error just scores the case 0
        pass
    ratio = (passed / total) if total else 0.0
    lines = [f"{passed}/{total} tests passed"] + failures(report)
    results.append({"id": name, "value": round(ratio, 4),
                    "test_output": "\n".join(lines)})

with open(os.path.join(OUT, "results.json"), "w") as fh:
    json.dump({"cases": results}, fh)
print("grade.py done")
