import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


SUSPICIOUS_PATTERNS = [
    ("prompt_injection", r"ignore\s+(all\s+)?(previous|prior)\s+instructions"),
    ("prompt_injection", r"(reveal|show|output|print).{0,30}(system prompt|hidden prompt|developer prompt)"),
    ("prompt_injection", r"(forget|bypass|override).{0,30}(rules|instructions|policy|guardrails)"),
    ("prompt_injection", r"you are now"),
    ("prompt_injection", r"act as .*?(without|ignore)"),
    ("data_exfiltration", r"(show|print|dump|reveal).{0,40}(api key|secret|token|credential)"),
    ("unsafe_capability", r"(malware|ransomware|virus|keylogger)"),
    ("unsafe_capability", r"how to (hack|exploit|bypass authentication|steal credentials)"),
]


def sanitize_query(text: str) -> str:
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def inspect_query(text: str) -> Dict[str, Any]:
    sanitized = sanitize_query(text or "")
    reasons: List[Dict[str, str]] = []

    if not sanitized:
        reasons.append({"category": "invalid_input", "rule": "empty_query"})
        return {
            "allowed": False,
            "blocked": True,
            "reason": "empty_query",
            "reasons": reasons,
            "sanitized_query": sanitized,
            "risk_level": "high",
        }

    for category, pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, sanitized, re.IGNORECASE | re.DOTALL):
            reasons.append({"category": category, "rule": pattern})

    blocked = len(reasons) > 0
    reason = reasons[0]["category"] if reasons else ""
    risk_level = "high" if blocked else "low"
    return {
        "allowed": not blocked,
        "blocked": blocked,
        "reason": reason,
        "reasons": reasons,
        "sanitized_query": sanitized,
        "risk_level": risk_level,
    }


def write_report(project_root: Path, run_id: str, report: Dict[str, Any]) -> Path:
    out_path = (
        project_root
        / "artifacts"
        / "experiments"
        / run_id
        / "guardrails"
        / "input_query_guard_report.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic input query guardrails.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to script parent.")
    parser.add_argument("--run-id", default="component3_input_query_guardrails")
    parser.add_argument("--query", default="What were Microsoft's FY2025 revenue and operating income?")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent

    result = inspect_query(args.query)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query": args.query,
        "result": result,
    }
    out_path = write_report(project_root, args.run_id, report)
    print(json.dumps({"ok": True, "report_path": str(out_path), "blocked": result["blocked"], "reason": result["reason"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
