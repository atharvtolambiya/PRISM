"""
Triggered by --interactive flag in the CLI.

Flow
----
1. Show each finding one at a time in the terminal
2. User chooses: Confirm / Reject / Change Severity / Add Note / Skip
3. After review:
   - Rejected findings removed from report
   - Severity overrides applied
   - Health score recalculated
   - Feedback logged to feedback_log.json
4. Returns an updated DiagnosisReport

Feedback log schema
-------------------
{
  "session_id": "uuid4",
  "timestamp":  "ISO-8601",
  "model_hash": "sha-256",
  "model_name": "str",
  "domain":     "str",
  "feedback": [
    {
      "finding_id":    "L1.1",
      "finding_name":  "Overfitting",
      "action":        "confirmed" | "rejected" | "severity_changed" | "note_added" | "skipped",
      "original_severity": "HIGH",
      "new_severity":       "CRITICAL" | null,
      "user_note":          "str" | null
    }
  ],
  "original_health_score":  int,
  "validated_health_score": int,
  "findings_removed":       int,
  "findings_escalated":     int
}
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

from core.report import Finding, DiagnosisReport, Severity
from core.correlation_engine import _rank_findings, _compute_health_score, _detect_interactions


# Constants
DEFAULT_FEEDBACK_LOG = "feedback_log.json"

_SEVERITY_MAP = {
    "1": Severity.CRITICAL,
    "2": Severity.HIGH,
    "3": Severity.MEDIUM,
    "4": Severity.LOW,
}

_ANSI_STRIP = re.compile(r"\033\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI_STRIP.sub("", text)


# Terminal helpers
def _hr(char: str = "─", width: int = 68) -> None:
    print(char * width)


def _header(text: str) -> None:
    _hr("═")
    print(f"  {text}")
    _hr("═")


def _ask(prompt: str, valid: List[str]) -> str:
    """Prompt until the user enters a valid choice (case-insensitive)."""
    while True:
        raw = input(prompt).strip()
        if raw.lower() in [v.lower() for v in valid]:
            return raw.lower()
        print(f"  Invalid input. Choose from: {valid}")


def _print_finding_summary(finding: Finding, index: int, total: int) -> None:
    """Print a concise finding card for review."""
    _hr()
    print(f"\n  Finding {index}/{total}  ·  {finding.severity.emoji()}  "
          f"[{finding.severity.value}]  {finding.name}  ({finding.id})\n")

    print("  Evidence:")
    for k, v in finding.evidence.items():
        if k in ("source", "domain_injected"):
            continue
        v_str = str(v)
        if len(v_str) > 70:
            v_str = v_str[:67] + "..."
        print(f"    {k}: {v_str}")

    print(f"\n  Why: {finding.explanation[:200]}{'...' if len(finding.explanation) > 200 else ''}")
    print(f"\n  Fix: {finding.fix.split(chr(10))[0][:120]}")

    if finding.notes:
        print(f"\n  Note: {finding.notes[:150]}{'...' if len(finding.notes) > 150 else ''}")
    print()


def _print_menu() -> None:
    print("  What would you like to do?")
    print("  [1] Confirm this finding  (keep as-is)")
    print("  [2] Reject this finding   (false positive — remove from report)")
    print("  [3] Change severity       (re-classify)")
    print("  [4] Add a note            (annotate without changing severity)")
    print("  [5] Skip                  (leave unchanged, don't log)")
    print()


def _ask_new_severity() -> Severity:
    print("\n  New severity:")
    print("  [1] CRITICAL   [2] HIGH   [3] MEDIUM   [4] LOW")
    choice = _ask("  Enter choice (1-4): ", ["1", "2", "3", "4"])
    return _SEVERITY_MAP[choice]


def _ask_note() -> str:
    return input("  Enter your note: ").strip()


# Feedback log
def _append_to_feedback_log(
    session: dict,
    log_path: str = DEFAULT_FEEDBACK_LOG,
) -> None:
    """
    Append a session record to the feedback log JSON file.
    Creates the file if it doesn't exist.
    Format: JSON array of session records.
    """
    # Load existing records
    existing: list = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
            if not isinstance(existing, list):
                existing = [existing]
        except (json.JSONDecodeError, IOError):
            existing = []

    existing.append(session)

    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2, default=str)


def _build_session_record(
    report: DiagnosisReport,
    validated_report: DiagnosisReport,
    feedback_items: list,
    model_hash: str,
) -> dict:
    removed   = sum(1 for f in feedback_items if f["action"] == "rejected")
    escalated = sum(
        1 for f in feedback_items
        if f["action"] == "severity_changed"
        and f.get("new_severity") is not None
        and Severity(f["new_severity"]) != Severity(f["original_severity"])
    )
    return {
        "session_id":             str(uuid.uuid4()),
        "timestamp":              datetime.now().isoformat(),
        "model_hash":             model_hash,
        "model_name":             report.model_name,
        "domain":                 report.domain,
        "feedback":               feedback_items,
        "original_health_score":  report.overall_health_score,
        "validated_health_score": validated_report.overall_health_score,
        "findings_removed":       removed,
        "findings_escalated":     escalated,
    }


# Core review loop
def _review_finding(
    finding: Finding,
    index: int,
    total: int,
) -> Tuple[Optional[Finding], dict]:
    """
    Present one finding to the user and collect their decision.

    Returns
    -------
    (updated_finding_or_None, feedback_item_dict)

    None means the finding was rejected.
    """
    _print_finding_summary(finding, index, total)
    _print_menu()

    choice = _ask("  Your choice (1-5): ", ["1", "2", "3", "4", "5"])

    feedback_item = {
        "finding_id":        finding.id,
        "finding_name":      finding.name,
        "action":            None,
        "original_severity": finding.severity.value,
        "new_severity":      None,
        "user_note":         None,
    }

    if choice == "1":
        # Confirm
        feedback_item["action"] = "confirmed"
        print(f"\n  ✅  Confirmed: {finding.name}\n")
        return finding, feedback_item

    elif choice == "2":
        # Reject
        reason = input("  Optional reason (press Enter to skip): ").strip()
        feedback_item["action"]    = "rejected"
        feedback_item["user_note"] = reason or None
        print(f"\n  🗑️  Rejected: {finding.name}\n")
        return None, feedback_item

    elif choice == "3":
        # Change severity
        new_sev = _ask_new_severity()
        user_note = input("  Reason for change (press Enter to skip): ").strip()

        updated = Finding(
            id=finding.id,
            name=finding.name,
            severity=new_sev,
            evidence=finding.evidence,
            explanation=finding.explanation,
            fix=finding.fix,
            confidence=finding.confidence,
            notes=(
                f"{finding.notes or ''}  [HUMAN OVERRIDE] Severity changed "
                f"from {finding.severity.value} to {new_sev.value}."
                + (f" Reason: {user_note}" if user_note else "")
            ),
        )
        feedback_item["action"]       = "severity_changed"
        feedback_item["new_severity"] = new_sev.value
        feedback_item["user_note"]    = user_note or None
        print(f"\n  ✏️  Severity changed: {finding.severity.value} → {new_sev.value}\n")
        return updated, feedback_item

    elif choice == "4":
        # Add note
        note = _ask_note()
        updated = Finding(
            id=finding.id,
            name=finding.name,
            severity=finding.severity,
            evidence=finding.evidence,
            explanation=finding.explanation,
            fix=finding.fix,
            confidence=finding.confidence,
            notes=f"{finding.notes or ''}  [HUMAN NOTE] {note}",
        )
        feedback_item["action"]    = "note_added"
        feedback_item["user_note"] = note
        print(f"\n  📝  Note added to: {finding.name}\n")
        return updated, feedback_item

    else:
        # Skip
        feedback_item["action"] = "skipped"
        print(f"\n  ⏭️  Skipped: {finding.name}\n")
        return finding, feedback_item


# Public API
def run_interactive_review(
    report: DiagnosisReport,
    feedback_log_path: str = DEFAULT_FEEDBACK_LOG,
    model_hash: str = "",
) -> DiagnosisReport:
    """
    Run an interactive terminal review of all findings.

    The user reviews each finding and can confirm, reject, re-classify,
    or annotate it. The session is logged to feedback_log.json.

    Parameters
    ----------
    report            : DiagnosisReport from run_diagnosis()
    feedback_log_path : Path to append feedback records
    model_hash        : SHA-256 hash of the model (for log tracing)

    Returns
    -------
    Updated DiagnosisReport with rejected findings removed and
    severity overrides applied.
    """
    if not report.findings:
        print("\n  No findings to review — report is clean.\n")
        return report

    _header(
        f"HUMAN-IN-THE-LOOP REVIEW  ·  {len(report.findings)} finding(s)"
    )
    print(
        "\n  You will review each finding one at a time.\n"
        "  Rejected findings are removed from the final report.\n"
        "  All decisions are logged to: " + feedback_log_path + "\n"
    )

    validated_findings: List[Finding] = []
    feedback_items: list = []

    for i, finding in enumerate(report.findings, start=1):
        updated, fb = _review_finding(finding, i, len(report.findings))
        feedback_items.append(fb)
        if updated is not None:
            validated_findings.append(updated)

    # Rebuild report from validated findings
    ranked       = _rank_findings(validated_findings)
    interactions = _detect_interactions(ranked)
    new_score    = max(0, _compute_health_score(ranked) - len(interactions) * 5)

    # Rebuild action sequence from updated findings
    from core.correlation_engine import _generate_action_sequence
    new_actions = _generate_action_sequence(ranked)

    validated_report = DiagnosisReport(
        model_name=report.model_name,
        task_type=report.task_type,
        domain=report.domain,
        overall_health_score=new_score,
        findings=ranked,
        interaction_warnings=interactions,
        recommended_action_sequence=new_actions,
    )

    # Print summary
    removed = len(report.findings) - len(validated_findings)
    _hr("═")
    print(f"\n  Review complete.")
    print(f"  Original findings : {len(report.findings)}")
    print(f"  Findings removed  : {removed}")
    print(f"  Remaining         : {len(validated_findings)}")
    print(f"  Original score    : {report.overall_health_score}/100")
    print(f"  Validated score   : {new_score}/100")
    _hr("═")

    #  Log to feedback file
    session = _build_session_record(
        report, validated_report, feedback_items, model_hash
    )
    _append_to_feedback_log(session, feedback_log_path)
    print(f"\n  Feedback logged → {feedback_log_path}\n")

    return validated_report