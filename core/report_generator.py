"""
Transforms a DiagnosisReport into:
  1. A formatted, human-readable console report (rich text via ANSI or plain)
  2. A JSON file saved to disk

Design goals
------------
- A junior developer who didn't build the model should be able to read
  the report and know exactly what to fix next.
- Every section is independently renderable for Streamlit (Phase 8).
- No external dependencies — plain Python only.
"""

from __future__ import annotations

import json
import os
import textwrap
from datetime import datetime
from typing import Optional

from core.report import DiagnosisReport, Finding, Severity


_WIDTH       = 70
_INNER_WIDTH = _WIDTH - 4   # inside padded borders


_USE_COLOUR = os.getenv("NO_COLOUR") is None and os.getenv("MLFIE_PLAIN") is None

_RESET  = "\033[0m"  if _USE_COLOUR else ""
_BOLD   = "\033[1m"  if _USE_COLOUR else ""

_COLOUR = {
    "red":    "\033[91m" if _USE_COLOUR else "",
    "orange": "\033[33m" if _USE_COLOUR else "",
    "yellow": "\033[93m" if _USE_COLOUR else "",
    "green":  "\033[92m" if _USE_COLOUR else "",
    "cyan":   "\033[96m" if _USE_COLOUR else "",
    "grey":   "\033[90m" if _USE_COLOUR else "",
    "white":  "\033[97m" if _USE_COLOUR else "",
}

_SEVERITY_COLOUR = {
    Severity.CRITICAL: _COLOUR["red"],
    Severity.HIGH:     _COLOUR["orange"],
    Severity.MEDIUM:   _COLOUR["yellow"],
    Severity.LOW:      _COLOUR["green"],
}


def _c(text: str, colour: str) -> str:
    return f"{_COLOUR.get(colour, '')}{text}{_RESET}"


def _bold(text: str) -> str:
    return f"{_BOLD}{text}{_RESET}"


def _severity_badge(severity: Severity) -> str:
    colour = _SEVERITY_COLOUR.get(severity, "")
    return f"{colour}{_BOLD}[{severity.value}]{_RESET}"


def _divider(char: str = "─") -> str:
    return _c(char * _WIDTH, "grey")


def _section_header(title: str) -> str:
    pad   = (_WIDTH - len(title) - 2) // 2
    left  = "─" * pad
    right = "─" * (_WIDTH - len(title) - 2 - pad)
    return _c(f"{left} {_bold(title)} {right}", "cyan")


def _render_header(report: DiagnosisReport) -> str:
    ts    = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    score = report.overall_health_score
    label = report.health_label()

    # Health bar  ████████░░  70/100
    filled = int(score / 10)
    bar    = "█" * filled + "░" * (10 - filled)
    if score >= 80:
        bar_coloured = _c(bar, "green")
    elif score >= 60:
        bar_coloured = _c(bar, "yellow")
    elif score >= 40:
        bar_coloured = _c(bar, "orange")
    else:
        bar_coloured = _c(bar, "red")

    lines = [
        "╔" + "═" * _WIDTH + "╗",
        "║" + _bold(" ML FAILURE INTELLIGENCE REPORT").center(_WIDTH + len(_BOLD) + len(_RESET)) + "║",
        "║" + " " * _WIDTH + "║",
        f"║  {_bold('Model  :')} {report.model_name:<{_INNER_WIDTH - 10}}║",
        f"║  {_bold('Task   :')} {report.task_type:<{_INNER_WIDTH - 10}}║",
        f"║  {_bold('Domain :')} {report.domain:<{_INNER_WIDTH - 10}}║",
        f"║  {_bold('Run at :')} {ts:<{_INNER_WIDTH - 10}}║",
        "║" + " " * _WIDTH + "║",
        f"║  {_bold('Health :')} {bar_coloured}  {score}/100  {label}",
        "║" + " " * _WIDTH + "║",
        "╚" + "═" * _WIDTH + "╝",
    ]
    return "\n".join(lines)


def _render_finding(finding: Finding, index: int) -> str:
    badge     = _severity_badge(finding.severity)
    conf_pct  = f"{finding.confidence:.0%}"
    conf_bar  = "█" * int(finding.confidence * 8) + "░" * (8 - int(finding.confidence * 8))

    lines = [
        "",
        f"  {badge}  {_bold(finding.name)}  "
        f"{_c(f'({finding.id})', 'grey')}  "
        f"{_c(f'confidence: {conf_bar} {conf_pct}', 'grey')}",
        "",
    ]

    # Evidence block
    lines.append(f"  {_bold('Evidence:')}")
    for key, val in finding.evidence.items():
        if key in ("source", "domain_injected"):
            continue
        val_str = str(val)
        if len(val_str) > 60:
            val_str = val_str[:57] + "..."
        lines.append(f"    {_c(key, 'cyan')}: {val_str}")

    # Why this matters
    lines.append("")
    lines.append(f"  {_bold('Why this matters:')}")
    wrapped = textwrap.wrap(finding.explanation, width=_INNER_WIDTH - 4)
    for line in wrapped:
        lines.append(f"    {line}")

    # Fix
    lines.append("")
    lines.append(f"  {_bold('Fix:')}")
    for fix_line in finding.fix.split("\n"):
        wrapped_fix = textwrap.wrap(fix_line, width=_INNER_WIDTH - 4)
        for wl in wrapped_fix:
            lines.append(f"    {_c(wl, 'green')}")

    # Notes
    if finding.notes:
        lines.append("")
        lines.append(f"  {_c('Note:', 'grey')}")
        for note_line in textwrap.wrap(finding.notes, width=_INNER_WIDTH - 4):
            lines.append(f"    {_c(note_line, 'grey')}")

    lines.append("")
    lines.append("  " + _c("─" * (_WIDTH - 2), "grey"))

    return "\n".join(lines)


def _render_findings_section(report: DiagnosisReport) -> str:
    if not report.findings:
        return (
            "\n" + _section_header("FINDINGS") + "\n\n"
            + _c("  No failures detected. Model looks healthy!", "green")
            + "\n"
        )

    count_by_severity = {}
    for f in report.findings:
        count_by_severity[f.severity.value] = (
            count_by_severity.get(f.severity.value, 0) + 1
        )
    summary_parts = [
        f"{_severity_badge(Severity[k])} ×{v}"
        for k, v in count_by_severity.items()
    ]
    summary = "  " + "   ".join(summary_parts)

    header = (
        "\n" + _section_header(f"FINDINGS  ({len(report.findings)} total)") + "\n"
        + summary + "\n"
        + "  " + _c("─" * (_WIDTH - 2), "grey")
    )

    body = ""
    for i, finding in enumerate(report.findings, start=1):
        body += _render_finding(finding, i)

    return header + body


def _render_interactions_section(report: DiagnosisReport) -> str:
    if not report.interaction_warnings:
        return ""
    lines = [
        "",
        _section_header("INTERACTION WARNINGS"),
        "",
    ]
    for warning in report.interaction_warnings:
        wrapped = textwrap.wrap(warning, width=_INNER_WIDTH - 2)
        lines.append(f"  {_c(wrapped[0], 'orange')}")
        for line in wrapped[1:]:
            lines.append(f"  {_c(line, 'orange')}")
        lines.append("")
    return "\n".join(lines)


def _render_action_sequence(report: DiagnosisReport) -> str:
    if not report.recommended_action_sequence:
        return ""
    lines = [
        "",
        _section_header("RECOMMENDED ACTION SEQUENCE"),
        "",
    ]
    for step in report.recommended_action_sequence:
        lines.append(f"  {_c('→', 'cyan')} {step}")
    lines.append("")
    return "\n".join(lines)


def _render_footer() -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        "\n" + _divider() + "\n"
        + _c(
            f"  Generated by MLFIE — ML Failure Intelligence Engine  |  {ts}",
            "grey",
        )
        + "\n" + _divider()
    )



def render_report(report: DiagnosisReport) -> str:
    """
    Render a DiagnosisReport as a formatted, human-readable string.

    Sections:
      1. Header (model info + health score gauge)
      2. Findings (ranked, with evidence, explanation, fix, notes)
      3. Interaction warnings
      4. Recommended action sequence
      5. Footer

    Returns the full report as a string (print it or write to file).
    """
    sections = [
        _render_header(report),
        _render_findings_section(report),
        _render_interactions_section(report),
        _render_action_sequence(report),
        _render_footer(),
    ]
    return "\n".join(sections)


def save_report(
    report: DiagnosisReport,
    output_path: str,
    also_save_text: bool = True,
) -> dict:
    """
    Save the diagnosis report to disk.

    Always saves a JSON file at `output_path`.
    Optionally saves a plain-text version alongside it.

    Parameters
    ----------
    report          : DiagnosisReport to save
    output_path     : Path for the JSON file (e.g. 'report.json')
    also_save_text  : If True, also write a .txt file next to the JSON

    Returns
    -------
    dict with keys 'json_path' and optionally 'text_path'
    """
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    json_data = report.to_dict()
    json_data["generated_at"] = datetime.now().isoformat()
    json_data["mlfie_version"] = "1.0.0"

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(json_data, fh, indent=2, default=str)

    result = {"json_path": output_path}

    if also_save_text:
        # Strip ANSI codes for plain text file
        import re
        ansi_escape = re.compile(r"\033\[[0-9;]*m")
        plain_text  = ansi_escape.sub("", render_report(report))
        text_path   = output_path.replace(".json", ".txt")
        if text_path == output_path:
            text_path = output_path + ".txt"
        with open(text_path, "w", encoding="utf-8") as fh:
            fh.write(plain_text)
        result["text_path"] = text_path

    return result


def print_report(report: DiagnosisReport) -> None:
    """Print the formatted report directly to stdout."""
    print(render_report(report))