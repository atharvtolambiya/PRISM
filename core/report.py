"""
Core data structures for MLFIE findings and diagnosis reports.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import json


class Severity(str, Enum):
    """Severity levels for a diagnosis finding."""
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"

    def score(self) -> int:
        """Points deducted from health score per finding."""
        return {
            Severity.CRITICAL: 30,
            Severity.HIGH:     15,
            Severity.MEDIUM:    7,
            Severity.LOW:       3,
        }[self]

    def emoji(self) -> str:
        return {
            Severity.CRITICAL: "🔴",
            Severity.HIGH:     "🟠",
            Severity.MEDIUM:   "🟡",
            Severity.LOW:      "🟢",
        }[self]

    @staticmethod
    def order() -> List[str]:
        """Severity rank for sorting (lowest index = highest priority)."""
        return ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


@dataclass
class Finding:
    """
    A single diagnosed model failure.

    Attributes
    ----------
    id          : Taxonomy code, e.g. "L0.1"
    name        : Human-readable failure name
    severity    : CRITICAL | HIGH | MEDIUM | LOW
    evidence    : Dict of metric_name -> value that triggered detection
    explanation : Why this failure matters / how it causes the observed problem
    fix         : Specific, actionable recommendation (code-level where possible)
    confidence  : Engine certainty in this diagnosis (0.0 – 1.0)
    notes       : Optional extra context (domain overrides, interaction flags)
    """
    id:          str
    name:        str
    severity:    Severity
    evidence:    Dict[str, object]
    explanation: str
    fix:         str
    confidence:  float                  = 1.0
    notes:       Optional[str]          = None


    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if isinstance(self.severity, str):
            self.severity = Severity(self.severity.upper())


    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "name":        self.name,
            "severity":    self.severity.value,
            "evidence":    self.evidence,
            "explanation": self.explanation,
            "fix":         self.fix,
            "confidence":  round(self.confidence, 4),
            "notes":       self.notes,
        }

    def __str__(self) -> str:
        bar = "█" * int(self.confidence * 10) + "░" * (10 - int(self.confidence * 10))
        lines = [
            f"{self.severity.emoji()} [{self.severity.value}] {self.name}  "
            f"(id={self.id}, confidence={bar} {self.confidence:.0%})",
            f"  Evidence   : {self.evidence}",
            f"  Why        : {self.explanation}",
            f"  Fix        : {self.fix}",
        ]
        if self.notes:
            lines.append(f"  Notes      : {self.notes}")
        return "\n".join(lines)


@dataclass
class DiagnosisReport:
    """
    Full output of a MLFIE diagnosis run.

    Attributes
    ----------
    model_name                 : Display name of the model
    task_type                  : 'classification' or 'regression'
    domain                     : Domain context used
    overall_health_score       : 0-100 composite score
    findings                   : List[Finding] sorted by severity
    interaction_warnings       : Cross-finding compound warnings
    recommended_action_sequence: Ordered list of fix steps
    """
    model_name:                  str
    task_type:                   str
    domain:                      str
    overall_health_score:        int
    findings:                    List[Finding]         = field(default_factory=list)
    interaction_warnings:        List[str]             = field(default_factory=list)
    recommended_action_sequence: List[str]             = field(default_factory=list)


    def health_label(self) -> str:
        s = self.overall_health_score
        if s >= 80:
            return "🟢 HEALTHY"
        if s >= 60:
            return "🟡 NEEDS ATTENTION"
        if s >= 40:
            return "🟠 POOR"
        return "🔴 CRITICAL"


    def to_dict(self) -> dict:
        return {
            "model_name":                  self.model_name,
            "task_type":                   self.task_type,
            "domain":                      self.domain,
            "overall_health_score":        self.overall_health_score,
            "health_label":                self.health_label(),
            "findings":                    [f.to_dict() for f in self.findings],
            "interaction_warnings":        self.interaction_warnings,
            "recommended_action_sequence": self.recommended_action_sequence,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str) -> None:
        with open(path, "w") as fh:
            fh.write(self.to_json())


    def __str__(self) -> str:
        width = 60
        border = "═" * width
        lines = [
            f"╔{border}╗",
            f"║{'ML FAILURE INTELLIGENCE REPORT':^{width}}║",
            f"║{f'Model : {self.model_name}':^{width}}║",
            f"║{f'Domain: {self.domain}  |  Task: {self.task_type}':^{width}}║",
            f"║{f'Health Score: {self.overall_health_score}/100  {self.health_label()}':^{width}}║",
            f"╚{border}╝",
            "",
        ]

        if not self.findings:
            lines.append("  ✅  No failures detected. Model looks healthy.")
        else:
            lines.append(f"  FINDINGS ({len(self.findings)} total)")
            lines.append("  " + "─" * (width - 2))
            for finding in self.findings:
                lines.append(str(finding))
                lines.append("  " + "─" * (width - 2))

        if self.interaction_warnings:
            lines.append("\n  ⚠  INTERACTION WARNINGS")
            for w in self.interaction_warnings:
                lines.append(f"  → {w}")

        if self.recommended_action_sequence:
            lines.append("\n  📋  RECOMMENDED ACTION SEQUENCE")
            for step in self.recommended_action_sequence:
                lines.append(f"  {step}")

        return "\n".join(lines)