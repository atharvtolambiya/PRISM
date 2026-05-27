"""
BenchmarkRunner — evaluates the MLFIE engine's diagnostic accuracy
by running it against datasets with known injected failures.

Metrics computed
----------------
Detection Rate (DR)
    TP / (TP + FN) per failure type.
    "Did the engine find the failure that was actually there?"
    Target: > 0.85

False Alarm Rate (FAR)
    FP / (FP + TN) on clean datasets.
    "How often does the engine cry wolf?"
    Target: < 0.15

Severity Accuracy (SA)
    Fraction of detected failures whose severity matches the hint.
    Target: > 0.70

Fix Recommendation Rate (FRR)
    Fraction of detected failures that include a non-empty fix.
    Target: 1.0 (always provide a fix)

Priority Score (PS)
    For multi-failure datasets: was the injected failure ranked #1?
    Target: > 0.75
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.ingestion import load_model_input
from core.correlation_engine import run_diagnosis
from core.report import DiagnosisReport, Severity
from evaluation.injector import FailureInjector, InjectedBundle



@dataclass
class CaseResult:
    """Result for one injected test case."""
    bundle:            InjectedBundle
    report:            DiagnosisReport
    detected:          bool           # Was the injected failure found?
    finding_ids:       List[str]      # All finding IDs the engine produced
    severity_match:    bool           # Did severity match the hint?
    fix_provided:      bool           # Was a fix present in the finding?
    ranked_first:      bool           # Was injected failure ranked #1?
    false_alarm:       bool           # Clean baseline but findings returned?
    elapsed_seconds:   float


@dataclass
class BenchmarkReport:
    """Aggregated results across all test cases."""
    case_results:          List[CaseResult]   = field(default_factory=list)
    detection_rate:        Dict[str, float]   = field(default_factory=dict)
    overall_detection_rate: float             = 0.0
    false_alarm_rate:      float              = 0.0
    severity_accuracy:     float              = 0.0
    fix_recommendation_rate: float            = 0.0
    priority_score:        float              = 0.0
    avg_elapsed_seconds:   float              = 0.0
    n_cases:               int                = 0
    n_clean_cases:         int                = 0
    n_failure_cases:       int                = 0
    passed:                bool               = False   # True if all targets met

    # Targets
    TARGET_DR:   float = 0.85
    TARGET_FAR:  float = 0.15
    TARGET_SA:   float = 0.70
    TARGET_FRR:  float = 1.00
    TARGET_PS:   float = 0.75



class BenchmarkRunner:
    """
    Runs the MLFIE engine against a suite of injected failure bundles
    and computes diagnostic accuracy metrics.

    Usage
    -----
        runner  = BenchmarkRunner(domain="general")
        report  = runner.run()
        runner.print_report(report)
        runner.save(report, "benchmark_results.json")
    """

    def __init__(
        self,
        domain: str = "general",
        random_state: int = 42,
        n_clean_cases: int = 5,
    ):
        self.domain        = domain
        self.random_state  = random_state
        self.n_clean_cases = n_clean_cases
        self.injector      = FailureInjector(random_state=random_state)


    def _build_suite(self) -> List[InjectedBundle]:
        """
        Build the full set of injected test bundles.

        Failure bundles — one per failure type:
          L1.1  Overfitting
          L0.1  Class Imbalance (severe)
          L0.1  Class Imbalance (moderate)
          L0.2  Data Leakage
          L0.3  Data Quality
          L2.1  Covariate Shift
          L2.2  Label Shift
          L3.1  Metric Mismatch

        Clean bundles — `n_clean_cases` clean baselines for FAR measurement.
        """
        bundles = [
            self.injector.inject_overfitting(noise_ratio=0.35),
            self.injector.inject_class_imbalance(minority_ratio=0.04),
            self.injector.inject_class_imbalance(minority_ratio=0.12),
            self.injector.inject_data_leakage(),
            self.injector.inject_data_quality(),
            self.injector.inject_distribution_shift(shift_magnitude=10.0),
            self.injector.inject_label_shift(shift_ratio=0.45),
            self.injector.inject_metric_mismatch(minority_ratio=0.04),
        ]
        for i in range(self.n_clean_cases):
            inj = FailureInjector(random_state=self.random_state + i + 1)
            bundles.append(inj.make_clean_baseline())

        return bundles


    def _run_case(self, bundle: InjectedBundle) -> CaseResult:
        """Run the engine on one bundle and record the result."""
        mi = load_model_input(
            model=bundle.model,
            X_train=bundle.X_train,
            X_test=bundle.X_test,
            y_train=bundle.y_train,
            y_test=bundle.y_test,
            task_type=bundle.task_type,
            domain=self.domain,
        )

        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = run_diagnosis(mi)
        elapsed = time.time() - t0

        finding_ids = [f.id for f in report.findings]
        is_clean    = bundle.injected_failure == "NONE"

        # Detection: does any finding ID match any of the injected codes?
        injected_codes = set(bundle.injected_failure.split(","))
        # L2.x shift types are related — L2.2 injection detected as L2.3 is correct
        expanded_codes = set(injected_codes)
        if "L2.2" in injected_codes or "L2.3" in injected_codes:
            expanded_codes.update(["L2.1", "L2.2", "L2.3"])
        detected = (
            False if is_clean
            else bool(expanded_codes & set(finding_ids))
        )

        # False alarm: clean case but engine produced findings
        false_alarm = is_clean and len(report.findings) > 0

        # Severity match: check against the primary injected code or any match
        primary_code   = bundle.injected_failure.split(",")[0]
        severity_match = False
        fix_provided   = False
        ranked_first   = False

        if detected:
            for finding in report.findings:
                if finding.id in expanded_codes:
                    # Severity: accept same or more severe than hint
                    hint_sev = Severity(bundle.severity_hint) if bundle.severity_hint != "NONE" else None
                    if hint_sev:
                        sev_rank = {
                            Severity.CRITICAL: 0, Severity.HIGH: 1,
                            Severity.MEDIUM: 2,   Severity.LOW: 3
                        }
                        severity_match = (
                            sev_rank[finding.severity] <= sev_rank[hint_sev] + 1
                        )
                    fix_provided = bool(finding.fix and finding.fix.strip())
                    break

            # Ranked first: any matching failure is the top finding
            if report.findings and report.findings[0].id in expanded_codes:
                ranked_first = True

        return CaseResult(
            bundle=bundle,
            report=report,
            detected=detected,
            finding_ids=finding_ids,
            severity_match=severity_match,
            fix_provided=fix_provided,
            ranked_first=ranked_first,
            false_alarm=false_alarm,
            elapsed_seconds=elapsed,
        )


    def _compute_metrics(
        self,
        case_results: List[CaseResult],
    ) -> BenchmarkReport:
        failure_cases = [r for r in case_results if r.bundle.injected_failure != "NONE"]
        clean_cases   = [r for r in case_results if r.bundle.injected_failure == "NONE"]

        # Detection rate per failure type
        by_type: Dict[str, List[bool]] = {}
        for r in failure_cases:
            code = r.bundle.injected_failure.split(",")[0]
            by_type.setdefault(code, []).append(r.detected)

        dr_per_type = {
            code: round(sum(hits) / len(hits), 3)
            for code, hits in by_type.items()
        }

        # Overall detection rate
        overall_dr = (
            round(sum(r.detected for r in failure_cases) / len(failure_cases), 3)
            if failure_cases else 0.0
        )

        # False alarm rate
        far = (
            round(sum(r.false_alarm for r in clean_cases) / len(clean_cases), 3)
            if clean_cases else 0.0
        )

        # Severity accuracy (over detected cases only)
        detected_cases = [r for r in failure_cases if r.detected]
        sa = (
            round(sum(r.severity_match for r in detected_cases) / len(detected_cases), 3)
            if detected_cases else 0.0
        )

        # Fix recommendation rate
        frr = (
            round(sum(r.fix_provided for r in detected_cases) / len(detected_cases), 3)
            if detected_cases else 0.0
        )

        # Priority score
        ps = (
            round(sum(r.ranked_first for r in detected_cases) / len(detected_cases), 3)
            if detected_cases else 0.0
        )

        # Average elapsed time
        avg_elapsed = round(
            sum(r.elapsed_seconds for r in case_results) / len(case_results), 4
        ) if case_results else 0.0

        bm = BenchmarkReport(
            case_results=case_results,
            detection_rate=dr_per_type,
            overall_detection_rate=overall_dr,
            false_alarm_rate=far,
            severity_accuracy=sa,
            fix_recommendation_rate=frr,
            priority_score=ps,
            avg_elapsed_seconds=avg_elapsed,
            n_cases=len(case_results),
            n_clean_cases=len(clean_cases),
            n_failure_cases=len(failure_cases),
        )

        # Evaluate against targets
        bm.passed = (
            bm.overall_detection_rate >= bm.TARGET_DR
            and bm.false_alarm_rate    <= bm.TARGET_FAR
            and bm.severity_accuracy   >= bm.TARGET_SA
            and bm.fix_recommendation_rate >= bm.TARGET_FRR
        )

        return bm


    def run(self) -> BenchmarkReport:
        """
        Build the test suite, run every case, compute metrics.
        Returns a BenchmarkReport.
        """
        suite   = self._build_suite()
        results = []

        for i, bundle in enumerate(suite, start=1):
            label = f"[{bundle.injected_failure}] {bundle.description}"
            print(f"  Running case {i:2d}/{len(suite)}  {label[:60]}", flush=True)
            result = self._run_case(bundle)

            status = "✓ DETECTED" if result.detected else (
                "— CLEAN   " if bundle.injected_failure == "NONE"
                else "✗ MISSED  "
            )
            if bundle.injected_failure != "NONE":
                fa_marker = ""
            else:
                fa_marker = " ← FALSE ALARM" if result.false_alarm else ""

            findings_str = ", ".join(result.finding_ids) if result.finding_ids else "none"
            print(f"           {status}  findings={findings_str}{fa_marker}")
            results.append(result)

        return self._compute_metrics(results)


    def print_report(self, bm: BenchmarkReport) -> None:
        """Print the benchmark summary table to stdout."""
        W = 68

        def row(label, value, target, achieved):
            tick = "✅" if achieved else "❌"
            return f"  {label:<35} {value:>6}   target: {target:>5}  {tick}"

        print("\n" + "═" * W)
        print(f"  MLFIE BENCHMARK RESULTS")
        print(f"  Domain: {self.domain}   Cases: {bm.n_cases} "
              f"({bm.n_failure_cases} failure + {bm.n_clean_cases} clean)")
        print("═" * W)

        print(f"\n  {'METRIC':<35} {'VALUE':>6}   {'TARGET':>8}  STATUS")
        print("  " + "─" * (W - 2))

        print(row("Overall Detection Rate",
                  f"{bm.overall_detection_rate:.1%}",
                  f">{bm.TARGET_DR:.0%}",
                  bm.overall_detection_rate >= bm.TARGET_DR))

        print(row("False Alarm Rate",
                  f"{bm.false_alarm_rate:.1%}",
                  f"<{bm.TARGET_FAR:.0%}",
                  bm.false_alarm_rate <= bm.TARGET_FAR))

        print(row("Severity Accuracy",
                  f"{bm.severity_accuracy:.1%}",
                  f">{bm.TARGET_SA:.0%}",
                  bm.severity_accuracy >= bm.TARGET_SA))

        print(row("Fix Recommendation Rate",
                  f"{bm.fix_recommendation_rate:.1%}",
                  f"={bm.TARGET_FRR:.0%}",
                  bm.fix_recommendation_rate >= bm.TARGET_FRR))

        print(row("Priority Score",
                  f"{bm.priority_score:.1%}",
                  f">{bm.TARGET_PS:.0%}",
                  bm.priority_score >= bm.TARGET_PS))

        print(f"\n  Avg diagnosis time : {bm.avg_elapsed_seconds:.4f}s")

        print("\n  Detection rate per failure type:")
        for code, dr in sorted(bm.detection_rate.items()):
            tick = "✅" if dr >= bm.TARGET_DR else "❌"
            print(f"    {code:<8} {dr:.1%}  {tick}")

        print("\n" + "─" * W)
        overall = "✅  ALL TARGETS MET" if bm.passed else "❌  SOME TARGETS MISSED"
        print(f"  VERDICT: {overall}")
        print("═" * W + "\n")


    def save(
        self,
        bm: BenchmarkReport,
        output_path: str = "benchmark_results.json",
    ) -> None:
        """Serialise benchmark results to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)),
                    exist_ok=True)

        data = {
            "domain":                   self.domain,
            "n_cases":                  bm.n_cases,
            "n_failure_cases":          bm.n_failure_cases,
            "n_clean_cases":            bm.n_clean_cases,
            "overall_detection_rate":   bm.overall_detection_rate,
            "false_alarm_rate":         bm.false_alarm_rate,
            "severity_accuracy":        bm.severity_accuracy,
            "fix_recommendation_rate":  bm.fix_recommendation_rate,
            "priority_score":           bm.priority_score,
            "avg_elapsed_seconds":      bm.avg_elapsed_seconds,
            "detection_rate_by_type":   bm.detection_rate,
            "passed":                   bm.passed,
            "targets": {
                "detection_rate":        bm.TARGET_DR,
                "false_alarm_rate":      bm.TARGET_FAR,
                "severity_accuracy":     bm.TARGET_SA,
                "fix_recommendation_rate": bm.TARGET_FRR,
                "priority_score":        bm.TARGET_PS,
            },
            "cases": [
                {
                    "injected_failure": r.bundle.injected_failure,
                    "description":      r.bundle.description,
                    "detected":         r.detected,
                    "false_alarm":      r.false_alarm,
                    "finding_ids":      r.finding_ids,
                    "severity_match":   r.severity_match,
                    "fix_provided":     r.fix_provided,
                    "ranked_first":     r.ranked_first,
                    "elapsed_seconds":  r.elapsed_seconds,
                }
                for r in bm.case_results
            ],
        }

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

        print(f"  Benchmark results saved → {output_path}")