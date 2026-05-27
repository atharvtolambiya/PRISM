"""
app.py
------
MLFIE Streamlit Dashboard — Phase 8.

Three pages:
  1. Upload   — provide model + datasets, select task/domain, run diagnosis
  2. Report   — health score gauge, findings table, action sequence
  3. Feedback — review each finding, confirm/reject/re-classify, submit

Run with:
  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import pickle
import tempfile
import os

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="MLFIE — ML Failure Intelligence Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project imports ───────────────────────────────────────────────────────────
from core.ingestion import load_model_input, IngestionError
from core.correlation_engine import run_diagnosis
from core.report import DiagnosisReport, Finding, Severity
from core.report_generator import save_report
from core.hitl import _append_to_feedback_log, _build_session_record
from core.correlation_engine import (
    _rank_findings,
    _compute_health_score,
    _detect_interactions,
    _generate_action_sequence,
)
from domain.rules import list_supported_domains


# ────────────────────────────────────────────────────────────────────────────
# Severity colours for Streamlit
# ────────────────────────────────────────────────────────────────────────────

_SEV_COLOUR = {
    "CRITICAL": "#FF4B4B",
    "HIGH":     "#FF8C00",
    "MEDIUM":   "#FFD700",
    "LOW":      "#00CC66",
}

_SEV_BG = {
    "CRITICAL": "#2D0000",
    "HIGH":     "#2D1500",
    "MEDIUM":   "#2D2500",
    "LOW":      "#002D12",
}


def _sev_badge(severity: str) -> str:
    colour = _SEV_COLOUR.get(severity, "#888888")
    return (
        f'<span style="background:{colour};color:white;'
        f'padding:2px 8px;border-radius:4px;'
        f'font-weight:bold;font-size:0.85em">{severity}</span>'
    )


# ────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "report":          None,
        "model_input":     None,
        "page":            "Upload",
        "feedback_items":  [],
        "validated_report": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ────────────────────────────────────────────────────────────────────────────

def _sidebar():
    with st.sidebar:
        st.image(
            "https://raw.githubusercontent.com/simple-icons/simple-icons/develop/"
            "icons/python.svg",
            width=32,
        )
        st.markdown("## 🔬 MLFIE")
        st.markdown("*ML Failure Intelligence Engine*")
        st.markdown("---")

        pages = ["📤 Upload & Diagnose", "📊 Report", "✅ Feedback"]
        icons = ["📤", "📊", "✅"]
        labels = ["Upload", "Report", "Feedback"]

        for i, (page_label, label) in enumerate(zip(pages, labels)):
            active = st.session_state.page == label
            if st.button(
                page_label,
                use_container_width=True,
                type="primary" if active else "secondary",
                key=f"nav_{label}",
            ):
                st.session_state.page = label

        st.markdown("---")

        if st.session_state.report:
            report = st.session_state.report
            score  = report.overall_health_score
            colour = (
                "🟢" if score >= 80 else
                "🟡" if score >= 60 else
                "🟠" if score >= 40 else "🔴"
            )
            st.markdown(f"**Model:** `{report.model_name}`")
            st.markdown(f"**Health:** {colour} `{score}/100`")
            st.markdown(f"**Findings:** `{len(report.findings)}`")
            st.markdown(f"**Domain:** `{report.domain}`")

        st.markdown("---")
        st.markdown(
            '<small style="color:#888">MLFIE v1.0.0</small>',
            unsafe_allow_html=True,
        )


# ────────────────────────────────────────────────────────────────────────────
# Page 1 — Upload & Diagnose
# ────────────────────────────────────────────────────────────────────────────

def page_upload():
    st.title("📤 Upload & Diagnose")
    st.markdown(
        "Upload your trained model and datasets, configure the task context, "
        "then run the diagnosis engine."
    )

    # ── Column layout ─────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Upload Files")

        model_file  = st.file_uploader(
            "Model file (.pkl)",
            type=["pkl"],
            help="A pickled sklearn-compatible or XGBoost model.",
        )
        train_file  = st.file_uploader(
            "Training data (.csv)",
            type=["csv"],
            help="CSV file containing features + target column.",
        )
        test_file   = st.file_uploader(
            "Test data (.csv)",
            type=["csv"],
            help="CSV file containing features + target column.",
        )

    with col2:
        st.subheader("2. Configure")

        # Dynamically populate target column after CSV upload
        target_col = ""
        if train_file:
            try:
                preview_df = pd.read_csv(train_file)
                train_file.seek(0)
                target_col = st.selectbox(
                    "Target column",
                    options=list(preview_df.columns),
                    help="The column containing the labels/values to predict.",
                )
            except Exception:
                target_col = st.text_input("Target column name")
        else:
            target_col = st.text_input(
                "Target column name",
                placeholder="e.g. target, label, y",
            )

        task_type = st.selectbox(
            "Task type",
            options=["classification", "regression"],
        )

        domain = st.selectbox(
            "Domain",
            options=list_supported_domains(),
            help=(
                "general: no overrides  |  healthcare: safety-driven rules  |  "
                "finance: fraud/risk rules  |  nlp: text-pipeline rules"
            ),
        )

        model_name = st.text_input(
            "Model name (optional)",
            placeholder="e.g. XGBoost v2 — churn model",
        )

    st.markdown("---")

    # ── Demo shortcut ─────────────────────────────────────────────────────
    with st.expander("🧪 No files? Run the built-in demo"):
        demo_domain = st.selectbox(
            "Demo domain", list_supported_domains(), key="demo_domain"
        )
        if st.button("▶ Run Demo", type="primary"):
            with st.spinner("Running demo diagnosis..."):
                _run_demo(demo_domain)

    # ── Run diagnosis ─────────────────────────────────────────────────────
    st.markdown("---")
    run_btn = st.button(
        "🔬 Run Diagnosis",
        type="primary",
        disabled=not (model_file and train_file and test_file and target_col),
        use_container_width=True,
    )

    if run_btn:
        with st.spinner("Loading files and running diagnosis engine..."):
            try:
                # Load model
                model = pickle.loads(model_file.read())

                # Load CSVs
                train_df = pd.read_csv(train_file)
                test_df  = pd.read_csv(test_file)

                if target_col not in train_df.columns:
                    st.error(f"Target column `{target_col}` not found in train CSV.")
                    return
                if target_col not in test_df.columns:
                    st.error(f"Target column `{target_col}` not found in test CSV.")
                    return

                X_train = train_df.drop(columns=[target_col])
                y_train = train_df[target_col]
                X_test  = test_df.drop(columns=[target_col])
                y_test  = test_df[target_col]

                mi = load_model_input(
                    model=model,
                    X_train=X_train, X_test=X_test,
                    y_train=y_train, y_test=y_test,
                    task_type=task_type,
                    domain=domain,
                    model_name=model_name or "",
                )

                report = run_diagnosis(mi)

                st.session_state.report        = report
                st.session_state.model_input   = mi
                st.session_state.feedback_items = []
                st.session_state.validated_report = None
                st.session_state.page          = "Report"
                st.rerun()

            except IngestionError as exc:
                st.error(f"Input validation failed: {exc}")
            except Exception as exc:
                st.error(f"Diagnosis failed: {exc}")
                st.exception(exc)


def _run_demo(domain: str):
    """Run the built-in synthetic demo and store report in session state."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(42)
    n   = 500
    y   = pd.Series([0] * 450 + [1] * 50, name="target")
    X   = pd.DataFrame(
        rng.standard_normal((n, 6)),
        columns=[f"feature_{i}" for i in range(6)],
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=None, random_state=0)
    model.fit(X_train, y_train)

    mi = load_model_input(
        model, X_train, X_test, y_train, y_test,
        task_type="classification",
        domain=domain,
        model_name="DecisionTreeClassifier (Demo)",
    )
    report = run_diagnosis(mi)

    st.session_state.report           = report
    st.session_state.model_input      = mi
    st.session_state.feedback_items   = []
    st.session_state.validated_report = None
    st.session_state.page             = "Report"
    st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# Page 2 — Report
# ────────────────────────────────────────────────────────────────────────────

def page_report():
    st.title("📊 Diagnosis Report")

    report = st.session_state.get("report")
    if report is None:
        st.info("No report yet. Go to **Upload & Diagnose** to run the engine.")
        return

    # ── Header metrics ────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    score = report.overall_health_score

    with col1:
        colour = (
            "normal" if score >= 80 else
            "off" if score >= 60 else
            "inverse"
        )
        st.metric("Health Score", f"{score}/100", delta=report.health_label())

    with col2:
        st.metric("Findings", len(report.findings))

    with col3:
        critical = sum(1 for f in report.findings if f.severity == Severity.CRITICAL)
        st.metric("Critical", critical, delta="⚠ Review immediately" if critical else None)

    with col4:
        st.metric("Domain", report.domain.capitalize())

    # ── Health gauge ──────────────────────────────────────────────────────
    st.markdown("---")
    gauge_colour = (
        "#00CC66" if score >= 80 else
        "#FFD700" if score >= 60 else
        "#FF8C00" if score >= 40 else
        "#FF4B4B"
    )
    bar_pct = score
    st.markdown(
        f"""
        <div style="background:#333;border-radius:8px;height:28px;width:100%;margin-bottom:4px">
          <div style="background:{gauge_colour};width:{bar_pct}%;height:100%;
                      border-radius:8px;display:flex;align-items:center;
                      padding-left:12px;color:white;font-weight:bold;font-size:1em">
            {score}/100  {report.health_label()}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Findings table ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"Findings ({len(report.findings)})")

    if not report.findings:
        st.success("✅ No failures detected. Model looks healthy!")
    else:
        # Summary severity counts
        sev_counts = {}
        for f in report.findings:
            sev_counts[f.severity.value] = sev_counts.get(f.severity.value, 0) + 1

        cols = st.columns(len(sev_counts))
        for col, (sev, cnt) in zip(cols, sev_counts.items()):
            with col:
                st.markdown(
                    f'<div style="text-align:center">'
                    f'{_sev_badge(sev)}<br>'
                    f'<span style="font-size:1.5em;font-weight:bold">{cnt}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("")

        # Individual finding cards
        for i, finding in enumerate(report.findings, start=1):
            bg    = _SEV_BG.get(finding.severity.value, "#111")
            badge = _sev_badge(finding.severity.value)
            conf  = f"{finding.confidence:.0%}"

            with st.expander(
                f"{finding.severity.emoji()}  [{finding.id}]  {finding.name}  "
                f"— confidence {conf}",
                expanded=(finding.severity == Severity.CRITICAL),
            ):
                c1, c2 = st.columns([1, 1])

                with c1:
                    st.markdown("**Evidence**")
                    ev_rows = []
                    for k, v in finding.evidence.items():
                        if k in ("source", "domain_injected"):
                            continue
                        v_str = str(v)
                        if len(v_str) > 80:
                            v_str = v_str[:77] + "..."
                        ev_rows.append({"Metric": k, "Value": v_str})
                    if ev_rows:
                        st.dataframe(
                            pd.DataFrame(ev_rows),
                            use_container_width=True,
                            hide_index=True,
                        )

                with c2:
                    st.markdown("**Why this matters**")
                    st.markdown(finding.explanation)

                st.markdown("**Recommended Fix**")
                st.code(finding.fix, language="text")

                if finding.notes:
                    st.info(f"📝 {finding.notes}")

    # ── Interaction warnings ──────────────────────────────────────────────
    if report.interaction_warnings:
        st.markdown("---")
        st.subheader("⚠ Interaction Warnings")
        for warning in report.interaction_warnings:
            st.warning(warning)

    # ── Action sequence ───────────────────────────────────────────────────
    if report.recommended_action_sequence:
        st.markdown("---")
        st.subheader("📋 Recommended Action Sequence")
        for step in report.recommended_action_sequence:
            st.markdown(f"- {step}")

    # ── Download buttons ──────────────────────────────────────────────────
    st.markdown("---")
    col_dl1, col_dl2, _ = st.columns([1, 1, 2])

    with col_dl1:
        json_str = report.to_json()
        st.download_button(
            label="⬇ Download JSON",
            data=json_str,
            file_name="mlfie_report.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_dl2:
        import re
        ansi_escape = re.compile(r"\033\[[0-9;]*m")
        from core.report_generator import render_report
        plain_text = ansi_escape.sub("", render_report(report))
        st.download_button(
            label="⬇ Download Text",
            data=plain_text,
            file_name="mlfie_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # ── Navigate to feedback ──────────────────────────────────────────────
    st.markdown("---")
    if st.button("✅ Proceed to Human Review →", use_container_width=True, type="primary"):
        st.session_state.page = "Feedback"
        st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# Page 3 — Feedback / HITL
# ────────────────────────────────────────────────────────────────────────────

def page_feedback():
    st.title("✅ Human Review")
    st.markdown(
        "Review each finding. Confirm, reject, or reclassify. "
        "Your decisions are logged and the health score is recalculated."
    )

    report = st.session_state.get("report")
    if report is None:
        st.info("No report to review. Run a diagnosis first.")
        return

    if not report.findings:
        st.success("No findings to review — the model is clean!")
        return

    # ── Per-finding review forms ──────────────────────────────────────────
    decisions = {}
    severity_choices = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

    for finding in report.findings:
        badge = _sev_badge(finding.severity.value)
        st.markdown(
            f"### {finding.severity.emoji()} {finding.name}  "
            f"<small>{badge}&nbsp;&nbsp;`{finding.id}`&nbsp;&nbsp;"
            f"confidence: {finding.confidence:.0%}</small>",
            unsafe_allow_html=True,
        )

        with st.container():
            c1, c2 = st.columns([2, 1])

            with c1:
                st.markdown(f"**Why:** {finding.explanation[:300]}")
                st.markdown(f"**Fix:** `{finding.fix.split(chr(10))[0][:120]}`")

            with c2:
                action = st.radio(
                    "Action",
                    options=["Confirm", "Reject", "Change Severity"],
                    key=f"action_{finding.id}",
                    horizontal=False,
                )

                new_sev = finding.severity.value
                if action == "Change Severity":
                    new_sev = st.selectbox(
                        "New severity",
                        options=severity_choices,
                        index=severity_choices.index(finding.severity.value),
                        key=f"newsev_{finding.id}",
                    )

                user_note = st.text_input(
                    "Note (optional)",
                    key=f"note_{finding.id}",
                    placeholder="Add context or reason...",
                )

            decisions[finding.id] = {
                "finding":   finding,
                "action":    action,
                "new_sev":   new_sev,
                "user_note": user_note or None,
            }

        st.markdown("---")

    # ── Submit button ─────────────────────────────────────────────────────
    if st.button("📨 Submit Review & Recalculate", type="primary",
                 use_container_width=True):
        _process_feedback(report, decisions)


def _process_feedback(report: DiagnosisReport, decisions: dict):
    """Apply decisions, rebuild report, log feedback."""
    validated_findings = []
    feedback_items     = []

    for fid, dec in decisions.items():
        finding   = dec["finding"]
        action    = dec["action"]
        new_sev   = dec["new_sev"]
        user_note = dec["user_note"]

        fb = {
            "finding_id":        finding.id,
            "finding_name":      finding.name,
            "action":            action.lower().replace(" ", "_"),
            "original_severity": finding.severity.value,
            "new_severity":      None,
            "user_note":         user_note,
        }

        if action == "Reject":
            feedback_items.append(fb)
            continue

        updated_finding = finding
        if action == "Change Severity":
            fb["new_severity"] = new_sev
            updated_finding = Finding(
                id=finding.id,
                name=finding.name,
                severity=Severity(new_sev),
                evidence=finding.evidence,
                explanation=finding.explanation,
                fix=finding.fix,
                confidence=finding.confidence,
                notes=(
                    f"{finding.notes or ''}  [HUMAN OVERRIDE] "
                    f"Severity changed {finding.severity.value} → {new_sev}."
                    + (f" Reason: {user_note}" if user_note else "")
                ),
            )

        validated_findings.append(updated_finding)
        feedback_items.append(fb)

    # Rebuild report
    ranked       = _rank_findings(validated_findings)
    interactions = _detect_interactions(ranked)
    new_score    = max(0, _compute_health_score(ranked) - len(interactions) * 5)
    new_actions  = _generate_action_sequence(ranked)

    validated_report = DiagnosisReport(
        model_name=report.model_name,
        task_type=report.task_type,
        domain=report.domain,
        overall_health_score=new_score,
        findings=ranked,
        interaction_warnings=interactions,
        recommended_action_sequence=new_actions,
    )

    # Log feedback
    mi   = st.session_state.get("model_input")
    hash_ = mi.model_hash if mi else ""
    session = _build_session_record(
        report, validated_report, feedback_items, hash_
    )
    try:
        _append_to_feedback_log(session, "feedback_log.json")
    except Exception:
        pass  # Don't block UI on log write failure

    # Store results
    st.session_state.validated_report = validated_report
    st.session_state.feedback_items   = feedback_items

    # Show summary
    removed = len(report.findings) - len(validated_findings)
    st.success(
        f"✅ Review complete! "
        f"Findings: {len(report.findings)} → {len(validated_findings)}  |  "
        f"Removed: {removed}  |  "
        f"Health score: {report.overall_health_score} → {new_score}/100"
    )

    # Validated report download
    st.download_button(
        label="⬇ Download Validated Report (JSON)",
        data=validated_report.to_json(),
        file_name="mlfie_report_validated.json",
        mime="application/json",
        use_container_width=True,
    )


# ────────────────────────────────────────────────────────────────────────────
# Main router
# ────────────────────────────────────────────────────────────────────────────

def main():
    _init_state()
    _sidebar()

    page = st.session_state.page

    if page == "Upload":
        page_upload()
    elif page == "Report":
        page_report()
    elif page == "Feedback":
        page_feedback()
    else:
        page_upload()


if __name__ == "__main__":
    main()