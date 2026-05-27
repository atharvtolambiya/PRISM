"""
main.py
-------
MLFIE Command-Line Interface — Phase 5.

Usage
-----
  python main.py diagnose \
      --model   path/to/model.pkl \
      --train   path/to/train.csv \
      --test    path/to/test.csv  \
      --target  column_name       \
      --task    classification    \
      --domain  healthcare        \
      --output  report.json

  python main.py demo --domain healthcare
  python main.py list-domains
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import pandas as pd


def _load_model(path: str):
    if not os.path.exists(path):
        print(f"[ERROR] Model file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        print(f"[ERROR] Could not load model: {exc}", file=sys.stderr)
        sys.exit(1)


def _load_csv(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[ERROR] {label} file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[ERROR] Could not read {label}: {exc}", file=sys.stderr)
        sys.exit(1)


def _split_xy(df: pd.DataFrame, target: str, label: str):
    if target not in df.columns:
        print(
            f"[ERROR] Target column '{target}' not found in {label}.\n"
            f"        Available columns: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)
    return df.drop(columns=[target]), df[target]


def _spinner(message: str) -> None:
    print(f"\n  >> {message}", flush=True)


# ── Command: diagnose ─────────────────────────────────────────────────────────

def cmd_diagnose(args: argparse.Namespace) -> None:
    from core.ingestion import load_model_input, IngestionError
    from core.correlation_engine import run_diagnosis
    from core.report_generator import print_report, save_report

    _spinner("Loading model...")
    model = _load_model(args.model)

    _spinner("Loading datasets...")
    train_df = _load_csv(args.train, "train CSV")
    test_df  = _load_csv(args.test,  "test CSV")
    X_train, y_train = _split_xy(train_df, args.target, "train CSV")
    X_test,  y_test  = _split_xy(test_df,  args.target, "test CSV")

    _spinner("Validating inputs...")
    try:
        model_input = load_model_input(
            model=model,
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            task_type=args.task,
            domain=args.domain,
        )
    except IngestionError as exc:
        print(f"\n[ERROR] Input validation failed:\n  {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n  OK  Loaded: {model_input}")

    _spinner("Running diagnosis engine...")
    t0      = time.time()
    report  = run_diagnosis(model_input)
    elapsed = time.time() - t0
    print(f"  OK  Diagnosis complete in {elapsed:.2f}s\n")

    print_report(report)

    if args.output:
        paths = save_report(report, args.output, also_save_text=True)
        print(f"\n  SAVED  JSON -> {paths['json_path']}")
        if "text_path" in paths:
            print(f"  SAVED  Text -> {paths['text_path']}")

    if getattr(args, "interactive", False):
        from core.hitl import run_interactive_review
        report = run_interactive_review(report)
        if args.output:
            validated_path = args.output.replace(".json", "_validated.json")
            paths = save_report(report, validated_path, also_save_text=True)
            print(f"\n  SAVED  Validated report -> {paths['json_path']}")

    sys.exit(0 if report.overall_health_score >= 60 else 1)


# ── Command: demo ─────────────────────────────────────────────────────────────

def cmd_demo(args: argparse.Namespace) -> None:
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from core.ingestion import load_model_input
    from core.correlation_engine import run_diagnosis
    from core.report_generator import print_report, save_report

    print("\n  MLFIE - Built-in Demo")
    print("  Injecting: class imbalance + overfitting into synthetic dataset\n")

    rng = np.random.default_rng(42)
    n   = 500
    y   = pd.Series([0] * 450 + [1] * 50, name="target")
    X   = pd.DataFrame(
        rng.standard_normal((n, 6)),
        columns=[f"feature_{i}" for i in range(6)]
    )

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    model = DecisionTreeClassifier(max_depth=None, random_state=0)
    model.fit(X_train, y_train)

    domain = getattr(args, "domain", "healthcare")
    model_input = load_model_input(
        model, X_train, X_test, y_train, y_test,
        task_type="classification",
        domain=domain,
        model_name="DecisionTreeClassifier (Demo)",
    )

    _spinner("Running diagnosis engine...")
    t0      = time.time()
    report  = run_diagnosis(model_input)
    elapsed = time.time() - t0
    print(f"  OK  Diagnosis complete in {elapsed:.2f}s\n")

    print_report(report)

    if getattr(args, "output", None):
        paths = save_report(report, args.output, also_save_text=True)
        print(f"\n  SAVED  Report -> {paths['json_path']}")


# ── Command: list-domains ──────────────────────────────────────────────────────

def cmd_list_domains(args: argparse.Namespace) -> None:
    from domain.rules import list_supported_domains
    descriptions = {
        "general":    "Default - no domain-specific overrides.",
        "healthcare": "Clinical AI - recall/safety-driven rules, subgroup audit.",
        "finance":    "Fraud/risk - cost-sensitive, regulatory compliance rules.",
        "nlp":        "Text models - leakage, vocabulary shift, style bias.",
    }
    print("\n  Supported domains:\n")
    for d in list_supported_domains():
        print(f"  {d:<14} {descriptions.get(d, '')}")
    print()


# ── Argument parser ────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlfie",
        description="ML Failure Intelligence Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py diagnose --model model.pkl --train train.csv --test test.csv
                          --target churn --task classification --domain finance
                          --output report.json

  python main.py diagnose ... --interactive

  python main.py demo --domain healthcare --output demo_report.json

  python main.py list-domains
        """,
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # diagnose
    diag = sub.add_parser("diagnose", help="Diagnose a trained model.")
    diag.add_argument("--model",  required=True, help="Path to .pkl model file.")
    diag.add_argument("--train",  required=True, help="Path to training CSV.")
    diag.add_argument("--test",   required=True, help="Path to test CSV.")
    diag.add_argument("--target", required=True, help="Target column name.")
    diag.add_argument("--task",   choices=["classification", "regression"],
                      default="classification")
    diag.add_argument("--domain", choices=["general", "healthcare", "finance", "nlp"],
                      default="general")
    diag.add_argument("--output",      default=None, help="Path to save JSON report.")
    diag.add_argument("--interactive", action="store_true",
                      help="Enable human-in-the-loop review.")

    # demo
    demo = sub.add_parser("demo", help="Run built-in synthetic demo.")
    demo.add_argument("--domain", choices=["general", "healthcare", "finance", "nlp"],
                      default="healthcare")
    demo.add_argument("--output", default=None)

    # list-domains
    sub.add_parser("list-domains", help="List supported domain names.")

    return parser


def main() -> None:
    parser  = build_parser()
    args    = parser.parse_args()
    handler = {
        "diagnose":     cmd_diagnose,
        "demo":         cmd_demo,
        "list-domains": cmd_list_domains,
    }.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()