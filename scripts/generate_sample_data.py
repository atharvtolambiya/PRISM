"""
Generates realistic sample datasets with injected failures
so anyone can run MLFIE immediately without their own data.

Outputs
-------
  samples/churn_train.csv          — customer churn, balanced
  samples/churn_test.csv
  samples/churn_model.pkl

  samples/fraud_train.csv          — fraud detection, severe imbalance
  samples/fraud_test.csv
  samples/fraud_model.pkl

  samples/overfit_train.csv        — small noisy dataset, overfit model
  samples/overfit_test.csv
  samples/overfit_model.pkl

Usage
-----
  python scripts/generate_sample_data.py
  python main.py diagnose --model samples/churn_model.pkl \\
      --train samples/churn_train.csv --test samples/churn_test.csv \\
      --target churn --domain general

  python main.py diagnose --model samples/fraud_model.pkl \\
      --train samples/fraud_train.csv --test samples/fraud_test.csv \\
      --target is_fraud --domain finance

  python main.py diagnose --model samples/overfit_model.pkl \\
      --train samples/overfit_train.csv --test samples/overfit_test.csv \\
      --target outcome --domain general
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")


def _save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(df)} rows, {df.shape[1]} cols)")


def _save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    print(f"  Saved: {path}")


# ── Dataset 1: Customer Churn (balanced, clean baseline) ─────────────────────

def generate_churn():
    print("\n[1/3] Generating customer churn dataset (balanced, clean)...")
    rng  = np.random.default_rng(42)
    n    = 1000

    tenure          = rng.integers(1, 72, n)
    monthly_charges = rng.uniform(20, 120, n)
    num_products    = rng.integers(1, 5, n)
    support_calls   = rng.integers(0, 10, n)
    contract_type   = rng.integers(0, 3, n)   # 0=monthly, 1=annual, 2=biannual

    # Churn probability increases with monthly charges and support calls
    churn_prob = (
        0.05
        + 0.003  * (monthly_charges - 70)
        + 0.04   * support_calls
        - 0.005  * tenure
        - 0.08   * contract_type
    )
    churn_prob = np.clip(churn_prob, 0.05, 0.90)
    churn      = (rng.random(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "tenure":           tenure,
        "monthly_charges":  monthly_charges.round(2),
        "num_products":     num_products,
        "support_calls":    support_calls,
        "contract_type":    contract_type,
        "churn":            churn,
    })

    train, test = train_test_split(df, test_size=0.25, random_state=42,
                                   stratify=df["churn"])
    X_train = train.drop(columns=["churn"])
    y_train = train["churn"]
    X_test  = test.drop(columns=["churn"])

    model = LogisticRegression(max_iter=500, C=0.5)
    model.fit(X_train, y_train)

    _save_csv(train, os.path.join(OUTPUT_DIR, "churn_train.csv"))
    _save_csv(test,  os.path.join(OUTPUT_DIR, "churn_test.csv"))
    _save_model(model, os.path.join(OUTPUT_DIR, "churn_model.pkl"))
    print("  Injected failures: NONE (clean baseline)")
    print("  Expected findings: possibly minor metric mismatch")


# ── Dataset 2: Fraud Detection (severe imbalance) ─────────────────────────────

def generate_fraud():
    print("\n[2/3] Generating fraud detection dataset (severe imbalance ~2%)...")
    rng = np.random.default_rng(7)
    n   = 2000

    transaction_amount   = rng.exponential(150, n).round(2)
    hour_of_day          = rng.integers(0, 24, n)
    merchant_category    = rng.integers(0, 10, n)
    card_age_months      = rng.integers(1, 120, n)
    foreign_transaction  = rng.integers(0, 2, n)
    velocity_last_hour   = rng.integers(0, 20, n)

    # ~2% fraud rate
    fraud_prob = (
        0.005
        + 0.0003 * transaction_amount
        + 0.06   * foreign_transaction
        + 0.005  * velocity_last_hour
    )
    fraud_prob = np.clip(fraud_prob, 0.002, 0.85)
    is_fraud   = (rng.random(n) < fraud_prob).astype(int)

    # Ensure at least 2% fraud
    n_fraud = is_fraud.sum()
    if n_fraud < int(n * 0.015):
        extra_idx = rng.choice(np.where(is_fraud == 0)[0],
                               size=int(n * 0.02) - n_fraud, replace=False)
        is_fraud[extra_idx] = 1

    df = pd.DataFrame({
        "transaction_amount":  transaction_amount,
        "hour_of_day":         hour_of_day,
        "merchant_category":   merchant_category,
        "card_age_months":     card_age_months,
        "foreign_transaction": foreign_transaction,
        "velocity_last_hour":  velocity_last_hour,
        "is_fraud":            is_fraud,
    })

    train, test = train_test_split(df, test_size=0.25, random_state=0,
                                   stratify=df["is_fraud"])
    X_train = train.drop(columns=["is_fraud"])
    y_train = train["is_fraud"]

    # Train WITHOUT class weight — model ignores fraud → triggers metric mismatch
    model = LogisticRegression(max_iter=500, class_weight=None)
    model.fit(X_train, y_train)

    _save_csv(train, os.path.join(OUTPUT_DIR, "fraud_train.csv"))
    _save_csv(test,  os.path.join(OUTPUT_DIR, "fraud_test.csv"))
    _save_model(model, os.path.join(OUTPUT_DIR, "fraud_model.pkl"))
    fraud_rate = is_fraud.mean()
    print(f"  Injected failures: L0.1 (class imbalance, {fraud_rate:.1%} fraud)")
    print("  Expected findings: L0.1 class imbalance, L3.1 metric mismatch")


# ── Dataset 3: Overfit Model (small data + unlimited tree) ────────────────────

def generate_overfit():
    print("\n[3/3] Generating overfitting dataset (small + noisy labels)...")
    rng = np.random.default_rng(99)
    n   = 300   # intentionally small → easier to overfit

    age        = rng.integers(18, 80, n)
    income     = rng.normal(50000, 20000, n).round(0)
    score_a    = rng.uniform(0, 100, n).round(1)
    score_b    = rng.uniform(0, 100, n).round(1)
    category   = rng.integers(0, 5, n)

    # Add 25% null values to score_b — triggers data quality finding
    null_idx   = rng.choice(n, size=int(n * 0.25), replace=False)
    score_b    = score_b.astype(float)
    score_b[null_idx] = np.nan

    # True outcome based on simple rules
    outcome_prob = (
        0.4
        + 0.005 * (score_a - 50)
        + 0.003 * (income / 10000 - 5)
    )
    outcome_prob = np.clip(outcome_prob, 0.05, 0.95)
    outcome      = (rng.random(n) < outcome_prob).astype(int)

    # Add 20% label noise → forces tree to memorise noise
    flip_idx     = rng.choice(n, size=int(n * 0.20), replace=False)
    noisy_outcome = outcome.copy()
    noisy_outcome[flip_idx] = 1 - noisy_outcome[flip_idx]

    df = pd.DataFrame({
        "age":      age,
        "income":   income,
        "score_a":  score_a,
        "score_b":  score_b,   # 25% nulls
        "category": category,
        "outcome":  noisy_outcome,
    })

    train, test = train_test_split(df, test_size=0.30, random_state=42)

    X_train = train.drop(columns=["outcome"]).fillna(
        train.drop(columns=["outcome"]).median(numeric_only=True)
    )
    y_train = train["outcome"]

    # Unlimited tree on noisy labels → guaranteed overfitting
    model = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1,
                                   random_state=0)
    model.fit(X_train, y_train)

    _save_csv(train, os.path.join(OUTPUT_DIR, "overfit_train.csv"))
    _save_csv(test,  os.path.join(OUTPUT_DIR, "overfit_test.csv"))
    _save_model(model, os.path.join(OUTPUT_DIR, "overfit_model.pkl"))
    print("  Injected failures: L1.1 (overfitting), L0.3 (25% nulls in score_b)")
    print("  Expected findings: L1.1 overfitting, L0.3 data quality")



def main():
    print("=" * 60)
    print("  MLFIE Sample Data Generator")
    print("=" * 60)

    generate_churn()
    generate_fraud()
    generate_overfit()

    print("\n" + "=" * 60)
    print("  Done! Run these commands to diagnose each dataset:\n")
    print("  # Clean baseline (minor findings)")
    print("  python main.py diagnose \\")
    print("    --model samples/churn_model.pkl \\")
    print("    --train samples/churn_train.csv \\")
    print("    --test  samples/churn_test.csv  \\")
    print("    --target churn --domain general\n")
    print("  # Fraud (imbalance + metric mismatch)")
    print("  python main.py diagnose \\")
    print("    --model samples/fraud_model.pkl \\")
    print("    --train samples/fraud_train.csv \\")
    print("    --test  samples/fraud_test.csv  \\")
    print("    --target is_fraud --domain finance\n")
    print("  # Overfitting + data quality")
    print("  python main.py diagnose \\")
    print("    --model samples/overfit_model.pkl \\")
    print("    --train samples/overfit_train.csv \\")
    print("    --test  samples/overfit_test.csv  \\")
    print("    --target outcome --domain general")
    print("=" * 60)


if __name__ == "__main__":
    main()