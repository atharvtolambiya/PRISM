<div align="center">

<img src="https://img.shields.io/badge/PRISM-ML%20Diagnostic%20Engine-6C63FF?style=for-the-badge&logoColor=white" alt="PRISM"/>

# PRISM
### *Predictive Reliability and Integrity Scoring for Models*

**The diagnostic engine that tells you exactly why your ML model is failing —**
**ranked by severity, backed by evidence, with specific fixes.**

<br/>

[![Tests](https://img.shields.io/badge/tests-221%20passing-4CAF50?style=flat-square&logo=checkmarx&logoColor=white)]()
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-compatible-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)]()
[![Status](https://img.shields.io/badge/Status-Research%20Grade-8B5CF6?style=flat-square)]()

<br/>

```
Model + Data  →  PRISM  →  Ranked Diagnosis  →  Specific Fixes
```

</div>

---

## The Problem

Your model has 94% accuracy. You deploy it. Three weeks later something feels wrong.

Now what?

Most teams do this:

```
→ Stare at a confusion matrix
→ Try random fixes (more data? different model? tune hyperparameters?)
→ Waste 2–3 days guessing
→ Maybe find the root cause. Maybe not.
```

There is no standardised way to diagnose why an ML model failed.
Every team reinvents this from scratch. Every time.

**PRISM is that standard.**

---

## What PRISM Does

PRISM ingests your trained model and dataset, runs a battery of diagnostic tests,
and produces a structured failure report — ranked by severity, with executable fixes.

```
                    Your Model + Train/Test Data
                              │
                              ▼
            ┌─────────────────────────────────────┐
            │         Layer 1 — Detectors         │
            │                                     │
            │  ┌──────────┐    ┌───────────────┐  │
            │  │Overfitting│    │Class Imbalance│  │
            │  ├──────────┤    ├───────────────┤  │
            │  │ Leakage  │    │ Data Quality  │  │
            │  ├──────────┤    ├───────────────┤  │
            │  │Dist.Shift│    │Metric Mismatch│  │
            │  └──────────┘    └───────────────┘  │
            └─────────────────┬───────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────┐
            │      Layer 2 — Correlation Engine   │
            │                                     │
            │  Detects compound failures          │
            │  (what individual tools miss)       │
            │  Ranks findings by severity         │
            │  Computes health score 0–100        │
            └─────────────────┬───────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────┐
            │      Layer 3 — Domain Rules         │
            │                                     │
            │  healthcare │ finance │ nlp │general│
            │  Adjusts severity for your context  │
            │  Injects domain-specific advisories │
            └─────────────────┬───────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  PRISM Report   │
                    │                 │
                    │  Health: 34/100 │
                    │  4 findings     │
                    │  2 interactions │
                    │  Action sequence│
                    └─────────────────┘
```

---

## Real Output

```
╔══════════════════════════════════════════════════════════════════════╗
║                    PRISM DIAGNOSTIC REPORT                           ║
║  Model : XGBoostClassifier      Domain : finance                     ║
║  Health: ████░░░░░░  34/100  🔴 CRITICAL                            ║
╚══════════════════════════════════════════════════════════════════════╝

🔴 [CRITICAL]  Data Leakage  (L0.2)  confidence: 97%

   Evidence   :  suspicious_features = {'future_payment_flag': 0.983}
                 correlation_threshold = 0.95
   Why        :  Feature 'future_payment_flag' has near-perfect
                 correlation with the target. It almost certainly
                 contains or is derived from the label. All performance
                 metrics are invalid.
   Fix        :  1. Remove 'future_payment_flag' from the feature set.
                 2. Audit your preprocessing pipeline — ensure all
                    transformers are fitted on training data only.
                 3. Check for any features derived from post-event data.

🔴 [CRITICAL]  Overfitting  (L1.1)  confidence: 95%

   Evidence   :  train_accuracy=0.9870, test_accuracy=0.7120, gap=27.5%
   Why        :  The model has memorised training data. A 27.5% gap
                 means it will fail silently on real-world inputs.
   Fix        :  Set max_depth=6, early_stopping_rounds=50,
                 reg_lambda=1.5. Reduce n_estimators from 900 to 200.

⚠  INTERACTION WARNING
   Data Leakage + Overfitting are compounding. All performance numbers
   are invalid until leakage is fixed. Fix L0.2 FIRST.

📋  RECOMMENDED ACTION SEQUENCE
   Step 1 [L0.2] → Remove 'future_payment_flag' from features  (~4-8 hrs)
   Step 2 [L1.1] → Reduce model complexity                      (~1-3 hrs)
   Step 3 [L0.1] → Set scale_pos_weight=49 for imbalance        (~2-4 hrs)
   Step 4 [L3.1] → Switch primary metric to PR-AUC              (~30 mins)
```

---

## How It Differs From Existing Tools

| Capability | SHAP | Evidently | Deepchecks | **PRISM** |
|---|:---:|:---:|:---:|:---:|
| Root cause diagnosis | ❌ | ❌ | Partial | ✅ |
| Cross-signal correlation | ❌ | ❌ | ❌ | ✅ |
| Formal failure taxonomy | ❌ | ❌ | ❌ | ✅ |
| Ranked findings with evidence | ❌ | ❌ | ❌ | ✅ |
| Executable fix recommendations | ❌ | ❌ | Partial | ✅ |
| Domain-aware severity rules | ❌ | ❌ | ❌ | ✅ |
| Human-in-the-loop validation | ❌ | ❌ | ❌ | ✅ |
| Controlled evaluation benchmark | ❌ | ❌ | ❌ | ✅ |

> SHAP answers *"what features drove this prediction?"*
> PRISM answers *"why is this model systematically failing?"*
> These are fundamentally different questions.

---

## Failure Taxonomy

PRISM is built on a formal 4-layer taxonomy — itself a research contribution.
No standardised ML failure classification like this currently exists in published form.

```
L0  Data Failures
    ├── L0.1  Class Imbalance
    ├── L0.2  Data Leakage
    └── L0.3  Data Quality  (nulls · duplicates · constant columns)

L1  Model-Data Fit Failures
    ├── L1.1  Overfitting
    ├── L1.2  Underfitting
    └── L1.3  Model-Complexity Mismatch

L2  Distribution Failures
    ├── L2.1  Covariate Shift
    ├── L2.2  Label Shift
    └── L2.3  Concept Drift

L3  Evaluation Failures
    ├── L3.1  Metric Mismatch
    ├── L3.2  Validation Design Failure
    └── L3.3  Model Miscalibration
```

---

## Domain Intelligence

The same finding means something different depending on your context.
PRISM adjusts automatically.

```
Finding: Class Imbalance  (2% minority class)

  General    →  🟠 HIGH
  Healthcare →  🔴 CRITICAL  +  "Clinical applications cannot
                                  ignore rare conditions."
  Finance    →  🟡 MEDIUM    +  "2% minority is normal for fraud
                                  detection — expected, not a defect."
```

### Healthcare Rules
- Overfitting → always **CRITICAL** (patient safety risk)
- Metric mismatch → always **CRITICAL** (accuracy unacceptable clinically)
- Distribution shift → always **CRITICAL** (FDA requires re-validation)
- Subgroup audit advisory always injected (regulatory requirement)

### Finance Rules
- Imbalance ≥ 0.5% → de-escalated to **MEDIUM** (normal for fraud)
- Distribution shift → always **CRITICAL** (fraud patterns change monthly)
- FPR-threshold advisory always injected (AUC alone is insufficient)

### NLP Rules
- Data leakage → always **CRITICAL** (text overlap inflates benchmarks)
- Data quality → always **HIGH** (null text can't be meaningfully imputed)
- Lexical overlap + writing style bias advisories always injected

---

## Quick Start

### Install

```bash
git clone https://github.com/atharvtolambiya/PRISM.git
cd PRISM
pip install -r requirements.txt
```

### Run the demo (no files needed)

```bash
python main.py demo --domain healthcare
```

### Diagnose your own model

```bash
python main.py diagnose \
    --model   model.pkl       \
    --train   train.csv       \
    --test    test.csv        \
    --target  churn           \
    --task    classification  \
    --domain  finance         \
    --output  report.json
```

### Python API

```python
from core.ingestion import load_model_input
from core.correlation_engine import run_diagnosis
from core.report_generator import print_report

model_input = load_model_input(
    model     = your_trained_model,
    X_train   = X_train,
    X_test    = X_test,
    y_train   = y_train,
    y_test    = y_test,
    task_type = "classification",
    domain    = "healthcare",
)

report = run_diagnosis(model_input)
print_report(report)
report.save_json("report.json")
```

### Streamlit Dashboard

```bash
streamlit run app.py
```

---

## Evaluation

Benchmark run against 13 controlled injection cases (8 failure types + 5 clean):

| Metric | Result | Target | Status |
|---|:---:|:---:|:---:|
| Overall Detection Rate | **100%** | > 85% | ✅ |
| Severity Accuracy | **100%** | > 70% | ✅ |
| Fix Recommendation Rate | **100%** | 100% | ✅ |
| Priority Score | **100%** | > 75% | ✅ |
| Avg Diagnosis Time | **0.011s** | < 1s | ✅ |

Every failure type detected. Every detected finding has a specific fix.
Diagnosis takes milliseconds.

### Reproduce the benchmark yourself

```bash
python - << 'EOF'
from evaluation.benchmark import BenchmarkRunner
runner = BenchmarkRunner(domain="general")
bm     = runner.run()
runner.print_report(bm)
runner.save(bm, "benchmark_results.json")
EOF
```

---

## Run the Tests

```bash
# All 221 tests
python -m unittest discover -s tests -v

# Individual modules
python -m unittest tests.test_ingestion          -v
python -m unittest tests.test_detectors          -v
python -m unittest tests.test_correlation_engine -v
python -m unittest tests.test_domain_rules       -v
python -m unittest tests.test_report_generator   -v
python -m unittest tests.test_hitl               -v
python -m unittest tests.test_evaluation         -v
python -m unittest tests.test_app                -v
```

---

## Project Structure

```
PRISM/
├── core/
│   ├── ingestion.py          # ModelInput validation + loading
│   ├── report.py             # Finding, DiagnosisReport, Severity
│   ├── registry.py           # Failure taxonomy L0–L3
│   ├── correlation_engine.py # Orchestrator: rank · interact · score
│   ├── report_generator.py   # Formatted text + JSON output
│   └── hitl.py               # Human-in-the-loop review + feedback log
│
├── detectors/
│   ├── overfitting.py        # L1.1 — train-test gap (KS + score delta)
│   ├── class_imbalance.py    # L0.1 — minority class ratio
│   ├── data_leakage.py       # L0.2 — feature-target correlation
│   ├── data_quality.py       # L0.3 — nulls, duplicates, constants
│   ├── distribution_shift.py # L2.x — KS test + label proportion diff
│   └── metric_mismatch.py    # L3.1 — accuracy vs F1 gap
│
├── domain/
│   └── rules.py              # Healthcare · finance · NLP · general rules
│
├── evaluation/
│   ├── injector.py           # Controlled failure injection (7 types)
│   └── benchmark.py          # Detection accuracy measurement
│
├── tests/                    # 221 tests — all synthetic data, no leaks
├── scripts/
│   └── generate_sample_data.py   # Creates 3 ready-to-use sample datasets
├── notebooks/
│   └── demo.py               # End-to-end walkthrough
├── app.py                    # Streamlit dashboard
├── main.py                   # CLI entry point
└── requirements.txt
```

---

## Research Contributions

1. **First formal ML failure taxonomy** — L0–L3, 12 failure types, no equivalent exists in published form
2. **Cross-signal correlation engine** — detects compound failures individual tools miss
3. **Domain-aware severity layer** — same failure, different stakes, encoded as executable rules
4. **Development-stage focus** — existing tools target production; PRISM targets development where fixing is 100× cheaper
5. **Controlled evaluation via failure injection** — reusable benchmark methodology for measuring diagnostic accuracy

---

## Tech Stack

```
Python 3.9+     scikit-learn     pandas     numpy
scipy           streamlit        argparse
```

No heavy dependencies. Core engine runs on scikit-learn + pandas + scipy only.

---

## License

MIT — free to use, modify, and publish.

---

<div align="center">

Built to turn *"why is my model broken?"* from a 3-day guessing game
into a 3-second diagnosis.

**[View Demo](https://github.com/atharvtolambiya/PRISM)**  ·
**[Report Bug](https://github.com/atharvtolambiya/PRISM/issues)**  ·
**[Request Feature](https://github.com/atharvtolambiya/PRISM/issues)**

</div>
