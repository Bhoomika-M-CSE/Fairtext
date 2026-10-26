"""Evaluation suite, scenario testing, and error analysis for FairText.

Runs comparative benchmarking across TensorFlow and PyTorch classifiers on:
- Test split (Accuracy, Precision, Recall, F1, Confusion Matrix)
- Held-out StereoSet and CrowS-Pairs gender benchmarks
- Bolukbasi hard debiasing WEAT effect size audits
- 7 core scenario evaluations (job ads, headlines, textbooks, neutral, female-surgeon, quoted, 'they' pronoun)
- Error analysis exporting misclassified test instances to results/errors.csv
- Consolidates complete performance metrics in results/summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src import config
from src.classifier_tf import evaluate_tf_model, predict_tf
from src.classifier_torch import evaluate_torch_model, predict_torch
from src.data_loader import load_crows_pairs, load_stereoset
from src.debias import before_after_report
from src.scoring import bias_score

SCENARIO_TESTS: List[Dict[str, str]] = [
    {
        "name": "job_ad_masculine",
        "description": "Masculine-coded job advertisement",
        "text": "We are seeking an aggressive and dominant sales leader who can ruthlessly crush competitors.",
    },
    {
        "name": "job_ad_neutral",
        "description": "Inclusive and neutral job advertisement",
        "text": "We are seeking a collaborative software engineer with experience designing scalable web services.",
    },
    {
        "name": "headline",
        "description": "Sensationalized gendered media headline",
        "text": "Emotional female prime minister faces fierce backlash following controversial budget vote.",
    },
    {
        "name": "textbook_passage",
        "description": "Factual scientific textbook excerpt",
        "text": "Mitochondria generate most of the chemical energy needed to power biochemical reactions in cells.",
    },
    {
        "name": "neutral_text",
        "description": "Neutral everyday professional discourse",
        "text": "The librarian sorted the newly arrived reference books and updated the digital catalog.",
    },
    {
        "name": "female_surgeon",
        "description": "Counter-stereotypical female surgeon sentence",
        "text": "The female surgeon performed the delicate operation with precision and steady focus.",
    },
    {
        "name": "quoted_speech",
        "description": "Quoted gender-specific speech",
        "text": "'The chairman insisted on inspecting every policeman's credentials,' the witness testified.",
    },
    {
        "name": "they_pronoun",
        "description": "Singular 'they' gender-neutral sentence",
        "text": "The developer said they would push the feature branch to the repository after unit testing.",
    },
]


def evaluate_stereoset_pairs(model_type: str = "torch") -> Dict[str, Any]:
    """Evaluate stereotype vs anti-stereotype bias score preference on StereoSet."""
    df = load_stereoset()
    stereo_preferred = 0
    total = len(df)

    for _, row in df.iterrows():
        st_text = str(row["stereotype"])
        anti_text = str(row["anti_stereotype"])
        score_st, _ = bias_score(st_text, classifier_type=model_type)
        score_anti, _ = bias_score(anti_text, classifier_type=model_type)
        if score_st >= score_anti:
            stereo_preferred += 1

    rate = (stereo_preferred / total) if total > 0 else 0.5
    return {
        "benchmark": "StereoSet (Gender)",
        "model_type": model_type,
        "sample_count": total,
        "stereotype_score_preference_rate": round(rate, 4),
    }


def evaluate_crows_pairs_benchmark(model_type: str = "torch") -> Dict[str, Any]:
    """Evaluate stereotype vs less-stereotypical sentence preference on CrowS-Pairs."""
    df = load_crows_pairs()
    more_biased_count = 0
    total = len(df)

    for _, row in df.iterrows():
        more_text = str(row["sent_more"])
        less_text = str(row["sent_less"])
        score_more, _ = bias_score(more_text, classifier_type=model_type)
        score_less, _ = bias_score(less_text, classifier_type=model_type)
        if score_more >= score_less:
            more_biased_count += 1

    rate = (more_biased_count / total) if total > 0 else 0.5
    return {
        "benchmark": "CrowS-Pairs (Gender)",
        "model_type": model_type,
        "sample_count": total,
        "stereotypical_preference_rate": round(rate, 4),
    }


def run_scenarios(model_type: str = "torch") -> List[Dict[str, Any]]:
    """Execute scenario tests across predefined text archetypes."""
    results: List[Dict[str, Any]] = []
    for item in SCENARIO_TESTS:
        score, b = bias_score(item["text"], classifier_type=model_type)
        results.append({
            "scenario": item["name"],
            "description": item["description"],
            "text": item["text"],
            "bias_score": score,
            "is_flagged": b["is_flagged"],
            "classifier_prob": b["classifier_prob"],
            "coded_words": b["coded_language_component"],
            "stereotypical_links": b["stereotypical_links_count"],
        })
    return results


def error_analysis(test_path: Optional[Path] = None, top_k: int = 10) -> pd.DataFrame:
    """Identify, inspect, and export misclassified examples on test split."""
    t_path = test_path or config.TEST_PATH
    df = pd.read_csv(t_path)
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        text = str(row["text"])
        true_label = int(row["label"])
        tf_pred, tf_prob = predict_tf(text)
        torch_pred, torch_prob = predict_torch(text)

        is_error = (tf_pred != true_label) or (torch_pred != true_label)
        if is_error:
            records.append({
                "text": text,
                "true_label": true_label,
                "tf_pred": tf_pred,
                "tf_prob": tf_prob,
                "torch_pred": torch_pred,
                "torch_prob": torch_prob,
                "error_type": "False Positive" if (torch_pred == 1 and true_label == 0) else "False Negative",
            })

    error_df = pd.DataFrame(records).head(top_k)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    error_df.to_csv(config.ERRORS_CSV_PATH, index=False)
    return error_df


def run_all() -> Dict[str, Any]:
    """Orchestrate entire evaluation pipeline and export summary.json."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Running FairText Comprehensive Evaluation ===")

    print("1. Evaluating Classifiers on Held-Out Test Split...")
    tf_metrics = evaluate_tf_model()
    torch_metrics = evaluate_torch_model()

    print("2. Evaluating Held-Out StereoSet & CrowS-Pairs...")
    stereoset_res = evaluate_stereoset_pairs(model_type="torch")
    crows_res = evaluate_crows_pairs_benchmark(model_type="torch")

    print("3. Evaluating WEAT Before/After Hard Debiasing...")
    debias_df = before_after_report()

    print("4. Running Scenario Tests...")
    scenarios = run_scenarios(model_type="torch")

    print("5. Conducting Error Analysis...")
    err_df = error_analysis()

    summary: Dict[str, Any] = {
        "classifiers": {
            "tensorflow_bilstm": tf_metrics,
            "distilbert_pytorch": torch_metrics,
        },
        "benchmarks": {
            "stereoset": stereoset_res,
            "crows_pairs": crows_res,
        },
        "debias_audit": debias_df.to_dict(orient="records"),
        "scenario_evaluations": scenarios,
        "error_analysis_count": len(err_df),
    }

    with open(config.SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Done] Summary saved to: {config.SUMMARY_JSON_PATH}")
    print(f"[Done] Errors saved to: {config.ERRORS_CSV_PATH}")
    return summary


if __name__ == "__main__":
    run_all()
