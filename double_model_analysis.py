"""
Compare two model prediction folders and analyze ensemble strategies.
 
Filename convention: {true_label}_PRED_{pred_label}_{score}_{sample_id}_grade{Z}.png
  - true_label: 0 (normal) or 1 (anomalous)
  - pred_label: 0 or 1
  - score: continuous anomaly score (float)
  - Sample identity: img{X}_obj{Y}_grade{Z}
 
Class semantics: 1 = anomalous, 0 = normal
"""
 
import os
import re
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score



_FNAME_RE = re.compile(
    r"^(?P<true>[01])_PRED_(?P<pred>[01])_(?P<score>[0-9.]+)_(?P<sample_id>img\d+_obj\d+_grade\d+)\.png$"
)
 
 
@dataclass
class SamplePrediction:
    sample_id: str
    true_label: int
    pred_a: int
    pred_b: int
    score_a: float
    score_b: float
    grade: int
 
 
def _parse_folder(folder: str) -> Dict[str, Tuple[int, int, float]]:
    """Parse a prediction folder. Returns {sample_id: (true_label, pred_label, score)}."""
    results = {}
    for fname in os.listdir(folder):
        m = _FNAME_RE.match(fname)
        if m is None:
            continue
        sid = m.group("sample_id")
        true = int(m.group("true"))
        pred = int(m.group("pred"))
        score = float(m.group("score"))
        if sid in results:
            print(f"WARNING: duplicate sample {sid} in {folder}")
        results[sid] = (true, pred, score)
    return results
 
 
def _compute_metrics(true_labels: list, pred_labels: list) -> dict:
    """Binary classification metrics. Positive class = 1 (anomalous)."""
    assert len(true_labels) == len(pred_labels)
    n = len(true_labels)
    if n == 0:
        return {"n": 0, "accuracy": None, "precision": None, "recall": None, "f1": None}
 
    tp = sum(t == 1 and p == 1 for t, p in zip(true_labels, pred_labels))
    tn = sum(t == 0 and p == 0 for t, p in zip(true_labels, pred_labels))
    fp = sum(t == 0 and p == 1 for t, p in zip(true_labels, pred_labels))
    fn = sum(t == 1 and p == 0 for t, p in zip(true_labels, pred_labels))
 
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
 
    return {
        "n": n,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
    }
 
 
def compare_predictions(folder_a: str, folder_b: str, verbose: bool = True) -> dict:
    """
    Compare predictions from two model folders.
 
    Returns a dict with:
      - model_a / model_b: individual model metrics
      - intersection_AND / union_OR: ensemble metrics
      - agreement_rate: fraction of images where both models agree
      - disagreements: detailed breakdown per grade
      - samples: list of SamplePrediction (for downstream use, e.g. score fusion)
    """
    parsed_a = _parse_folder(folder_a)
    parsed_b = _parse_folder(folder_b)
 
    common_ids = sorted(set(parsed_a.keys()) & set(parsed_b.keys()))
    only_a = set(parsed_a.keys()) - set(parsed_b.keys())
    only_b = set(parsed_b.keys()) - set(parsed_a.keys())
 
    if only_a:
        print(f"WARNING: {len(only_a)} samples only in folder A")
    if only_b:
        print(f"WARNING: {len(only_b)} samples only in folder B")
 
    samples = []
    for sid in common_ids:
        true_a, pred_a, score_a = parsed_a[sid]
        true_b, pred_b, score_b = parsed_b[sid]
        if true_a != true_b:
            print(f"ERROR: true label mismatch for {sid}: A={true_a}, B={true_b}")
            continue
        grade = int(re.search(r"grade(\d+)", sid).group(1))
        samples.append(SamplePrediction(sid, true_a, pred_a, pred_b, score_a, score_b, grade))
 
    true_labels = [s.true_label for s in samples]
    preds_a = [s.pred_a for s in samples]
    preds_b = [s.pred_b for s in samples]
    preds_intersection = [int(a == 1 and b == 1) for a, b in zip(preds_a, preds_b)]
    preds_union = [int(a == 1 or b == 1) for a, b in zip(preds_a, preds_b)]
 
    results = {
        "n_common": len(samples),
        "n_only_a": len(only_a),
        "n_only_b": len(only_b),
        "model_a": _compute_metrics(true_labels, preds_a),
        "model_b": _compute_metrics(true_labels, preds_b),
        "intersection_AND": _compute_metrics(true_labels, preds_intersection),
        "union_OR": _compute_metrics(true_labels, preds_union),
        "samples": samples,
    }
 
    agree = sum(a == b for a, b in zip(preds_a, preds_b))
    results["agreement_rate"] = agree / len(samples) if samples else None
 
    disagreements = []
    for s in samples:
        if s.pred_a != s.pred_b:
            disagreements.append({
                "sample_id": s.sample_id,
                "true_label": s.true_label,
                "pred_a": s.pred_a,
                "pred_b": s.pred_b,
                "score_a": s.score_a,
                "score_b": s.score_b,
                "grade": s.grade,
            })
    results["disagreements"] = disagreements
 
    if verbose:
        _print_report(results, folder_a, folder_b)
 
    return results
 
 
def _print_report(results: dict, folder_a: str, folder_b: str):
    name_a = Path(folder_a).name
    name_b = Path(folder_b).name
 
    print(f"\n{'='*80}")
    print(f"ENSEMBLE COMPARISON")
    print(f"  A: {name_a}")
    print(f"  B: {name_b}")
    print(f"  Common samples: {results['n_common']}")
    print(f"  Agreement rate: {results['agreement_rate']:.1%}")
    print(f"{'='*80}\n")
 
    def _fmt(m: dict) -> str:
        return (
            f"Acc={m['accuracy']:.3f}  Prec={m['precision']:.3f}  "
            f"Rec={m['recall']:.3f}  F1={m['f1']:.3f}  FPR={m['fpr']:.3f}  "
            f"(TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']})"
        )
 
    print(f"Model A:          {_fmt(results['model_a'])}")
    print(f"Model B:          {_fmt(results['model_b'])}")
    print(f"Intersection AND: {_fmt(results['intersection_AND'])}")
    print(f"Union OR:         {_fmt(results['union_OR'])}")
 
    # Disagreement breakdown
    n_dis = len(results["disagreements"])
    print(f"\n--- Disagreements: {n_dis} ---")
    if n_dis > 0:
        a_right = sum(1 for d in results["disagreements"] if d["pred_a"] == d["true_label"])
        b_right = sum(1 for d in results["disagreements"] if d["pred_b"] == d["true_label"])
        print(f"  Overall: A correct {a_right}/{n_dis}, B correct {b_right}/{n_dis}")
 
        grade_dis: Dict[int, list] = {}
        for d in results["disagreements"]:
            grade_dis.setdefault(d["grade"], []).append(d)
 
        for g in sorted(grade_dis.keys()):
            ds = grade_dis[g]
            label = "normal" if g <= 3 else "anomalous"
            n_g = len(ds)
 
            a_flags = [d for d in ds if d["pred_a"] == 1 and d["pred_b"] == 0]
            b_flags = [d for d in ds if d["pred_b"] == 1 and d["pred_a"] == 0]
 
            print(f"\n  Grade {g} ({label}): {n_g} disagreements")
 
            if g <= 3:
                print(f"    A flags, B doesn't: {len(a_flags)} (A false-positives that B avoids)")
                for d in a_flags:
                    print(f"      {d['sample_id']}  score_a={d['score_a']:.4f}  score_b={d['score_b']:.4f}")
                print(f"    B flags, A doesn't: {len(b_flags)} (B false-positives that A avoids)")
                for d in b_flags:
                    print(f"      {d['sample_id']}  score_a={d['score_a']:.4f}  score_b={d['score_b']:.4f}")
            else:
                a_catches = [d for d in ds if d["pred_a"] == 1]
                b_catches = [d for d in ds if d["pred_b"] == 1]
                print(f"    A catches, B misses: {len(a_catches)}")
                for d in a_catches:
                    print(f"      {d['sample_id']}  score_a={d['score_a']:.4f}  score_b={d['score_b']:.4f}")
                print(f"    B catches, A misses: {len(b_catches)}")
                for d in b_catches:
                    print(f"      {d['sample_id']}  score_a={d['score_a']:.4f}  score_b={d['score_b']:.4f}")


"""
Score-level ensemble fusion via logistic regression.
 
Takes two prediction folders (same filename format as compare_predictions.py),
extracts continuous anomaly scores, and trains a logistic regression to combine them.
 
Evaluated via stratified k-fold cross-validation — never trains and evaluates on same data.
"""
 

 
CLASSIFIERS = {
    "LogisticRegression": lambda seed: LogisticRegression(random_state=seed, max_iter=1000),
    "SVM_linear": lambda seed: SVC(kernel="linear", probability=True, random_state=seed),
    "SVM_rbf": lambda seed: SVC(kernel="rbf", probability=True, random_state=seed),
    "RandomForest": lambda seed: RandomForestClassifier(
        n_estimators=100, max_depth=4, random_state=seed
    ),
}
 
 
def _build_arrays(folder_a: str, folder_b: str):
    """Parse both folders, align by sample_id, return X (n, 2), y (n,), grades, sample_ids."""
    parsed_a = _parse_folder(folder_a)
    parsed_b = _parse_folder(folder_b)
 
    common_ids = sorted(set(parsed_a.keys()) & set(parsed_b.keys()))
    if len(common_ids) < len(parsed_a) or len(common_ids) < len(parsed_b):
        n_only_a = len(set(parsed_a.keys()) - set(parsed_b.keys()))
        n_only_b = len(set(parsed_b.keys()) - set(parsed_a.keys()))
        print(f"WARNING: {n_only_a} samples only in A, {n_only_b} only in B, {len(common_ids)} common")
 
    scores_a, scores_b, true_labels, grades, sample_ids = [], [], [], [], []
 
    for sid in common_ids:
        true_a, pred_a, score_a = parsed_a[sid]
        true_b, pred_b, score_b = parsed_b[sid]
        if true_a != true_b:
            print(f"ERROR: true label mismatch for {sid}")
            continue
        scores_a.append(score_a)
        scores_b.append(score_b)
        true_labels.append(true_a)
        grades.append(int(re.search(r"grade(\d+)", sid).group(1)))
        sample_ids.append(sid)
 
    X = np.column_stack([scores_a, scores_b])
    y = np.array(true_labels)
    grades = np.array(grades)
    return X, y, grades, sample_ids
 
 
def _run_single_classifier(name, clf_factory, X, y, grades, skf, seed):
    """Run one classifier through all CV folds. Returns aggregate metrics dict."""
    n = len(y)
    oof_preds = np.zeros(n, dtype=int)
    oof_probs = np.zeros(n, dtype=float)
 
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
 
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
 
        clf = clf_factory(seed)
        clf.fit(X_train_s, y_train)
 
        oof_preds[test_idx] = clf.predict(X_test_s)
        oof_probs[test_idx] = clf.predict_proba(X_test_s)[:, 1]
 
    agg = _compute_metrics(y.tolist(), oof_preds.tolist())
    try:
        agg["auroc"] = roc_auc_score(y, oof_probs)
    except ValueError:
        agg["auroc"] = None
 
    # Per-grade
    per_grade = {}
    for g in sorted(set(grades)):
        mask = grades == g
        gm = _compute_metrics(y[mask].tolist(), oof_preds[mask].tolist())
        per_grade[g] = gm
 
    return {
        "aggregate": agg,
        "per_grade": per_grade,
        "oof_preds": oof_preds,
        "oof_probs": oof_probs,
    }
 
 
def score_fusion(folder_a: str, folder_b: str, n_folds: int = 5, seed: int = 42,
                 classifiers: dict = None):
    """
    Compare multiple classifiers for score-level fusion.
    All share the same CV splits for fair comparison.
    """
    if classifiers is None:
        classifiers = CLASSIFIERS
 
    X, y, grades, sample_ids = _build_arrays(folder_a, folder_b)
    n = len(y)
 
    print(f"\n{'='*80}")
    print(f"SCORE FUSION — {n_folds}-fold Stratified CV")
    print(f"  Samples: {n}  (anomalous: {int(y.sum())}, normal: {int((1-y).sum())})")
    print(f"  Score A range: [{X[:, 0].min():.4f}, {X[:, 0].max():.4f}]")
    print(f"  Score B range: [{X[:, 1].min():.4f}, {X[:, 1].max():.4f}]")
    print(f"{'='*80}")
 
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_results = {}
 
    for name, clf_factory in classifiers.items():
        res = _run_single_classifier(name, clf_factory, X, y, grades, skf, seed)
        all_results[name] = res
 
    # Print comparison table
    print(f"\n  {'Classifier':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'FPR':>6} {'AUROC':>6}   TP  TN  FP  FN")
    print(f"  {'-'*100}")
    for name, res in all_results.items():
        a = res["aggregate"]
        auroc_str = f"{a['auroc']:.3f}" if a["auroc"] is not None else "  N/A"
        print(f"  {name:<22} {a['accuracy']:6.3f} {a['precision']:6.3f} {a['recall']:6.3f} "
              f"{a['f1']:6.3f} {a['fpr']:6.3f} {auroc_str}  {a['tp']:3d} {a['tn']:3d} {a['fp']:3d} {a['fn']:3d}")
 
    # Per-grade comparison
    print(f"\n  Per-grade breakdown:")
    for g in sorted(set(grades)):
        label = "normal" if g <= 3 else "anomalous"
        n_g = int((grades == g).sum())
        print(f"\n    Grade {g} ({label}, n={n_g}):")
        for name, res in all_results.items():
            gm = res["per_grade"][g]
            if g <= 3:
                print(f"      {name:<22} FPR={gm['fpr']:.3f} ({gm['fp']} FPs)")
            else:
                print(f"      {name:<22} Recall={gm['recall']:.3f} ({gm['fn']} missed)")
 
    # Logistic regression coefficients (interpretability)
    if "LogisticRegression" in classifiers:
        print(f"\n  LogisticRegression coefficient analysis:")
        # Fit on full data for interpretability (CV already gave unbiased metrics)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        clf = LogisticRegression(random_state=seed, max_iter=1000)
        clf.fit(X_s, y)
        ca, cb = clf.coef_[0]
        ratio = abs(ca) / (abs(ca) + abs(cb)) if (abs(ca) + abs(cb)) > 0 else 0.5
        print(f"    Coefficients (scaled): A={ca:+.3f}  B={cb:+.3f}")
        print(f"    Relative weight: A={ratio:.1%}  B={1-ratio:.1%}")
 
    all_results["_meta"] = {
        "X": X, "y": y, "grades": grades, "sample_ids": sample_ids,
    }
    return all_results
 
 

 
if __name__ == "__main__":
  #  compare_predictions(
  #      folder_a="../../disk/visual_test/patchcore_dinov2_vitb14_data_FULL_NO_FILTERS_SEED_0_256_MAXMEAN_1_test_set_NONE/",
  #      folder_b="../../disk/visual_test/sinbad_dinov2_vitb14_data_FULL_NO_FILTERS_SEED_0_256_MAXMEAN_1_test_set_NONE/",
  #  )
    score_fusion(
        folder_a="../../disk/visual_test/patchcore_dinov2_vitb14_data_FULL_NO_FILTERS_SEED_0_256_MAXMEAN_1_test_set_NONE/",
        folder_b="../../disk/visual_test/sinbad_dinov2_vitb14_data_FULL_NO_FILTERS_SEED_0_256_MAXMEAN_1_test_set_NONE/",
    )