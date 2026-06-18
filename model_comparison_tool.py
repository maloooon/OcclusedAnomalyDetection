# Comparing models (StructCore vs. default scoring)

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import percentileofscore

VISUAL_TEST_ROOT = Path("/mnt/nvme1/marlon_helbing/thesis/visual_test")

# Pattern: {label}_PRED_{pred}_{score}_{img_id}_{obj_id}_{grade}.png
_FNAME_RE = re.compile(
    r"^(?P<label>\d+)_PRED_(?P<pred>\d+)_(?P<score>[\d.]+)"
    r"_(?P<img_id>img\d+)_(?P<obj_id>obj\d+)_(?P<grade>grade\d+)\.png$"
)


def load_threshold(folder_name: str) -> float:
    """Load the optimal threshold saved by test_model for a given visual_test folder."""
    path = VISUAL_TEST_ROOT / folder_name / "threshold.json"
    with open(path) as f:
        return json.load(f)["optimal_threshold"]


def parse_visual_test_folder(folder_name: str) -> pd.DataFrame:
    """Return a DataFrame of all samples in a visual_test folder.

    Columns: label, pred, score, img_id, obj_id, grade, sample_id, percentile
    sample_id = f"{img_id}_{obj_id}", used to join across folders and mark diffuse.
    """
    folder = VISUAL_TEST_ROOT / folder_name
    rows = []
    for p in folder.iterdir():
        m = _FNAME_RE.match(p.name)
        if m is None:
            continue
        rows.append({
            "label":     int(m["label"]),
            "pred":      int(m["pred"]),
            "score":     float(m["score"]),
            "img_id":    m["img_id"],
            "obj_id":    m["obj_id"],
            "grade":     m["grade"],
            "sample_id": f"{m['img_id']}_{m['obj_id']}",
            "filename":  p.name,
        })

    df = pd.DataFrame(rows)
    return df.sort_values("score").reset_index(drop=True)


def compare_runs(
    folder_a: str,
    folder_b: str,
    diffuse_ids: list[str] | None = None,
    label_a: str = "A",
    label_b: str = "B",
    threshold_a: float | None = None,
    threshold_b: float | None = None,
) -> pd.DataFrame:
    """Compare two visual_test folders (any two models or scoring methods).

    pct_delta    = pct_b - pct_a   (positive → folder_b ranks sample higher)
    margin_delta = margin_b - margin_a, where
      margin = (score - threshold) / std(all_scores)   [only if thresholds are given]

    Returns a DataFrame of anomalies only (label=1), columns:
      sample_id, label, grade, score_a, score_b,
      pct_a, pct_b, pct_delta,
      margin_a, margin_b, margin_delta  (only if thresholds provided),
      is_diffuse, label_a, label_b
    """
    df_a = parse_visual_test_folder(folder_a)
    df_b = parse_visual_test_folder(folder_b)

    merged = df_a[["sample_id", "label", "grade", "score", "pred"]].rename(
        columns={"score": "score_a", "pred": "pred_a"}
    ).merge(
        df_b[["sample_id", "score", "pred"]].rename(columns={"score": "score_b", "pred": "pred_b"}),
        on="sample_id",
        how="inner",
    )

    # Percentiles over all samples — tells you where each anomaly sits relative
    # to the full score distribution and the detection boundary.
    def _pct(series: pd.Series) -> pd.Series:
        arr = series.to_numpy()
        return series.apply(lambda s: percentileofscore(arr, s, kind="rank"))

    merged["pct_a"] = _pct(merged["score_a"])
    merged["pct_b"] = _pct(merged["score_b"])

    # Normalised margin: (score - threshold) / std — positive means above the boundary.
    if threshold_a is not None:
        merged["margin_a"] = (merged["score_a"] - threshold_a) / merged["score_a"].std()
    if threshold_b is not None:
        merged["margin_b"] = (merged["score_b"] - threshold_b) / merged["score_b"].std()

    anomalies = merged[merged["label"] == 1].copy()
    anomalies["pct_delta"]  = anomalies["pct_b"] - anomalies["pct_a"]
    if threshold_a is not None and threshold_b is not None:
        anomalies["margin_delta"] = anomalies["margin_b"] - anomalies["margin_a"]
    anomalies["is_diffuse"] = (
        anomalies["sample_id"].isin(diffuse_ids) if diffuse_ids is not None else False
    )
    anomalies["label_a"] = label_a
    anomalies["label_b"] = label_b

    return anomalies.sort_values("pct_delta", ascending=False).reset_index(drop=True)


def summarise_score_shift(
    folder_a: str,
    folder_b: str,
    label_a: str = "A",
    label_b: str = "B",
) -> None:
    """Show how scores shift for normals vs anomalies between two folders.

    This is the complement to the anomaly-only percentile analysis — it reveals
    whether an AUROC gain comes from anomaly scores going up, normal scores going
    down, or both.
    """
    df_a = parse_visual_test_folder(folder_a)
    df_b = parse_visual_test_folder(folder_b)

    merged = df_a[["sample_id", "label", "score"]].rename(columns={"score": "score_a"}).merge(
        df_b[["sample_id", "score"]].rename(columns={"score": "score_b"}),
        on="sample_id", how="inner",
    )
    merged["delta"] = merged["score_b"] - merged["score_a"]

    for lbl, name in [(0, "Normals  (label=0)"), (1, "Anomalies (label=1)")]:
        g = merged[merged["label"] == lbl]
        print(f"{name} (n={len(g)}):")
        print(f"  score_{label_a}: mean={g['score_a'].mean():.4f}  median={g['score_a'].median():.4f}")
        print(f"  score_{label_b}: mean={g['score_b'].mean():.4f}  median={g['score_b'].median():.4f}")
        print(f"  delta (b-a):  mean={g['delta'].mean():+.4f}  median={g['delta'].median():+.4f}")
        print()


def summarise_diffuse_effect(comparison_df: pd.DataFrame) -> None:
    """Print a quick summary of how folder_b shifts diffuse vs. non-diffuse anomalies vs folder_a."""
    label_a = comparison_df["label_a"].iloc[0]
    label_b = comparison_df["label_b"].iloc[0]
    diffuse     = comparison_df[comparison_df["is_diffuse"]]
    non_diffuse = comparison_df[~comparison_df["is_diffuse"]]

    print(f"=== Percentile delta: {label_b} vs {label_a} (positive = {label_b} ranks higher) ===")
    print(f"  Diffuse     (n={len(diffuse):3d}): mean={diffuse['pct_delta'].mean():+.1f}, "
          f"median={diffuse['pct_delta'].median():+.1f}")
    print(f"  Non-diffuse (n={len(non_diffuse):3d}): mean={non_diffuse['pct_delta'].mean():+.1f}, "
          f"median={non_diffuse['pct_delta'].median():+.1f}")
    print(f"\n  Diffuse:     pct_{label_a} mean={diffuse['pct_a'].mean():.1f}, "
          f"pct_{label_b} mean={diffuse['pct_b'].mean():.1f}")
    print(f"  Non-diffuse: pct_{label_a} mean={non_diffuse['pct_a'].mean():.1f}, "
          f"pct_{label_b} mean={non_diffuse['pct_b'].mean():.1f}")

    if "margin_delta" in comparison_df.columns:
        print(f"\n=== Normalised margin delta: {label_b} vs {label_a} (positive = {label_b} further above threshold) ===")
        print(f"  Diffuse     (n={len(diffuse):3d}): mean={diffuse['margin_delta'].mean():+.3f}, "
              f"median={diffuse['margin_delta'].median():+.3f}")
        print(f"  Non-diffuse (n={len(non_diffuse):3d}): mean={non_diffuse['margin_delta'].mean():+.3f}, "
              f"median={non_diffuse['margin_delta'].median():+.3f}")
        print(f"\n  Diffuse:     margin_{label_a} mean={diffuse['margin_a'].mean():+.3f}, "
              f"margin_{label_b} mean={diffuse['margin_b'].mean():+.3f}")
        print(f"  Non-diffuse: margin_{label_a} mean={non_diffuse['margin_a'].mean():+.3f}, "
              f"margin_{label_b} mean={non_diffuse['margin_b'].mean():+.3f}")


def summarise_within_run(
    folder: str,
    diffuse_ids: list[str],
    label: str | None = None,
) -> pd.DataFrame:
    """Within a single folder, compare anomaly scores of diffuse vs non-diffuse raspberries.

    Hypothesis: if the model scores whole raspberries, diffuse anomalies may already
    have a higher mean score than non-diffuse ones — i.e. the model is sensitive to
    the overall raspberry quality, not just local defect patches.

    Returns the anomaly DataFrame with an is_diffuse column for further inspection.
    """
    name = label or folder.split("_")[0]
    df = parse_visual_test_folder(folder)
    anomalies = df[df["label"] == 1].copy()
    anomalies["is_diffuse"] = anomalies["sample_id"].isin(diffuse_ids)

    diffuse     = anomalies[anomalies["is_diffuse"]]
    non_diffuse = anomalies[~anomalies["is_diffuse"]]

    print(f"=== Within-run score comparison: {name} (anomalies only) ===")
    print(f"  Diffuse     (n={len(diffuse):3d}): mean={diffuse['score'].mean():.4f}  "
          f"median={diffuse['score'].median():.4f}  "
          f"std={diffuse['score'].std():.4f}")
    print(f"  Non-diffuse (n={len(non_diffuse):3d}): mean={non_diffuse['score'].mean():.4f}  "
          f"median={non_diffuse['score'].median():.4f}  "
          f"std={non_diffuse['score'].std():.4f}")
    print(f"  Diffuse mean {'HIGHER' if diffuse['score'].mean() > non_diffuse['score'].mean() else 'LOWER'} "
          f"by {abs(diffuse['score'].mean() - non_diffuse['score'].mean()):.4f}")

    return anomalies.sort_values("score", ascending=False).reset_index(drop=True)


def diffuse_recall_ratio(
    folder: str,
    diffuse_ids: list[str],
    threshold: float | None = None,
    label: str | None = None,
) -> dict:
    """At a model's optimal threshold, compute diffuse and non-diffuse recall separately.

    The ratio diffuse_recall / non_diffuse_recall is comparable across models with
    different overall performance — a higher ratio means the model is relatively less
    biased against diffuse cases.

    threshold: pass explicitly or leave None to load from threshold.json.
    Returns a dict so results can be collected and compared across models.
    """
    name = label or folder.split("_")[0]
    thr  = threshold if threshold is not None else load_threshold(folder)

    df       = parse_visual_test_folder(folder)
    anomalies = df[df["label"] == 1].copy()
    anomalies["is_diffuse"] = anomalies["sample_id"].isin(diffuse_ids)
    anomalies["detected"]   = anomalies["score"] >= thr

    diffuse     = anomalies[anomalies["is_diffuse"]]
    non_diffuse = anomalies[~anomalies["is_diffuse"]]

    rec_diffuse     = diffuse["detected"].mean() #tp/tp+fn
    rec_non_diffuse = non_diffuse["detected"].mean()
    ratio           = rec_diffuse / rec_non_diffuse if rec_non_diffuse > 0 else float("nan")

    print(f"=== Diffuse recall ratio: {name} (threshold={thr:.4f}) ===")
    print(f"  Diffuse     (n={len(diffuse):3d}): recall={rec_diffuse:.3f}  "
          f"({diffuse['detected'].sum()}/{len(diffuse)} detected)")
    print(f"  Non-diffuse (n={len(non_diffuse):3d}): recall={rec_non_diffuse:.3f}  "
          f"({non_diffuse['detected'].sum()}/{len(non_diffuse)} detected)")
    print(f"  Ratio diffuse/non-diffuse: {ratio:.3f}  "
          f"({'relatively better' if ratio > 0.5 else 'relatively worse'} on diffuse)")

    return {
        "model":            name,
        "threshold":        thr,
        "n_diffuse":        len(diffuse),
        "n_non_diffuse":    len(non_diffuse),
        "rec_diffuse":      rec_diffuse,
        "rec_non_diffuse":  rec_non_diffuse,
        "ratio":            ratio,
    }


def compare_diffuse_recall_ratios(results: list[dict]) -> None:
    """Print a sorted comparison table from multiple diffuse_recall_ratio() calls."""
    df = pd.DataFrame(results).sort_values("ratio", ascending=False)
    print("=== Diffuse recall ratio comparison (higher ratio = less biased against diffuse) ===")
    print(df[["model", "threshold", "rec_diffuse", "rec_non_diffuse", "ratio"]].to_string(index=False))


def diffuse_detection_at_fpr(
    folder: str,
    diffuse_ids: list[str],
    target_fpr: float = 0.10,
    label: str | None = None,
) -> dict:
    """At a fixed false positive rate, count how many diffuse anomalies each model catches.

    The threshold is derived per-model as the score where exactly target_fpr fraction
    of normals are flagged. This makes counts directly comparable across models —
    same FPR operating point, same total diffuse raspberries, so more caught = better.
    """
    name = label or folder.split("_")[0]
    df   = parse_visual_test_folder(folder)

    normals   = df[df["label"] == 0]["score"]
    anomalies = df[df["label"] == 1].copy()
    anomalies["is_diffuse"] = anomalies["sample_id"].isin(diffuse_ids)

    # threshold = score above which target_fpr fraction of normals fall
    thr = float(np.quantile(normals, 1 - target_fpr))
    anomalies["detected"] = anomalies["score"] >= thr

    diffuse     = anomalies[anomalies["is_diffuse"]]
    non_diffuse = anomalies[~anomalies["is_diffuse"]]

    n_diffuse_caught     = int(diffuse["detected"].sum())
    n_non_diffuse_caught = int(non_diffuse["detected"].sum())
    ratio = n_diffuse_caught / n_non_diffuse_caught if n_non_diffuse_caught > 0 else float("nan")

    print(f"=== {name} at FPR={target_fpr:.2f} (threshold={thr:.4f}) ===")
    print(f"  Diffuse     caught: {n_diffuse_caught:3d} / {len(diffuse)}  "
          f"({100*n_diffuse_caught/len(diffuse):.1f}%)")
    print(f"  Non-diffuse caught: {n_non_diffuse_caught:3d} / {len(non_diffuse)}  "
          f"({100*n_non_diffuse_caught/len(non_diffuse):.1f}%)")
    print(f"  Ratio diffuse/non-diffuse: {ratio:.3f}")

    return {
        "model":                name,
        "fpr":                  target_fpr,
        "threshold":            thr,
        "n_diffuse":            len(diffuse),
        "n_diffuse_caught":     n_diffuse_caught,
        "n_non_diffuse":        len(non_diffuse),
        "n_non_diffuse_caught": n_non_diffuse_caught,
        "pct_diffuse_caught":   n_diffuse_caught / len(diffuse),
        "pct_non_diffuse_caught": n_non_diffuse_caught / len(non_diffuse),
        "ratio":                ratio,
    }


def compare_diffuse_detection_at_fpr(results: list[dict]) -> None:
    """Print a sorted comparison table from multiple diffuse_detection_at_fpr() calls."""
    df = pd.DataFrame(results).sort_values("n_diffuse_caught", ascending=False)
    print(f"=== Diffuse detection at FPR={results[0]['fpr']:.2f} "
          f"(n_diffuse={results[0]['n_diffuse']}, n_non_diffuse={results[0]['n_non_diffuse']}) ===")
    print(df[["model", "threshold", "n_diffuse_caught", "pct_diffuse_caught",
              "n_non_diffuse_caught", "pct_non_diffuse_caught", "ratio"]].to_string(index=False))


def summarise_score_delta(comparison_df: pd.DataFrame) -> None:
    """Compare raw score delta (score_b - score_a) for diffuse vs non-diffuse anomalies.

    The delta for normals (from summarise_score_shift) is the baseline — anomalies
    with a delta above that are being selectively boosted by folder_b.
    """
    label_a = comparison_df["label_a"].iloc[0]
    label_b = comparison_df["label_b"].iloc[0]

    comparison_df = comparison_df.copy()
    comparison_df["score_delta"] = comparison_df["score_b"] - comparison_df["score_a"]

    diffuse     = comparison_df[comparison_df["is_diffuse"]]
    non_diffuse = comparison_df[~comparison_df["is_diffuse"]]

    print(f"=== Raw score delta ({label_b} - {label_a}) for anomalies ===")
    print(f"  Diffuse     (n={len(diffuse):3d}): mean={diffuse['score_delta'].mean():+.4f}  "
          f"median={diffuse['score_delta'].median():+.4f}")
    print(f"  Non-diffuse (n={len(non_diffuse):3d}): mean={non_diffuse['score_delta'].mean():+.4f}  "
          f"median={non_diffuse['score_delta'].median():+.4f}")
    print(f"  (Compare against normal delta from summarise_score_shift to see who benefits more)")


def compare_detection_correctness(
    comparison_df: pd.DataFrame,
    diffuse_only: bool = True,
) -> pd.DataFrame:
    """Show which raspberries are correctly detected (pred==1) by each run.

    For anomalies (label=1), correct detection means pred==1.
    Prints a 2x2 breakdown and returns a per-sample DataFrame with columns:
      sample_id, grade, is_diffuse, correct_a, correct_b, status
    where status is one of: both | a_only | b_only | neither
    """
    label_a = comparison_df["label_a"].iloc[0]
    label_b = comparison_df["label_b"].iloc[0]

    df = comparison_df[comparison_df["is_diffuse"]].copy() if diffuse_only else comparison_df.copy()
    tag = "diffuse" if diffuse_only else "all anomalies"

    df["correct_a"] = df["pred_a"] == 1
    df["correct_b"] = df["pred_b"] == 1

    def _status(row):
        if row["correct_a"] and row["correct_b"]:   return "both"
        if row["correct_a"] and not row["correct_b"]: return "a_only"
        if row["correct_b"] and not row["correct_a"]: return "b_only"
        return "neither"

    df["status"] = df.apply(_status, axis=1)

    counts = df["status"].value_counts().reindex(["both", "a_only", "b_only", "neither"], fill_value=0)
    n = len(df)

    print(f"=== Detection correctness ({tag}, n={n}) ===")
    print(f"  Detected by both         : {counts['both']:3d} ({100*counts['both']/n:.1f}%)")
    print(f"  {label_a} only : {counts['a_only']:3d} ({100*counts['a_only']/n:.1f}%)")
    print(f"  {label_b} only : {counts['b_only']:3d} ({100*counts['b_only']/n:.1f}%)")
    print(f"  Missed by both           : {counts['neither']:3d} ({100*counts['neither']/n:.1f}%)")
    print(f"\n  Total correct — {label_a}: {counts['both']+counts['a_only']:3d}/{n}  "
          f"{label_b}: {counts['both']+counts['b_only']:3d}/{n}")

    return df[["sample_id", "grade", "is_diffuse", "correct_a", "correct_b", "status"]].sort_values("status")


def create_visual_comparison(
    comparison_df: pd.DataFrame,
    folder_a: str,
    folder_b: str,
    out_folder: str | None = None,
    diffuse_only: bool = False,
) -> Path:
    """For each anomaly in comparison_df, create a stacked image:
      [info bar] / [folder_a composite] / [folder_b composite]
    and save it to out_folder inside VISUAL_TEST_ROOT.

    diffuse_only: if True, only render diffuse raspberries.
    Returns the output folder path.
    """
    label_a = comparison_df["label_a"].iloc[0]
    label_b = comparison_df["label_b"].iloc[0]
    dir_a   = VISUAL_TEST_ROOT / folder_a
    dir_b   = VISUAL_TEST_ROOT / folder_b

    if out_folder is None:
        out_folder = f"comparison_{label_a}_vs_{label_b}"
    out_dir = VISUAL_TEST_ROOT / out_folder
    out_dir.mkdir(exist_ok=True)

    def _index(folder: Path) -> dict[str, Path]:
        idx = {}
        for p in folder.iterdir():
            m = _FNAME_RE.match(p.name)
            if m:
                idx[f"{m['img_id']}_{m['obj_id']}"] = p
        return idx

    idx_a = _index(dir_a)
    idx_b = _index(dir_b)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    df = comparison_df[comparison_df["is_diffuse"]] if diffuse_only else comparison_df

    for _, row in df.iterrows():
        sid = row["sample_id"]
        if sid not in idx_a or sid not in idx_b:
            print(f"  skipping {sid}: not found in both folders")
            continue

        img_a = Image.open(idx_a[sid]).convert("RGBA")
        img_b = Image.open(idx_b[sid]).convert("RGBA")
        W, H  = img_a.size

        BAR    = 60
        canvas = Image.new("RGBA", (W, H * 2 + BAR), (30, 30, 30, 255))
        draw   = ImageDraw.Draw(canvas)

        diffuse_tag = "DIFFUSE" if row["is_diffuse"] else "non-diffuse"
        hdr_color   = (255, 200, 50, 255) if row["is_diffuse"] else (180, 180, 180, 255)
        line1 = f"{sid}  |  {row['grade']}  |  {diffuse_tag}"
        line2 = (
            f"{label_a}: score={row['score_a']:.4f}  pct={row['pct_a']:.1f}"
            f"      {label_b}: score={row['score_b']:.4f}  pct={row['pct_b']:.1f}"
            f"      Δpct={row['pct_delta']:+.1f}"
        )
        draw.text((10, 6),  line1, font=font, fill=hdr_color)
        draw.text((10, 32), line2, font=font, fill=(220, 220, 220, 255))

        canvas.paste(img_a, (0, BAR))
        canvas.paste(img_b, (0, BAR + H))

        draw.text((6, BAR + 4),     label_a, font=font, fill=(100, 180, 255, 255))
        draw.text((6, BAR + H + 4), label_b, font=font, fill=(100, 255, 140, 255))

        fname = f"{'DIFFUSE_' if row['is_diffuse'] else ''}{sid}_{row['grade']}_delta{row['pct_delta']:+.1f}.png"
        canvas.save(out_dir / fname)

    print(f"Saved {len(df)} images to {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Example usage (uncomment to run)
# ---------------------------------------------------------------------------


diffuse = [
     "img001_obj15", "img001_obj19", "img006_obj0", "img006_obj10", "img007_obj5",
     "img008_obj16", "img009_obj11", "img011_obj17", "img011_obj27", "img012_obj0",
     "img014_obj24", "img015_obj9", "img016_obj26", "img018_obj12", "img018_obj13",
     "img018_obj16", "img019_obj13", "img021_obj11", "img022_obj3", "img024_obj21",
     "img024_obj4", "img025_obj0", "img025_obj11", "img028_obj13", "img028_obj19",
     "img028_obj20", "img029_obj13", "img029_obj5", "img030_obj10", "img033_obj11",
     "img035_obj0", "img036_obj14", "img036_obj2", "img036_obj6", "img037_obj1",
     "img038_obj15", "img040_obj22", "img040_obj23", "img044_obj11", "img045_obj21",
     "img047_obj22", "img050_obj12", "img051_obj8", "img053_obj15", "img054_obj1",
     "img054_obj10", "img062_obj6", "img062_obj7", "img064_obj2", "img065_obj12",
     "img066_obj20", "img067_obj20", "img068_obj13", "img070_obj18", "img070_obj22",
     "img070_obj24", "img071_obj8", "img073_obj8", "img074_obj17", "img075_obj16",
     "img076_obj8", "img079_obj18", "img082_obj12", "img084_obj20", "img085_obj7",
     "img086_obj10", "img090_obj24", "img099_obj20", "img105_obj4", "img107_obj1",
     "img108_obj13", "img126_obj6", "img129_obj16", "img130_obj0", "img134_obj23",
     "img139_obj14", "img140_obj19", "img149_obj10", "img149_obj7", "img150_obj1",
     "img150_obj10", "img150_obj8", "img150_obj9", "img152_obj0", "img152_obj3",
     "img154_obj11", "img154_obj2", "img158_obj19", "img159_obj6", "img160_obj4",
     "img161_obj21", "img163_obj0", "img163_obj14", "img167_obj23", "img170_obj24",
     "img171_obj10", "img176_obj3", "img177_obj1", "img182_obj11", "img184_obj12",
     "img186_obj15", "img187_obj9", "img188_obj16", "img194_obj19", "img196_obj15",
     "img196_obj16", "img196_obj18", "img196_obj5", "img200_obj8",
 ] 

'''
diffuse = [
    "img001_obj15", "img006_obj0", "img006_obj10", "img007_obj5",
    "img008_obj16", "img009_obj11", "img011_obj17", "img011_obj27", "img012_obj0",
    "img014_obj24", "img015_obj9", "img018_obj13",
    "img018_obj16", "img019_obj13", "img021_obj11", "img022_obj3", "img024_obj21",
    "img024_obj4", "img025_obj0", "img025_obj11", "img028_obj19",
    "img028_obj20", "img029_obj13", "img029_obj5", "img030_obj10", "img033_obj11", "img036_obj14", "img036_obj2", "img036_obj6", "img037_obj1",
    "img038_obj15", "img040_obj22", "img040_obj23", "img044_obj11", "img045_obj21",
    "img047_obj22", "img050_obj12", "img051_obj8", "img053_obj15", "img054_obj1",
    "img054_obj10", "img062_obj6", "img065_obj12",
    "img066_obj20", "img067_obj20", "img068_obj13",
    "img070_obj24", "img071_obj8", "img073_obj8", "img074_obj17", "img075_obj16",
    "img076_obj8", "img082_obj12", "img084_obj20", "img085_obj7", "img090_obj24", "img105_obj4", "img107_obj1",
    "img108_obj13", "img126_obj6", "img129_obj16", "img130_obj0",
    "img139_obj14", "img140_obj19", "img149_obj10", "img149_obj7", "img150_obj1",
    "img150_obj10", "img150_obj8", "img150_obj9", "img152_obj0", "img152_obj3",
    "img154_obj11", "img154_obj2", "img158_obj19", "img159_obj6", "img160_obj4",
    "img161_obj21", "img163_obj14", "img167_obj23", "img170_obj24",
    "img171_obj10", "img176_obj3", "img177_obj1", "img182_obj11", "img184_obj12", "img187_obj9", "img188_obj16", "img194_obj19",
    "img196_obj16", "img196_obj18", "img200_obj8"
]  
'''

non_diffuse = ["img020_obj0", "img012_obj26", "img164_obj20", "img020_obj26", "img054_obj9", "img191_obj25", "img003_obj16", "img048_obj26", "img122_obj11", 
"img056_obj15", "img063_obj11", "img046_obj14", "img001_obj17", "img110_obj21", "img037_obj15", "img117_obj4", "img075_obj6", "img166_obj18", "img030_obj5", "img028_obj16", 
"img116_obj12", "img141_obj23", "img126_obj8",  "img161_obj11", "img021_obj2", "img091_obj4", "img016_obj4", "img084_obj0", "img004_obj1", "img179_obj16", "img005_obj3",
 "img074_obj10", "img185_obj1", "img059_obj21", "img124_obj6", "img196_obj10", "img195_obj13", "img022_obj15", "img076_obj20", "img046_obj18", "img126_obj22", "img029_obj12",
  "img076_obj11", "img037_obj13", "img006_obj14", "img186_obj22", "img045_obj1"]



df = compare_runs(
folder_a    = "patchcore_dinov2_vitb14_5_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_0_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug",
folder_b    = "patchcore_dinov2_vitb14_5_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_0_YOLO_640_SHARED_TEST_SET_256_STRUCTCORE_test_set_NONE_no_aug",
diffuse_ids = diffuse,
label_a     = "PATCHCORE_MAXMEAN",
label_b     = "PATCHCORE_STRUCTCORE",
)



#summarise_score_shift(folder_a="patchcore_dinov2_vitb14_5_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_42_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug",
#                      folder_b = "patchcore_dinov2_vitb14_5_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_42_YOLO_640_SHARED_TEST_SET_256_STRUCTCORE_test_set_NONE_no_aug", 
#                      label_a="MAXMEAN", label_b="STRUCTCORE")

#summarise_score_delta(df)







df = compare_runs(
    folder_a    = "stfpm_wide_resnet50_2_layer2_layer3_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_0_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_augment",
    folder_b    = "stfpm_wide_resnet50_2_layer2_layer3_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_0_YOLO_640_SHARED_TEST_SET_256_STRUCTCORE_test_set_NONE_augment",
    diffuse_ids = diffuse,
    label_a     = "STFPM_MAXMEAN",
    label_b     = "STFPM_STRUCTCORE",
)

#summarise_score_shift(folder_a="stfpm_wide_resnet50_2_layer2_layer3_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_42_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_augment",
 #                     folder_b = "stfpm_wide_resnet50_2_layer2_layer3_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_42_YOLO_640_SHARED_TEST_SET_256_STRUCTCORE_test_set_NONE_augment", 
#                      label_a="MAXMEAN", label_b="STRUCTCORE")

#summarise_score_delta(df)


#summarise_score_shift(folder_a="stfpm_wide_resnet50_2_layer2_layer3_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_1_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_augment",
#                      folder_b = "stfpm_wide_resnet50_2_layer2_layer3_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_1_YOLO_640_SHARED_TEST_SET_256_STRUCTCORE_test_set_NONE_augment", 
#                      label_a="MAXMEAN", label_b="STRUCTCORE")




#summarise_within_run(
#    folder      = "sinbad_mobilenet_v2_features.13_features.17_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_0_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_replace",
#    diffuse_ids = diffuse,
#    label       = "SINBAD",
#)


#    diffuse_recall_ratio(
#        folder   = "patchcore_dinov2_vitb14_5_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_1_YOLO_640_SHARED_TEST_SET_256_STRUCTCORE_test_set_NONE_no_aug",
#        diffuse_ids = diffuse,
#        label    = "PATCHCORE_STRUCTCORE",
#    ),

results = [

    diffuse_recall_ratio(
        folder   = "stfpm_wide_resnet50_2_layer2_layer3_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_42_YOLO_640_SHARED_TEST_SET_256_STRUCTCORE_test_set_NONE_augment",
        diffuse_ids = diffuse,
        label    = "STFPM_STRUCTCORE_SEED_42",

   ),
    diffuse_recall_ratio(
        folder   = "stfpm_wide_resnet50_2_layer2_layer3_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_0_YOLO_640_SHARED_TEST_SET_256_STRUCTCORE_test_set_NONE_augment",
        diffuse_ids = diffuse,
        label    = "STFPM_STRUCTCORE_SEED_0",
    ),
    diffuse_recall_ratio(
        folder   = "stfpm_wide_resnet50_2_layer2_layer3_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_1_YOLO_640_SHARED_TEST_SET_256_STRUCTCORE_test_set_NONE_augment",
        diffuse_ids = diffuse,
        label    = "STFPM_STRUCTCORE_SEED_1",
    ),



  #  diffuse_recall_ratio(
  #      folder   = "stfpm_wide_resnet50_2_layer2_layer3_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_42_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_augment",
  #      diffuse_ids = diffuse,
  #      label    = "STFPM_MAXMEAN_SEED_42",
  #  ),
  #  diffuse_recall_ratio(
  #      folder   = "stfpm_wide_resnet50_2_layer2_layer3_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_0_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_augment",
  #      diffuse_ids = diffuse,
  #      label    = "STFPM_MAXMEAN_SEED_0",
  #  ),
  #  diffuse_recall_ratio(
  #      folder   = "stfpm_wide_resnet50_2_layer2_layer3_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_1_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_augment",
  #      diffuse_ids = diffuse,
  #      label    = "STFPM_MAXMEAN_SEED_1",
  #  ),



    diffuse_recall_ratio(
        folder   = "patchcore_dinov2_vitb14_5_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_42_YOLO_640_SHARED_TEST_SET_256_STRUCTCORE_test_set_NONE_no_aug",
        diffuse_ids = diffuse,
        label    = "PATCHCORE_STRUCTCORE_SEED_42",
    ),
    diffuse_recall_ratio(
        folder   = "patchcore_dinov2_vitb14_5_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_0_YOLO_640_SHARED_TEST_SET_256_STRUCTCORE_test_set_NONE_no_aug",
        diffuse_ids = diffuse,
        label    = "PATCHCORE_STRUCTCORE_SEED_0",
    ),
    diffuse_recall_ratio(
        folder   = "patchcore_dinov2_vitb14_5_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_1_YOLO_640_SHARED_TEST_SET_256_STRUCTCORE_test_set_NONE_no_aug",
        diffuse_ids = diffuse,
        label    = "PATCHCORE_STRUCTCORE_SEED_1",
    ),
  #  diffuse_recall_ratio(
  #      folder   = "patchcore_dinov2_vitb14_5_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_42_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug",
  #      diffuse_ids = diffuse,
  #      label    = "PATCHCORE_MAXMEAN_SEED_42",
  #  ),
  #  diffuse_recall_ratio(
  #      folder   = "patchcore_dinov2_vitb14_5_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_0_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug",
  #      diffuse_ids = diffuse,
  #      label    = "PATCHCORE_MAXMEAN_SEED_0",
  #  ),
  #  diffuse_recall_ratio(
  #      folder   = "patchcore_dinov2_vitb14_5_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_1_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug",
  #      diffuse_ids = diffuse,
  #      label    = "PATCHCORE_MAXMEAN_SEED_1",
  #  ),

    diffuse_recall_ratio(
        folder   = "sinbad_mobilenet_v2_features.13_features.17_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_42_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_replace",
        diffuse_ids = diffuse,
        label    = "SINBAD_MAXMEAN_SEED_42",
    ),
    diffuse_recall_ratio(
        folder   = "sinbad_mobilenet_v2_features.13_features.17_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_0_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_replace",
        diffuse_ids = diffuse,
        label    = "SINBAD_MAXMEAN_SEED_0",
    ),
    diffuse_recall_ratio(
        folder   = "sinbad_mobilenet_v2_features.13_features.17_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_1_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_replace",
        diffuse_ids = diffuse,
        label    = "SINBAD_MAXMEAN_SEED_1",
    ),


]

compare_diffuse_recall_ratios(results)




results = [

    diffuse_detection_at_fpr(
        folder   = "stfpm_wide_resnet50_2_layer2_layer3_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_42_YOLO_640_SHARED_TEST_SET_256_STRUCTCORE_test_set_NONE_augment",
        diffuse_ids = diffuse,
        label    = "STFPM_STRUCTCORE_SEED_42",

    ),
    diffuse_detection_at_fpr(
        folder   = "stfpm_wide_resnet50_2_layer2_layer3_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_0_YOLO_640_SHARED_TEST_SET_256_STRUCTCORE_test_set_NONE_augment",
        diffuse_ids = diffuse,
        label    = "STFPM_STRUCTCORE_SEED_0",
    ),
    diffuse_detection_at_fpr(
        folder   = "stfpm_wide_resnet50_2_layer2_layer3_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_1_YOLO_640_SHARED_TEST_SET_256_STRUCTCORE_test_set_NONE_augment",
        diffuse_ids = diffuse,
        label    = "STFPM_STRUCTCORE_SEED_1",
    ),



    diffuse_detection_at_fpr(
        folder   = "stfpm_wide_resnet50_2_layer2_layer3_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_42_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_augment",
        diffuse_ids = diffuse,
        label    = "STFPM_MAXMEAN_SEED_42",
    ),
    diffuse_detection_at_fpr(
        folder   = "stfpm_wide_resnet50_2_layer2_layer3_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_0_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_augment",
        diffuse_ids = diffuse,
        label    = "STFPM_MAXMEAN_SEED_0",
    ),
    diffuse_detection_at_fpr(
        folder   = "stfpm_wide_resnet50_2_layer2_layer3_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_1_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_augment",
        diffuse_ids = diffuse,
        label    = "STFPM_MAXMEAN_SEED_1",
    ),



  #  diffuse_detection_at_fpr(
  #      folder   = "patchcore_dinov2_vitb14_5_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_42_YOLO_640_SHARED_TEST_SET_256_STRUCTCORE_test_set_NONE_no_aug",
  #      diffuse_ids = diffuse,
  #      label    = "PATCHCORE_STRUCTCORE_SEED_42",
  #  ),
  #  diffuse_detection_at_fpr(
  #      folder   = "patchcore_dinov2_vitb14_5_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_0_YOLO_640_SHARED_TEST_SET_256_STRUCTCORE_test_set_NONE_no_aug",
  #      diffuse_ids = diffuse,
  #      label    = "PATCHCORE_STRUCTCORE_SEED_0",
  #  ),
  #  diffuse_detection_at_fpr(
  #      folder   = "patchcore_dinov2_vitb14_5_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_1_YOLO_640_SHARED_TEST_SET_256_STRUCTCORE_test_set_NONE_no_aug",
   #     diffuse_ids = diffuse,
   #     label    = "PATCHCORE_STRUCTCORE_SEED_1",
   # ),
   # diffuse_detection_at_fpr(
   #     folder   = "patchcore_dinov2_vitb14_5_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_42_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug",
   #     diffuse_ids = diffuse,
   #     label    = "PATCHCORE_MAXMEAN_SEED_42",
   # ),
   # diffuse_detection_at_fpr(
   #     folder   = "patchcore_dinov2_vitb14_5_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_0_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug",
   #     diffuse_ids = diffuse,
   #     label    = "PATCHCORE_MAXMEAN_SEED_0",
   # ),
   # diffuse_detection_at_fpr(
   #     folder   = "patchcore_dinov2_vitb14_5_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_1_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug",
   #     diffuse_ids = diffuse,
   #     label    = "PATCHCORE_MAXMEAN_SEED_1",
   # ),

  #  diffuse_recall_ratio(
  #      folder   = "patchcore_dinov2_vitb14_5_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_1_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug",
  #      diffuse_ids = diffuse,
  #      label    = "PATCHCORE_MAXMEAN_SEED_1",
  #  ),

  #  diffuse_detection_at_fpr(
  #      folder   = "sinbad_mobilenet_v2_features.13_features.17_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_42_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_replace",
  #      diffuse_ids = diffuse,
  #      label    = "SINBAD_MAXMEAN_SEED_42",
  #  ),
  #  diffuse_detection_at_fpr(
  #      folder   = "sinbad_mobilenet_v2_features.13_features.17_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_0_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_replace",
  #      diffuse_ids = diffuse,
  #      label    = "SINBAD_MAXMEAN_SEED_0",
  #  ),
  #  diffuse_detection_at_fpr(
  #      folder   = "sinbad_mobilenet_v2_features.13_features.17_data_FILTERED_UNBLURRED_AND_CLEAN_PROTRUSIONS_SEED_1_YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_replace",
  #      diffuse_ids = diffuse,
  #      label    = "SINBAD_MAXMEAN_SEED_1",
 #   ),


]
#compare_diffuse_detection_at_fpr(results)






#summarise_diffuse_effect(df)
#print(df[df["is_diffuse"]][["sample_id", "pct_a", "pct_b", "pct_delta"]])
#compare_detection_correctness(df, diffuse_only=True)


# create_visual_comparison(
#     df,
#     folder_a     = "...",
#     folder_b     = "...",
#     diffuse_only = True,
# )
