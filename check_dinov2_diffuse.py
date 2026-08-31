#!/usr/bin/env python3

# DONT PUSH THIS
"""
check_dinov2_diffuse.py

Reads anomaly scores from visual_test result folders for:
  - PatchCore + DINOv2 (vitb14)
  - PatchCore + WideResNet-50-2

and computes AUROC broken down by diffuse vs non-diffuse anomalies across
seeds 0, 1, 42, mirroring the StructCore comparison done in anomaly_detection.py.

Filenames in each folder are expected to follow the format produced by
test_model() / visual_test_path:
    {label}_PRED_{pred}_{score}_img{X}_obj{Y}_grade{Z}.png

Usage:
    python check_dinov2_diffuse.py \\
        --visual_test_dir /home/marlon_helbing/nvme1/thesis/visual_test \\
        --dinov2_template "patchcore_dinov2_vitb14_5_data_FULL_NO_FILTERS_SEED_{SEED}_..." \\
        --wrn_template "patchcore_wide_resnet50_2_data_FULL_NO_FILTERS_SEED_{SEED}_..."

The {SEED} placeholder is replaced with 0, 1, and 42 automatically.
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


_DIFFUSE_IDS = frozenset([
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
])


def _auroc(pos, neg):
    """AUROC via Wilcoxon rank-sum — P(score_pos > score_neg)."""
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    all_s  = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    ranks  = np.argsort(np.argsort(all_s)) + 1
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def parse_folder(folder: Path):
    """
    Parse all result image filenames in folder.

    Filename format:
        {label}_PRED_{pred}_{score}_img{X}_obj{Y}_grade{Z}.png

    Returns a dict with arrays:
        'scores_normal'    : scores for normal samples (label=0)
        'scores_diffuse'   : scores for diffuse anomalous samples
        'scores_nondiffuse': scores for non-diffuse anomalous samples
    """
    normal, diffuse, nondiffuse = [], [], []

    for fname in os.listdir(folder):
        if not fname.endswith('.png'):
            continue
        if not (fname.startswith('0_') or fname.startswith('1_')):
            continue

        parts = fname.replace('.png', '').split('_')
        # Expected: ['0','PRED','0','0.5348','img094','obj0','grade2']
        if len(parts) < 7 or parts[1] != 'PRED':
            continue

        try:
            label = int(parts[0])
            score = float(parts[3])
            sample_id = f"{parts[4]}_{parts[5]}"
        except (ValueError, IndexError):
            continue

        if label == 0:
            normal.append(score)
        else:
            if sample_id in _DIFFUSE_IDS:
                diffuse.append(score)
            else:
                nondiffuse.append(score)

    return {
        'scores_normal':     np.array(normal),
        'scores_diffuse':    np.array(diffuse),
        'scores_nondiffuse': np.array(nondiffuse),
    }


def auroc_subsets(data):
    """Compute AUROC for all / diffuse / non-diffuse anomalies vs normals."""
    neg = data['scores_normal']
    diff = data['scores_diffuse']
    ndiff = data['scores_nondiffuse']
    all_anom = np.concatenate([diff, ndiff])
    return {
        'all':      _auroc(all_anom, neg),
        'diffuse':  _auroc(diff,     neg),
        'nondiff':  _auroc(ndiff,    neg),
    }


def load_seeds(visual_test_dir: Path, template: str, seeds=(0, 1, 42)):
    """
    For each seed, resolve the template (replace {SEED}), parse the folder,
    and return per-seed data dicts plus aggregated arrays across all seeds.
    """
    per_seed = {}
    agg = {'scores_normal': [], 'scores_diffuse': [], 'scores_nondiffuse': []}

    for seed in seeds:
        dirname = template.replace('{SEED}', str(seed))
        folder = visual_test_dir / dirname
        if not folder.exists():
            print(f"  WARNING: folder not found: {folder}")
            per_seed[seed] = None
            continue
        data = parse_folder(folder)
        per_seed[seed] = data
        for k in agg:
            agg[k].append(data[k])

    for k in agg:
        agg[k] = np.concatenate(agg[k]) if agg[k] else np.array([])

    return per_seed, agg


def print_auroc_table(label, per_seed, seeds=(0, 1, 42)):
    """Print per-seed and mean±std AUROC for all / diffuse / non-diffuse."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  {'Seed':>6}  {'All anoms':>10}  {'Diffuse':>10}  {'Non-diff':>10}  {'n_norm':>7}  {'n_diff':>7}  {'n_ndiff':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*8}")

    aurocs_all, aurocs_diff, aurocs_ndiff = [], [], []

    for seed in seeds:
        data = per_seed.get(seed)
        if data is None:
            print(f"  {seed:>6}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}")
            continue
        a = auroc_subsets(data)
        aurocs_all.append(a['all'])
        aurocs_diff.append(a['diffuse'])
        aurocs_ndiff.append(a['nondiff'])
        n_norm  = len(data['scores_normal'])
        n_diff  = len(data['scores_diffuse'])
        n_ndiff = len(data['scores_nondiffuse'])
        print(f"  {seed:>6}  {a['all']:>10.3f}  {a['diffuse']:>10.3f}  {a['nondiff']:>10.3f}  {n_norm:>7}  {n_diff:>7}  {n_ndiff:>8}")

    if aurocs_all:
        mu_all   = float(np.mean(aurocs_all))
        std_all  = float(np.std(aurocs_all))
        mu_diff  = float(np.mean(aurocs_diff))
        std_diff = float(np.std(aurocs_diff))
        mu_nd    = float(np.mean(aurocs_ndiff))
        std_nd   = float(np.std(aurocs_ndiff))
        print(f"  {'mean':>6}  {mu_all:>10.3f}  {mu_diff:>10.3f}  {mu_nd:>10.3f}")
        print(f"  {'±std':>6}  {std_all:>10.3f}  {std_diff:>10.3f}  {std_nd:>10.3f}")

    return aurocs_all, aurocs_diff, aurocs_ndiff


def plot_kde_overlay(models, output_path: Path, title_suffix=''):
    """
    N-panel KDE plot, one panel per model.
    models: list of (panel_title, agg_data_dict)
    Each panel shows normal / diffuse / non-diffuse score distributions.
    """
    groups = [
        ('normal',    'scores_normal',     '#55A868'),
        ('diffuse',   'scores_diffuse',    '#C44E52'),
        ('non-diff',  'scores_nondiffuse', '#DD8452'),
    ]

    def _panel(ax, data, title):
        all_vals = np.concatenate([data[k] for _, k, _ in groups if len(data[k]) > 0])
        if len(all_vals) == 0:
            ax.set_title(title)
            return
        x_lo, x_hi = float(all_vals.min()), float(all_vals.max())
        pad = (x_hi - x_lo) * 0.05
        xs = np.linspace(x_lo - pad, x_hi + pad, 500)

        for lbl, key, color in groups:
            vals = data[key]
            if len(vals) < 2:
                continue
            dens = gaussian_kde(vals)(xs)
            ax.plot(xs, dens, color=color, linewidth=2,
                    label=f'{lbl} (n={len(vals)}, μ={vals.mean():.3f})')
            ax.axvline(float(vals.mean()), color=color, linestyle='--',
                       linewidth=1, alpha=0.6)

        ax.set_title(title)
        ax.set_xlabel('anomaly score')
        ax.set_ylabel('density')
        ax.set_yticks([])
        ax.legend(fontsize=8)

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (title, data) in zip(axes, models):
        _panel(ax, data, f'{title}  (aggregated seeds)')
    fig.suptitle(f'Score distributions — diffuse vs non-diffuse{title_suffix}', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  KDE overlay saved: {output_path}")


def plot_auroc_summary(seeds, models, output_path: Path):
    """
    Grouped bar chart comparing N models for diffuse / non-diffuse / all subsets.
    models: list of (label, aurocs_dict, color)
            aurocs_dict has keys 'all', 'diffuse', 'nondiff', each a list over seeds.
    """
    subsets      = ['all', 'diffuse', 'nondiff']
    subset_labels = ['All anoms vs normal', 'Diffuse vs normal', 'Non-diffuse vs normal']
    n_models = len(models)
    n_subsets = len(subsets)
    total_width = 0.7
    width = total_width / n_models
    x = np.arange(n_subsets)

    fig, ax = plt.subplots(figsize=(4 + 3 * n_models, 5))
    for i, (label, aurocs, color) in enumerate(models):
        offset = (i - (n_models - 1) / 2) * width
        mu  = [np.nanmean(aurocs[s]) for s in subsets]
        std = [np.nanstd(aurocs[s])  for s in subsets]
        bars = ax.bar(x + offset, mu, width, yerr=std, capsize=4,
                      label=label, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(subset_labels)
    ax.set_ylabel('AUROC')
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color='grey', linestyle=':', linewidth=1)
    ax.legend(fontsize=8)
    ax.set_title('AUROC comparison  (mean ± std over seeds)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  AUROC bar chart saved: {output_path}")


def plot_per_seed_auroc(seeds, models, output_dir: Path):
    """
    One figure per subset showing all models per seed.
    models: list of (label, per_seed_dict, color)
    """
    subsets = [
        ('all',     'All anoms vs normal'),
        ('diffuse', 'Diffuse vs normal'),
        ('nondiff', 'Non-diffuse vs normal'),
    ]
    n_models = len(models)
    total_width = 0.7
    width = total_width / n_models
    x = np.arange(len(seeds))

    for key, title in subsets:
        fig, ax = plt.subplots(figsize=(4 + 2 * n_models, 4))
        for i, (label, per_seed, color) in enumerate(models):
            offset = (i - (n_models - 1) / 2) * width
            vals = [
                auroc_subsets(per_seed[s])[key] if per_seed.get(s) is not None else float('nan')
                for s in seeds
            ]
            ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f'seed {s}' for s in seeds])
        ax.set_ylabel('AUROC')
        ax.set_ylim(0, 1.1)
        ax.axhline(0.5, color='grey', linestyle=':', linewidth=1)
        ax.legend(fontsize=8)
        ax.set_title(f'AUROC per seed — {title}')
        plt.tight_layout()
        path = output_dir / f'per_seed_{key}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Per-seed plot saved: {path}")


def main():
    # ------------------------------------------------------------------ #
    #  Configure these variables before running                           #
    # ------------------------------------------------------------------ #
    VISUAL_TEST_DIR = '/home/marlon_helbing/nvme1/thesis/visual_test'

    # Use {SEED} as placeholder — it is replaced with 0, 1, and 42.
    # Uncomment exactly one block at a time.

    # --- PatchCore ---
  #  DINOV2_TEMPLATE = (
  #      'patchcore_dinov2_vitb14_5_data_'
  #      'FULL_NO_FILTERS_SEED_{SEED}_'
  #      'YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug'
  #  )
  #  WRN_TEMPLATE = (
  #      'patchcore_wide_resnet50_2_layer2_layer3_data_'
  #      'FULL_NO_FILTERS_SEED_{SEED}_'
  #      'YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug'
  #  )
  #  WRN_ADAPTED_TEMPLATE = (
  #      'patchcore_wide_resnet50_2_layer2_layer3_pretrained_data_'
  #      'FULL_NO_FILTERS_SEED_{SEED}_'
  #      'YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug'
  #  )

    # --- FastFlow ---
  #  DINOV2_TEMPLATE = (
  #      'fastflow_dinov2_vitb14_5_data_'
  #      'FULL_NO_FILTERS_SEED_{SEED}_'
  #      'YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug'
  #  )
  #  WRN_TEMPLATE = (
  #      'fastflow_wide_resnet50_2_layer1_layer2_layer3_data_'
  #      'FULL_NO_FILTERS_SEED_{SEED}_'
  #      'YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug'
  #  )
  #  WRN_ADAPTED_TEMPLATE = (
  #      'fastflow_wide_resnet50_2_layer1_layer2_layer3_pretrained_data_'
  #      'FULL_NO_FILTERS_SEED_{SEED}_'
  #      'YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug'
  #  )

    # --- SuperSimpleNet ---
  #  DINOV2_TEMPLATE = (
  #      'supersimplenet_dinov2_vitb14_5_data_'
  #      'FULL_NO_FILTERS_SEED_{SEED}_'
  #      'YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug'
  #  )
  #  WRN_TEMPLATE = (
  #      'supersimplenet_wide_resnet50_2_layer2_layer3_data_'
  #      'FULL_NO_FILTERS_SEED_{SEED}_'
  #      'YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug'
  #  )
  #  WRN_ADAPTED_TEMPLATE = (
  #      'supersimplenet_wide_resnet50_2_layer2_layer3_pretrained_data_'
  #      'FULL_NO_FILTERS_SEED_{SEED}_'
  #      'YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug'
  #  )

    # --- SINBAD ---
  #  DINOV2_TEMPLATE = (
  #      'sinbad_dinov2_vitb14_5_data_'
  #      'FULL_NO_FILTERS_SEED_{SEED}_'
  #      'YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug'
  #  )
  #  WRN_TEMPLATE = (
  #      'sinbad_wide_resnet50_2_layer3_layer4_data_'
  #      'FULL_NO_FILTERS_SEED_{SEED}_'
  #      'YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug'
  #  )
  #  WRN_ADAPTED_TEMPLATE = (
  #      'sinbad_wide_resnet50_2_layer3_layer4_pretrained_data_'
  #      'FULL_NO_FILTERS_SEED_{SEED}_'
  #      'YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug'
  #  )

    # --- CFA ---
  #  DINOV2_TEMPLATE = (
  #      'cfa_dinov2_vitb14_5_data_'
  #      'FULL_NO_FILTERS_SEED_{SEED}_'
  #      'YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug'
  #  )
  #  WRN_TEMPLATE = (
  #      'cfa_wide_resnet50_2_layer2_layer3_data_'
  #      'FULL_NO_FILTERS_SEED_{SEED}_'
  #      'YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug'
  #  )
  #  WRN_ADAPTED_TEMPLATE = (
  #      'cfa_wide_resnet50_2_layer2_layer3_pretrained_data_'
  #      'FULL_NO_FILTERS_SEED_{SEED}_'
  #      'YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug'
  #  )

    # --- STFPM ---
    DINOV2_TEMPLATE = (
        'stfpm_dinov2_vitb14_3_6_data_'
        'FULL_NO_FILTERS_SEED_{SEED}_'
        'YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug'
    )
    WRN_TEMPLATE = (
        'stfpm_wide_resnet50_2_layer2_layer3_data_'
        'FULL_NO_FILTERS_SEED_{SEED}_'
        'YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug'
    )
    WRN_ADAPTED_TEMPLATE = (
        'stfpm_wide_resnet50_2_layer2_layer3_pretrained_data_'
        'FULL_NO_FILTERS_SEED_{SEED}_'
        'YOLO_640_SHARED_TEST_SET_256_MAXMEAN_1_test_set_NONE_no_aug'
    )

    SEEDS = (0, 1, 42)

    # Where to save figures (None → visual_test_dir/dinov2_diffuse_check/)
    OUTPUT_DIR = None
    # ------------------------------------------------------------------ #

    visual_test_dir = Path(VISUAL_TEST_DIR)
    seeds = SEEDS

    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else visual_test_dir / 'dinov2_diffuse_check'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nVisual test dir: {visual_test_dir}")
    print(f"Output dir:      {output_dir}")
    print(f"Seeds:           {seeds}")
    print(f"\nDINOv2 template:       {DINOV2_TEMPLATE}")
    print(f"WRN    template:       {WRN_TEMPLATE}")
    print(f"WRN adapted template:  {WRN_ADAPTED_TEMPLATE}")

    # --- Load per-seed data ---
    print("\n[DINOv2] Loading folders ...")
    per_seed_dinov2, agg_dinov2 = load_seeds(visual_test_dir, DINOV2_TEMPLATE, seeds)

    print("\n[WRN-50-2] Loading folders ...")
    per_seed_wrn, agg_wrn = load_seeds(visual_test_dir, WRN_TEMPLATE, seeds)

    print("\n[WRN-50-2 adapted] Loading folders ...")
    per_seed_wrn_ada, agg_wrn_ada = load_seeds(visual_test_dir, WRN_ADAPTED_TEMPLATE, seeds)

    # --- Per-seed AUROC tables ---
    aurocs_all_d,  aurocs_diff_d,  aurocs_nd_d  = print_auroc_table('DINOv2 vitb14',        per_seed_dinov2,  seeds)
    aurocs_all_w,  aurocs_diff_w,  aurocs_nd_w  = print_auroc_table('WideResNet-50-2',       per_seed_wrn,     seeds)
    aurocs_all_wa, aurocs_diff_wa, aurocs_nd_wa = print_auroc_table('WideResNet-50-2 (ada)', per_seed_wrn_ada, seeds)

    # --- Comparison summary ---
    col_w = 14
    print(f"\n{'='*75}")
    print("  AUROC comparison summary  (mean ± std over seeds)")
    print(f"{'='*75}")
    print(f"  {'Subset':25}  {'DINOv2':>{col_w}}  {'WRN-50-2':>{col_w}}  {'WRN-50-2 (ada)':>{col_w}}")
    print(f"  {'-'*25}  {'-'*col_w}  {'-'*col_w}  {'-'*col_w}")

    all_auroc_sets = [
        ('All anoms vs normal',   aurocs_all_d,  aurocs_all_w,  aurocs_all_wa),
        ('Diffuse vs normal',     aurocs_diff_d, aurocs_diff_w, aurocs_diff_wa),
        ('Non-diffuse vs normal', aurocs_nd_d,   aurocs_nd_w,   aurocs_nd_wa),
    ]
    for name, a_d, a_w, a_wa in all_auroc_sets:
        def _fmt(a):
            return f"{np.nanmean(a):6.3f}±{np.nanstd(a):.3f}" if a else f"{'N/A':>{col_w}}"
        print(f"  {name:25}  {_fmt(a_d):>{col_w}}  {_fmt(a_w):>{col_w}}  {_fmt(a_wa):>{col_w}}")

    # --- Figures ---
    print("\nGenerating figures ...")

    model_specs = [
        ('DINOv2 vitb14',        agg_dinov2,  '#4C72B0', per_seed_dinov2,
         {'all': aurocs_all_d,  'diffuse': aurocs_diff_d,  'nondiff': aurocs_nd_d}),
        ('WRN-50-2',             agg_wrn,     '#DD8452', per_seed_wrn,
         {'all': aurocs_all_w,  'diffuse': aurocs_diff_w,  'nondiff': aurocs_nd_w}),
        ('WRN-50-2 (adapted)',   agg_wrn_ada, '#55A868', per_seed_wrn_ada,
         {'all': aurocs_all_wa, 'diffuse': aurocs_diff_wa, 'nondiff': aurocs_nd_wa}),
    ]

    plot_kde_overlay(
        [(label, agg) for label, agg, *_ in model_specs],
        output_dir / 'kde_overlay.png',
    )
    plot_auroc_summary(
        seeds,
        [(label, aurocs, color) for label, _, color, _, aurocs in model_specs],
        output_dir / 'auroc_bar.png',
    )
    plot_per_seed_auroc(
        seeds,
        [(label, per_seed, color) for label, _, color, per_seed, _ in model_specs],
        output_dir,
    )

    print(f"\nDone. All outputs in: {output_dir}")


if __name__ == '__main__':
    main()
