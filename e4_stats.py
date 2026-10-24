"""
Revision experiment E4 (Reviewer #2.3).

The manuscript treats the 12 directed source-target pairs as independent paired observations in
Wilcoxon signed-rank tests. They are not: they are generated from only four datasets, each of
which appears repeatedly as both source and target, so each dataset contributes to six pairs.
This module provides dependence-aware alternatives plus effect sizes and confidence intervals.

  naive_wilcoxon        - the published test, retained for reference only
  cluster_bootstrap     - resamples the four DATASETS with replacement (pair multiplicity =
                          n_source x n_target), giving a percentile CI on the mean difference
  cluster_signflip      - exhaustive sign-flip permutation over the 2^4 = 16 dataset-level sign
                          assignments; a pair's sign is the product of its two datasets' signs.
                          The smallest attainable p-value under this scheme is 1/16 = 0.0625,
                          which is itself the key point: with four underlying datasets, no
                          comparison in this design can reach p < 0.05.
  rank_biserial         - paired rank-biserial effect size with a cluster-bootstrap CI
  effective_n           - design effect / effective sample size given the clustering

Imported by revision_summary.py; can also be run directly for a self-test.
"""
import itertools

import numpy as np
from scipy.stats import wilcoxon


def naive_wilcoxon(delta):
    delta = np.asarray(delta, dtype=float)
    if np.allclose(delta, 0):
        return np.nan, np.nan
    w, p = wilcoxon(delta)
    return float(w), float(p)


def _pair_weights(pairs, counts):
    """Multiplicity of each directed pair when datasets are resampled with replacement."""
    return np.array([counts.get(s, 0) * counts.get(t, 0) for s, t in pairs], dtype=float)


def cluster_bootstrap(pairs, delta, datasets, n_boot=10000, seed=42, alpha=0.05):
    """Percentile CI on the mean difference, resampling DATASETS rather than pairs."""
    rng   = np.random.default_rng(seed)
    delta = np.asarray(delta, dtype=float)
    ds    = list(datasets)
    means = []
    for _ in range(n_boot):
        drawn  = rng.choice(ds, size=len(ds), replace=True)
        counts = {d: int((drawn == d).sum()) for d in ds}
        w      = _pair_weights(pairs, counts)
        if w.sum() == 0:
            continue
        means.append(float(np.average(delta, weights=w)))
    means = np.array(means)
    return {
        "mean":    float(delta.mean()),
        "ci_low":  float(np.percentile(means, 100 * alpha / 2)),
        "ci_high": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "p_boot":  float(2 * min((means <= 0).mean(), (means >= 0).mean())),
    }


def cluster_signflip(pairs, delta, datasets):
    """Exhaustive dataset-level sign-flip permutation test (exchangeability blocks)."""
    delta = np.asarray(delta, dtype=float)
    ds    = list(datasets)
    obs   = abs(delta.mean())
    stats = []
    for signs in itertools.product([1, -1], repeat=len(ds)):
        smap = dict(zip(ds, signs))
        flip = np.array([smap[s] * smap[t] for s, t in pairs], dtype=float)
        stats.append(abs((delta * flip).mean()))
    stats = np.array(stats)
    return {
        "p":          float((stats >= obs - 1e-12).mean()),
        "n_perms":    len(stats),
        "p_floor":    float(1.0 / len(stats)),
    }


def rank_biserial(pairs, delta, datasets, n_boot=10000, seed=42, alpha=0.05):
    """Paired rank-biserial correlation with a dataset-level cluster-bootstrap CI."""
    delta = np.asarray(delta, dtype=float)

    def _rb(d, w=None):
        d = d[d != 0]
        if len(d) == 0:
            return 0.0
        if w is None:
            return float(np.sign(d).mean())
        w = w[: len(d)]
        return float(np.average(np.sign(d), weights=w)) if w.sum() else 0.0

    rng, ds, vals = np.random.default_rng(seed), list(datasets), []
    for _ in range(n_boot):
        drawn  = rng.choice(ds, size=len(ds), replace=True)
        counts = {d: int((drawn == d).sum()) for d in ds}
        w      = _pair_weights(pairs, counts)
        nz     = delta != 0
        if w[nz].sum() == 0:
            continue
        vals.append(float(np.average(np.sign(delta[nz]), weights=w[nz])))
    vals = np.array(vals)
    return {
        "rank_biserial": _rb(delta),
        "ci_low":        float(np.percentile(vals, 100 * alpha / 2)),
        "ci_high":       float(np.percentile(vals, 100 * (1 - alpha / 2))),
    }


def effective_n(pairs, datasets):
    """Design effect from the clustering: 12 pairs carried by len(datasets) independent units."""
    n_pairs = len(pairs)
    k       = len(datasets)
    # each dataset appears in 2*(k-1) of the k*(k-1) directed pairs
    icc_pairs_per_ds = 2 * (k - 1)
    return {
        "n_pairs":            n_pairs,
        "n_datasets":         k,
        "pairs_per_dataset":  icc_pairs_per_ds,
        "max_independent":    k // 2,   # disjoint source/target pairs available at once
    }


def analyse(pairs, delta, datasets, label=""):
    w, p = naive_wilcoxon(delta)
    cb   = cluster_bootstrap(pairs, delta, datasets)
    sf   = cluster_signflip(pairs, delta, datasets)
    rb   = rank_biserial(pairs, delta, datasets)
    return {
        "Comparison":        label,
        "Mean_Delta":        round(cb["mean"], 4),
        "CI_Low":            round(cb["ci_low"], 4),
        "CI_High":           round(cb["ci_high"], 4),
        "Positive_Pairs":    int((np.asarray(delta) > 0).sum()),
        "N_Pairs":           len(delta),
        "Wilcoxon_p_naive":  round(p, 4) if p == p else np.nan,
        "ClusterBoot_p":     round(cb["p_boot"], 4),
        "SignFlip_p":        round(sf["p"], 4),
        "SignFlip_p_floor":  round(sf["p_floor"], 4),
        "RankBiserial":      round(rb["rank_biserial"], 3),
        "RB_CI_Low":         round(rb["ci_low"], 3),
        "RB_CI_High":        round(rb["ci_high"], 3),
    }


if __name__ == "__main__":
    ds    = ["EDMA", "MS", "MW", "SAM"]
    prs   = [(a, b) for a in ds for b in ds if a != b]
    rng   = np.random.default_rng(0)
    print(analyse(prs, rng.normal(0.05, 0.05, 12), ds, "self-test"))
