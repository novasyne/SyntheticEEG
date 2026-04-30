import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

AGG_FILE = "cross_dataset_results_synthetic_agg.csv"

CONDITIONS = [
    ("Real→Cross",             "Real-Cross"),
    ("Synthetic→Cross",        "Synthetic-Cross (Interpolation)"),
    ("Real+Synthetic→Cross",   "Real+Synthetic-Cross"),
    ("Real+SyntheticDA→Cross", "DA-Cross"),
]

def has_mw(df):
    src_mw = df["Train_Dataset"].str.contains("MW", na=False)
    tgt_mw = df["Test_Dataset"].str.contains("MW", na=False)
    return src_mw | tgt_mw


def best_mean_auc(df, exp_type, synth_method_filter=None):
    sub = df[df["Experiment_Type"] == exp_type].copy()
    if synth_method_filter is not None:
        sub = sub[sub["Synth_Method"] == synth_method_filter]

    if sub.empty:
        return pd.DataFrame(columns=["source", "target", "best_model", "auc"])

    idx = sub.groupby(["Train_Dataset", "Test_Dataset", "Model"])["AUC"].mean()
    idx = idx.reset_index(name="mean_auc")
    best_idx = idx.groupby(["Train_Dataset", "Test_Dataset"])["mean_auc"].idxmax()
    best = idx.loc[best_idx].copy()
    best = best.rename(columns={
        "Train_Dataset": "source",
        "Test_Dataset":  "target",
        "Model":         "best_model",
        "mean_auc":      "auc",
    })
    return best.reset_index(drop=True)


def da_best_per_pair(df):
    sub = df[df["Experiment_Type"] == "Real+SyntheticDA→Cross"].copy()
    if sub.empty:
        return pd.DataFrame(columns=["source", "target", "best_model", "auc"])

    idx = (sub.groupby(["Train_Dataset", "Test_Dataset", "Model", "Synth_Method"])
           ["AUC"].mean().reset_index(name="mean_auc"))
    best_idx = idx.groupby(["Train_Dataset", "Test_Dataset"])["mean_auc"].idxmax()
    best = idx.loc[best_idx].copy()
    best = best.rename(columns={
        "Train_Dataset": "source",
        "Test_Dataset":  "target",
        "Model":         "best_model",
        "mean_auc":      "auc",
    })
    return best.reset_index(drop=True)


def aug_best_per_pair(df):
    sub = df[df["Experiment_Type"] == "Real+Synthetic→Cross"].copy()
    if sub.empty:
        return pd.DataFrame(columns=["source", "target", "best_model", "auc"])

    idx = (sub.groupby(["Train_Dataset", "Test_Dataset", "Model", "Synth_Method"])
           ["AUC"].mean().reset_index(name="mean_auc"))
    best_idx = idx.groupby(["Train_Dataset", "Test_Dataset"])["mean_auc"].idxmax()
    best = idx.loc[best_idx].copy()
    best = best.rename(columns={
        "Train_Dataset": "source",
        "Test_Dataset":  "target",
        "Model":         "best_model",
        "mean_auc":      "auc",
    })
    return best.reset_index(drop=True)


def synth_cross_best(df, method="Interpolation"):
    sub = df[(df["Experiment_Type"] == "Synthetic→Cross") &
             (df["Synth_Method"] == method)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["source", "target", "best_model", "auc"])
    idx = (sub.groupby(["Train_Dataset", "Test_Dataset", "Model"])
           ["AUC"].mean().reset_index(name="mean_auc"))
    best_idx = idx.groupby(["Train_Dataset", "Test_Dataset"])["mean_auc"].idxmax()
    best = idx.loc[best_idx].copy()
    best = best.rename(columns={
        "Train_Dataset": "source",
        "Test_Dataset":  "target",
        "Model":         "best_model",
        "mean_auc":      "auc",
    })
    return best.reset_index(drop=True)


def synth_cross_best_overall(df):
    sub = df[df["Experiment_Type"] == "Synthetic→Cross"].copy()
    if sub.empty:
        return pd.DataFrame(columns=["source", "target", "best_model", "auc"])
    idx = (sub.groupby(["Train_Dataset", "Test_Dataset", "Model", "Synth_Method"])
           ["AUC"].mean().reset_index(name="mean_auc"))
    best_idx = idx.groupby(["Train_Dataset", "Test_Dataset"])["mean_auc"].idxmax()
    best = idx.loc[best_idx].copy()
    best = best.rename(columns={
        "Train_Dataset": "source",
        "Test_Dataset":  "target",
        "Model":         "best_model",
        "mean_auc":      "auc",
    })
    return best.reset_index(drop=True)


def delta(aug_pairs, baseline_pairs):
    merged = aug_pairs.merge(
        baseline_pairs[["source", "target", "auc"]],
        on=["source", "target"],
        suffixes=("_aug", "_base"),
    )
    merged["delta"] = (merged["auc_aug"] - merged["auc_base"]).round(4)
    return merged


def print_pairs(pairs_df, title, delta_col=False):
    print(f"\n  {title} ({len(pairs_df)} pairs):")
    for _, row in pairs_df.sort_values("auc", ascending=False).iterrows():
        src = row["source"].replace("Synthetic_", "")
        tgt = row["target"]
        auc = f"{row['auc']:.4f}"
        extra = f"  Δ={row['delta']:+.4f}" if delta_col and "delta" in row else ""
        print(f"    {src:>6} → {tgt:<6}  AUC={auc}{extra}")


def run():
    print("=" * 60)
    print("MW Exclusion Check")
    print("Datasets retained: EDMA, MS, SAM40 (6 source-target pairs)")
    print("=" * 60)

    df_full = pd.read_csv(AGG_FILE)
    df_excl = df_full[~has_mw(df_full)].copy()

    print(f"\nFull dataset: {len(df_full)} rows")
    print(f"MW-excluded:  {len(df_excl)} rows")

    summary_rows = []
    pair_rows    = []

    for label, df_src, tag in [("Full (12 pairs)", df_full, "full"),
                                ("MW-excluded (6 pairs)", df_excl, "excl")]:
        pairs = best_mean_auc(df_src, "Real→Cross")
        mean  = pairs["auc"].mean()
        print(f"\nReal-Cross  [{label}]  mean AUC = {mean:.4f}")
        print_pairs(pairs, f"Pairs [{label}]")
        summary_rows.append({"condition": "Real-Cross", "scope": label,
                              "n_pairs": len(pairs), "mean_auc": round(mean, 4)})
        for _, r in pairs.iterrows():
            pair_rows.append({"condition": "Real-Cross", "scope": tag,
                               "source": r["source"], "target": r["target"],
                               "best_model": r["best_model"], "auc": r["auc"]})

    for label, df_src, tag in [("Full (12 pairs)", df_full, "full"),
                                ("MW-excluded (6 pairs)", df_excl, "excl")]:
        pairs = synth_cross_best_overall(df_src)
        mean  = pairs["auc"].mean()
        print(f"\nSynthetic-Cross (best method) [{label}]  mean AUC = {mean:.4f}")
        print_pairs(pairs, f"Pairs [{label}]")
        summary_rows.append({"condition": "Synthetic-Cross", "scope": label,
                              "n_pairs": len(pairs), "mean_auc": round(mean, 4)})
        for _, r in pairs.iterrows():
            pair_rows.append({"condition": "Synthetic-Cross", "scope": tag,
                               "source": r["source"], "target": r["target"],
                               "best_model": r["best_model"], "auc": r["auc"]})

    for label, df_src, tag in [("Full (12 pairs)", df_full, "full"),
                                ("MW-excluded (6 pairs)", df_excl, "excl")]:
        pairs_aug  = aug_best_per_pair(df_src)
        pairs_base = best_mean_auc(df_src, "Real→Cross")
        pairs_aug  = pairs_aug.copy()

        d_pairs = delta(pairs_aug, pairs_base)
        mean_aug   = pairs_aug["auc"].mean()
        mean_delta = d_pairs["delta"].mean() if not d_pairs.empty else float("nan")
        n_pos = (d_pairs["delta"] > 0).sum() if not d_pairs.empty else 0

        print(f"\nReal+Synthetic-Cross [{label}]  "
              f"mean AUC = {mean_aug:.4f}  Δ={mean_delta:+.4f}  "
              f"positive: {n_pos}/{len(pairs_aug)}")
        summary_rows.append({"condition": "Real+Synthetic-Cross", "scope": label,
                              "n_pairs": len(pairs_aug), "mean_auc": round(mean_aug, 4),
                              "mean_delta": round(mean_delta, 4),
                              "n_positive": n_pos})

    for label, df_src, tag in [("Full (12 pairs)", df_full, "full"),
                                ("MW-excluded (6 pairs)", df_excl, "excl")]:
        pairs_da   = da_best_per_pair(df_src)
        pairs_base = best_mean_auc(df_src, "Real→Cross")

        d_pairs    = delta(pairs_da, pairs_base)
        mean_da    = pairs_da["auc"].mean()
        mean_delta = d_pairs["delta"].mean() if not d_pairs.empty else float("nan")
        n_pos = (d_pairs["delta"] > 0).sum() if not d_pairs.empty else 0

        print(f"\nDA-Cross [{label}]  "
              f"mean AUC = {mean_da:.4f}  Δ={mean_delta:+.4f}  "
              f"positive: {n_pos}/{len(pairs_da)}")
        summary_rows.append({"condition": "DA-Cross", "scope": label,
                              "n_pairs": len(pairs_da), "mean_auc": round(mean_da, 4),
                              "mean_delta": round(mean_delta, 4),
                              "n_positive": n_pos})

    print("\n" + "=" * 60)
    print("Condition Ranking (mean AUC, best model/method per pair)")
    for scope_label, scope_tag in [("Full (12 pairs)", "Full (12 pairs)"),
                                    ("MW-excluded (6 pairs)", "MW-excluded (6 pairs)")]:
        row_real  = next((r for r in summary_rows
                          if r["condition"] == "Real-Cross" and r["scope"] == scope_label), None)
        row_synth = next((r for r in summary_rows
                          if r["condition"] == "Synthetic-Cross" and r["scope"] == scope_label), None)
        row_aug   = next((r for r in summary_rows
                          if r["condition"] == "Real+Synthetic-Cross" and r["scope"] == scope_label), None)
        row_da    = next((r for r in summary_rows
                          if r["condition"] == "DA-Cross" and r["scope"] == scope_label), None)
        if not all([row_real, row_synth, row_aug, row_da]):
            continue
        vals = [
            ("Real-Cross",          row_real["mean_auc"]),
            ("Synthetic-Cross",     row_synth["mean_auc"]),
            ("Real+Synthetic",      row_aug["mean_auc"]),
            ("DA-Cross",            row_da["mean_auc"]),
        ]
        ranked = sorted(vals, key=lambda x: x[1], reverse=True)
        rank_str = " > ".join(f"{n}({v:.3f})" for n, v in ranked)
        print(f"\n  {scope_label}: {rank_str}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("mw_exclusion_summary.csv", index=False)
    print(f"\nSummary → mw_exclusion_summary.csv")

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv("mw_exclusion_pairs.csv", index=False)
    print(f"Pairs    → mw_exclusion_pairs.csv")

    print("\n" + "=" * 60)
    print("Key conclusion check (MW-excluded):")

    excl_real  = next((r["mean_auc"] for r in summary_rows
                       if r["condition"] == "Real-Cross"
                       and "excl" in r["scope"].lower()), None)
    excl_synth = next((r["mean_auc"] for r in summary_rows
                       if r["condition"] == "Synthetic-Cross"
                       and "excl" in r["scope"].lower()), None)
    excl_da    = next((r["mean_auc"] for r in summary_rows
                       if r["condition"] == "DA-Cross"
                       and "excl" in r["scope"].lower()), None)

    if excl_real and excl_synth:
        gain = excl_synth - excl_real
        pct  = 100 * gain / excl_real
        print(f"  Synthetic-Cross gain over Real-Cross: "
              f"{gain:+.4f} ({pct:+.1f}%)  "
              f"({'✓ holds' if gain > 0 else '✗ does not hold'})")

    if excl_real and excl_da:
        da_gain = excl_da - excl_real
        print(f"  DA-Cross gain over Real-Cross:        "
              f"{da_gain:+.4f}  "
              f"({'✓ holds' if da_gain > 0 else '✗ does not hold'})")

    print("=" * 60)


if __name__ == "__main__":
    run()
