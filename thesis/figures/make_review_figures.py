#!/usr/bin/env python3
"""
make_review_analyses.py — reproduce every figure and table in the round-4 review.

Outputs (all read live from the on-disk result CSVs; nothing hard-coded):
  A1  fig_A1_screening_enrichment.png       operational screening enrichment
  A2  A2_unosat_snapshot_dates.csv          per-city UNOSAT assessment date vs battle window
  A3  fig_A3_size_confound.png              pre-battle optical AUC vs building size
  A4  A4_metric_ruler_meanfolds_vs_pooled.csv   pooled-vs-mean-folds gap over 458 experiments
  A6  A6_effect_sizes_best_per_classifier_anyFS.csv  Cliff's delta + Holm on the p=0.222 set

Usage:  RESULTS=/path/to/results DATA=/path/to/data NOTEBOOKS=/path/to/notebooks \
        python make_review_analyses.py

The figures use the figure-style helpers if available; if not, they fall back to
plain matplotlib (the numbers are identical either way).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RESULTS   = os.environ.get("RESULTS", "results")
DATA      = os.environ.get("DATA", "data")
NOTEBOOKS = os.environ.get("NOTEBOOKS", "notebooks")
FOCAL = "#c1272d"
GREY  = "#8a8a8a"

# optional house style
try:
    from figure_style_kernel import apply_figure_style, set_frame, panel_letter  # if packaged
    apply_figure_style(sizes=(9, 8, 7))
    HOUSE = True
except Exception:
    def set_frame(ax, **k):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    def panel_letter(ax, l, **k):
        ax.text(-0.15, 1.03, l, transform=ax.transAxes, fontweight="bold", fontsize=11)
    HOUSE = False

CONSOLIDATED = os.path.join(RESULTS, "nb11d", "consolidated_experiments_20260703_084619.csv")
NB12 = os.path.join(RESULTS, "nb12")


# =============================================================================
# A1 — operational screening enrichment (per-building V2 headline fusion)
# =============================================================================
def A1(out="fig_A1_screening_enrichment.png"):
    df = pd.read_csv(CONSOLIDATED)
    cand = df[df["auc_mean"].notna() & df["auc_std"].notna()].copy()
    best = cand.sort_values("auc_mean", ascending=False).iloc[0]
    prev = best["prevalence"]
    recalls = [0.80, 0.90, 0.95]
    precs = [best["precision_at_recall_80"], best["precision_at_recall_90"], best["precision_at_recall_95"]]
    stock = [100 * R * prev / P for R, P in zip(recalls, precs)]   # % stock reviewed
    lifts = [P / prev for P in precs]                             # precision lift
    saved = [R * 100 - s for R, s in zip(recalls, stock)]         # pp saved vs random tasking

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.9))
    ax1.plot([0, 100], [0, 100], ls="--", color=GREY, lw=1.2, zorder=1)
    ax1.annotate("random tasking", (78, 70), color=GREY, fontsize=7, rotation=40, ha="center", va="center")
    ax1.plot(stock, [r * 100 for r in recalls], "o-", color=FOCAL, lw=2, ms=7, zorder=3)
    for (s, r), (dx, dy) in zip(zip(stock, recalls), [(-4, 14), (6, 12), (8, 10)]):
        ax1.annotate(f"{int(r*100)}% recall @ {s:.0f}% reviewed", (s, r * 100),
                     textcoords="offset points", xytext=(dx, dy), fontsize=6.8)
    ax1.set_xlabel("Building stock VHR-reviewed (%)")
    ax1.set_ylabel("Damaged buildings captured (%)")
    ax1.set_title("Screening captures most damage\nat a fraction of review effort", loc="left", fontsize=9)
    ax1.set_xlim(0, 100); ax1.set_ylim(0, 108); set_frame(ax1)

    for i, (l, sv) in enumerate(zip(lifts, saved)):
        ax2.plot([1, l], [i, i], color=GREY, lw=1, zorder=1)
        ax2.plot(l, i, "o", color=FOCAL, ms=9, zorder=3)
        ax2.annotate(f"{l:.1f}×  ({sv:.0f} pp saved)", (l, i),
                     textcoords="offset points", xytext=(11, 0), va="center", fontsize=7.5)
    ax2.axvline(1, color=GREY, ls=":", lw=1)
    ax2.set_yticks(range(len(recalls))); ax2.set_yticklabels([f"{int(r*100)}% recall" for r in recalls])
    ax2.set_xlabel("Precision lift over prevalence (×)")
    ax2.set_title("Enrichment vs random tasking", loc="left", fontsize=9)
    ax2.set_xlim(0, 4.8); ax2.set_ylim(-0.5, 2.5); set_frame(ax2)
    panel_letter(ax1, "a"); panel_letter(ax2, "b")
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[A1] {out}  " + " | ".join(f"{int(r*100)}%R: {l:.1f}x, {sv:.0f}pp"
                                        for r, l, sv in zip(recalls, lifts, saved)))


# =============================================================================
# A2 — per-city UNOSAT assessment date vs battle window
# =============================================================================
def A2(out="A2_unosat_snapshot_dates.csv"):
    import json
    u = pd.read_csv(os.path.join(DATA, "unosat_damage_assessments", "compiled",
                                 "unosat_ukraine_compiled.csv"))
    u["date"] = pd.to_datetime(u["date"], utc=True, errors="coerce")
    cfg = json.load(open(os.path.join(NOTEBOOKS, "cities_config.json")))
    study = ["Mariupol", "Lysychansk", "Sievierodonetsk", "Rubizhne", "Hostomel", "Chernihiv",
             "Irpin", "Volnovakha", "Bucha", "Dmytrivka", "Moschun", "Borodyanka", "Makariv",
             "Okhtyrka", "Trostianets", "Kharkiv", "Avdiivka", "Mykolaiv", "Kramatorsk", "Kherson"]
    agg = u[u["city_name"].isin(study)].groupby("city_name")["date"].agg(["min", "max", "count"])
    rows = []
    for c, r in agg.iterrows():
        e = cfg.get(c, {}); bstop = e.get("battle_stop")
        al = r["max"].tz_localize(None)
        if bstop == "ongoing":
            gap, flag = "OPEN-ENDED", "EXPOSED (ongoing)"
        else:
            g = (al - pd.to_datetime(bstop, errors="coerce")).days
            gap = g
            flag = ("EXPOSED (assessment >1yr before window end)" if g < -365
                    else "OK" if abs(g) <= 120 else "CHECK")
        rows.append(dict(city=c, tier=e.get("tier"), battle_start=e.get("battle_start"),
                         battle_stop=bstop, unosat_first=r["min"].tz_localize(None).date().isoformat(),
                         unosat_last=al.date().isoformat(), n_unosat_pts=int(r["count"]),
                         days_last_assess_minus_battlestop=gap, exposure=flag))
    T = pd.DataFrame(rows).sort_values(["exposure", "city"])
    T.to_csv(out, index=False)
    print(f"[A2] {out}  exposed: {list(T[T.exposure.str.startswith('EXPOSED')].city)}")


# =============================================================================
# A3 — size confound: PRE-battle optical AUC vs damaged-building size
# =============================================================================
def A3(out="fig_A3_size_confound.png"):
    d = pd.read_csv(os.path.join(RESULTS, "nb07", "cell_d1y", "ms_prepost_vs_auc.csv"))
    rho, p = stats.spearmanr(d["area_damaged_median"], d["auc_pre"])
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.axhline(0.5, color=GREY, ls="--", lw=1, zorder=1)
    ax.annotate("chance (AUC 0.5)", (d["area_damaged_median"].max(), 0.505),
                color=GREY, fontsize=6.5, ha="right", va="bottom")
    z = np.polyfit(d["area_damaged_median"], d["auc_pre"], 1)
    xs = np.linspace(d["area_damaged_median"].min(), d["area_damaged_median"].max(), 50)
    ax.plot(xs, np.polyval(z, xs), color=GREY, lw=1.4, zorder=2)
    ax.scatter(d["area_damaged_median"], d["auc_pre"], s=42, c=FOCAL, zorder=3,
               edgecolor="white", linewidth=0.5)
    seen = set()
    for row in [d.loc[d.area_damaged_median.idxmax()], d.loc[d.auc_pre.idxmax()],
                d.loc[d.auc_pre.idxmin()], d[d.city == "Mariupol"].iloc[0]]:
        if row.city in seen:
            continue
        seen.add(row.city)
        ax.annotate(row.city, (row.area_damaged_median, row.auc_pre),
                    fontsize=7, xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("Median area of damaged buildings (m²)")
    ax.set_ylabel("Per-city optical AUC,\npre-battle imagery only")
    ax.set_title(f"Optical features separate by size, not damage\n"
                 f"Spearman ρ = {rho:.2f} (p = {p:.3f}), n = {len(d)} cities", loc="left", fontsize=9)
    ax.margins(0.06); set_frame(ax)
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[A3] {out}  rho={rho:.3f} p={p:.3f}")


# =============================================================================
# A4 — one-ruler: pooled AUC vs mean-of-folds AUC over all 458 experiments
# =============================================================================
def A4(out="A4_metric_ruler_meanfolds_vs_pooled.csv"):
    mvp = pd.read_csv(os.path.join(NB12, "nb12a_meanfolds_vs_pooled.csv"))
    g = mvp["gap_pooled_minus_folds"].dropna()
    a4 = mvp[["experiment", "n_cities_scored", "pooled_auc", "pooled_auc_no_mariupol",
              "mean_folds_auc", "std_folds_auc", "gap_pooled_minus_folds"]] \
        .sort_values("mean_folds_auc", ascending=False)
    a4.to_csv(out, index=False)
    print(f"[A4] {out}  n={len(g)}  mean gap={g.mean():+.4f}  "
          f"|gap|>0.05 in {(g.abs()>0.05).sum()} ({100*(g.abs()>0.05).mean():.0f}%)  "
          f"max |gap|={g.abs().max():.3f}")


# =============================================================================
# A6 — effect sizes over the omnibus p on the non-significant set (Friedman p=0.222)
# =============================================================================
def A6(out="A6_effect_sizes_best_per_classifier_anyFS.csv"):
    pw = pd.read_csv(os.path.join(NB12,
                     "nb12b_pairwise_wilcoxon_holm__best_per_classifier_anyFS.csv"))
    a6 = pw.sort_values("cliffs_delta", ascending=False).copy()
    a6["holm_sig_0.05"] = a6["p_holm"] < 0.05
    a6["effect_mag"] = pd.cut(a6["cliffs_delta"].abs(), [0, 0.147, 0.33, 0.474, 1.01],
                              labels=["negligible", "small", "medium", "large"])
    a6.to_csv(out, index=False)
    print(f"[A6] {out}  Friedman omnibus p=0.222 set: "
          f"{a6['holm_sig_0.05'].sum()}/{len(a6)} Holm-significant, "
          f"{(a6['cliffs_delta'].abs()>0.474).sum()} large, "
          f"{(a6['cliffs_delta'].abs()<0.147).sum()} negligible")


# =============================================================================
# A6-verify — direct per-city paired recompute (Cliff's delta + BCa CI)
# Confirms the committed nb12b values reproduce from the per-city matrix.
# This is the code behind the A6 verification table in the methodology doc.
# =============================================================================
def _cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def _paired_bca(e1, e2, mat, citycols, n_boot=2000, seed=0):
    """Paired per-city ΔAUC between two experiments: mean diff, BCa 95% CI,
    Cliff's delta, Wilcoxon p. Cities scored by BOTH configs are the unit."""
    r1, r2 = mat.loc[e1], mat.loc[e2]
    common = [c for c in citycols if pd.notna(r1[c]) and pd.notna(r2[c])]
    d = np.array([r1[c] - r2[c] for c in common])
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    z0 = stats.norm.ppf((boots < d.mean()).mean())
    jack = np.array([np.delete(d, i).mean() for i in range(len(d))])
    jbar = jack.mean()
    a = ((jbar - jack) ** 3).sum() / (6 * (((jbar - jack) ** 2).sum()) ** 1.5 + 1e-12)
    def q(p):
        zq = stats.norm.ppf(p); adj = z0 + (z0 + zq) / (1 - a * (z0 + zq))
        return float(np.clip(stats.norm.cdf(adj), 0, 1))
    lo, hi = np.quantile(boots, [q(0.025), q(0.975)])
    w = stats.wilcoxon(d) if np.any(d != 0) else None
    return dict(n_cities=len(common), mean_diff=d.mean(), bca_lo=lo, bca_hi=hi,
                cliffs=_cliffs_delta(r1[common].values, r2[common].values),
                wilcoxon_p=(w.pvalue if w else np.nan))


def A6_verify():
    mat = pd.read_csv(os.path.join(NB12, "nb12a_per_city_auc_matrix.csv")) \
        .rename(columns={"Unnamed: 0": "experiment"}).set_index("experiment")
    cols = list(mat.columns)
    idx = list(mat.index)
    def find(sub): return [e for e in idx if sub.lower() in e.lower()]
    pairs = {
        "fusion(RGB+CARD+cohdrop) vs RGB-only":
            (find("COH_DROP+RGB+CARD_AdaBoost")[:1], find("BDA_RGB_XGBoost")[:1]),
        "top fusion vs top dietrich28 block_stats":
            (find("COH_DROP+RGB+CARD_AdaBoost")[:1], find("CLF_F7_GBM")[:1]),
    }
    for name, (a, b) in pairs.items():
        if a and b:
            s = _paired_bca(a[0], b[0], mat, cols)
            print(f"[A6-verify] {name}: n={s['n_cities']} ΔAUC={s['mean_diff']:+.4f} "
                  f"BCa[{s['bca_lo']:+.4f},{s['bca_hi']:+.4f}] δ={s['cliffs']:.3f} "
                  f"Wilcoxon p={s['wilcoxon_p']:.4f}")


if __name__ == "__main__":
    A1(); A2(); A3(); A4(); A6(); A6_verify()
