#!/usr/bin/env python3
"""Reproduce a decision-quality benchmark; no generated observations or labels.

Run: python research.py --run
Then: python research.py --report  (rebuilds all tables and plots without fitting)
Test: python research.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import os
import platform
import time
import urllib.request
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
SEED = 802
RATES = [0.0, 0.15, 0.30, 0.50]
REPS = 5
COST = 0.10
CAPACITY = 0.20
METHODS = ["BI frequency", "Logistic", "Random forest", "Gradient boosting"]
SOURCES = {
    "bank": ("https://archive.ics.uci.edu/static/public/222/bank+marketing.zip", "bank_marketing.zip"),
    "shop": ("https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip", "online_shoppers.zip"),
}
SOURCE_SHA256 = {
    "bank": "e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4",
    "shop": "2972e6184d3ad7beaaa831d9fc2b059dc3ee29df69d1ec593c466a5cd8485d14",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_data(name):
    DATA.mkdir(exist_ok=True)
    url, filename = SOURCES[name]
    path = DATA / filename
    if not path.exists():
        with urllib.request.urlopen(url, timeout=90) as r:
            path.write_bytes(r.read())
    if digest(path) != SOURCE_SHA256[name]:
        raise ValueError(f"Source archive checksum differs from the audited dataset: {path}")
    z = zipfile.ZipFile(path)
    if name == "bank":
        zz = zipfile.ZipFile(io.BytesIO(z.read("bank-additional.zip")))
        df = pd.read_csv(io.BytesIO(zz.read("bank-additional/bank-additional-full.csv")), sep=";")
        excluded = ["duration"]
        y = df.y.eq("yes").astype(int)
        x = df.drop(columns=["y", *excluded])
        categorical = x.select_dtypes(include="object").columns.tolist()
        bi_columns = ["poutcome", "job"]
    else:
        df = pd.read_csv(io.BytesIO(z.read("online_shoppers_intention.csv")))
        excluded = ["PageValues"]
        y = df.Revenue.astype(int)
        x = df.drop(columns=["Revenue", *excluded])
        categorical = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]
        bi_columns = ["VisitorType", "TrafficType"]
    # Remove exact source-record duplicates before splitting, retaining original order.
    keep = ~df.duplicated()
    ids = np.flatnonzero(keep.to_numpy())
    x = x.loc[keep].reset_index(drop=True)
    y = y.loc[keep].to_numpy()
    for c in categorical:
        x[c] = x[c].astype(str)
    info = dict(raw_n=len(df), n=len(x), duplicates_removed=int((~keep).sum()),
                positives=int(y.sum()), features=x.columns.tolist(), excluded=excluded,
                categorical=categorical, source_url=url, source_sha256=digest(path))
    return x, y, ids, categorical, bi_columns, info


def splits(name, y):
    n = len(y)
    if name == "bank":
        for f in range(5):
            a = int(n * (0.4 + 0.1 * f))
            b = int(n * (0.5 + 0.1 * f))
            c = int(n * (0.6 + 0.1 * f)) if f < 4 else n
            yield np.arange(a), np.arange(a, b), np.arange(b, c)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for f, (development, test) in enumerate(cv.split(np.zeros(n), y)):
            train, cal = train_test_split(development, test_size=0.2, stratify=y[development], random_state=SEED + f)
            yield np.sort(train), np.sort(cal), np.sort(test)


def preprocessing(x, categorical):
    numeric = [c for c in x if c not in categorical]
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="__MISSING__"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical),
    ], sparse_threshold=0)


class Frequency:
    """Descriptive BI benchmark with training-only beta smoothing."""
    def __init__(self, columns):
        self.columns = columns

    def keys(self, x):
        return list(map(tuple, x[self.columns].fillna("__MISSING__").astype(str).to_numpy()))

    def fit(self, x, y):
        self.prior = float(np.mean(y))
        totals = {}
        for k, label in zip(self.keys(x), y):
            n, s = totals.get(k, (0, 0))
            totals[k] = (n + 1, s + label)
        self.rates = {k: (s + 20 * self.prior) / (n + 20) for k, (n, s) in totals.items()}
        return self

    def predict_proba(self, x):
        p = np.array([self.rates.get(k, self.prior) for k in self.keys(x)])
        return np.column_stack([1 - p, p])


def calibrator(p, y):
    """Strictly increasing Platt map; ranking cannot be inverted by calibration."""
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    def objective(theta):
        a, b = theta
        t = a * z + b
        error = expit(t) - y
        return np.mean(np.logaddexp(0, t) - y * t), np.array([np.mean(error * z), np.mean(error)])
    opt = minimize(objective, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B",
                   bounds=[(1e-3, 50), (-25, 25)], options={"ftol": 1e-12, "maxiter": 500})
    if not opt.success:
        raise RuntimeError(f"Calibration failed: {opt.message}")
    return opt.x


def apply_calibration(p, ab):
    return expit(ab[0] * logit(np.clip(p, 1e-6, 1 - 1e-6)) + ab[1])


def choose(p, rank_score, ids, capacity=CAPACITY, threshold=COST):
    """Optimal equal-cost action set; original score resolves numeric saturation.

    rank_score and p have the same underlying ordering under monotone calibration.
    Stable source-record IDs break score ties. No outcomes are accepted here.
    """
    k = int(np.floor(len(p) * capacity))
    order = np.lexsort((ids, -rank_score))
    eligible = order[p[order] > threshold]
    a = np.zeros(len(p), dtype=np.int8)
    a[eligible[:k]] = 1
    return a


def action_by_fold(p, rank_score, ids, fold, capacity=CAPACITY, threshold=COST):
    result = np.zeros(len(p), dtype=np.int8)
    for f in np.unique(fold):
        ii = np.flatnonzero(fold == f)
        result[ii] = choose(p[ii], rank_score[ii], ids[ii], capacity, threshold)
    return result


def run():
    DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    manifest = dict(seed=SEED, missingness_rates=RATES, corruption_replicates=REPS,
                    primary_cost=COST, primary_capacity=CAPACITY,
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    scipy=scipy.__version__, sklearn=sklearn.__version__, cpu=platform.machine(),
                    protocol="Fixed before first benchmark execution; not externally preregistered.", datasets={})
    for name in SOURCES:
        x, y, ids, categorical, bi_columns, info = read_data(name)
        records = []
        raw_parts = []; cal_parts = []; test_parts = []
        for f, (tr, ca, te) in enumerate(splits(name, y)):
            assert not (set(tr) & set(ca) or set(tr) & set(te) or set(ca) & set(te))
            if name == "bank":
                assert tr.max() < ca.min() and ca.max() < te.min()
            assert "duration" not in x and "PageValues" not in x
            models = [Frequency(bi_columns),
                      make_pipeline(preprocessing(x, categorical), LogisticRegression(C=1, max_iter=1000, random_state=SEED)),
                      make_pipeline(preprocessing(x, categorical), RandomForestClassifier(n_estimators=200, min_samples_leaf=20, max_features="sqrt", n_jobs=2, random_state=SEED)),
                      make_pipeline(preprocessing(x, categorical), HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1, early_stopping=False, random_state=SEED))]
            raw = np.zeros((len(METHODS), len(RATES), REPS, len(te)), dtype=np.float64)
            cal = np.zeros_like(raw)
            # Same nested masks for every estimator; test predictors only are corrupted.
            masks = [np.random.default_rng(SEED + (10000 if name == "shop" else 0) + 100 * f + r).random((len(te), x.shape[1])) for r in range(REPS)]
            foldrec = dict(fold=f, train_n=len(tr), calibration_n=len(ca), test_n=len(te),
                           train_positive=float(y[tr].mean()), calibration_positive=float(y[ca].mean()),
                           test_positive=float(y[te].mean()), models=[])
            for m, model in enumerate(models):
                t0 = time.perf_counter()
                with threadpool_limits(limits=2):
                    model.fit(x.iloc[tr], y[tr])
                pc = model.predict_proba(x.iloc[ca])[:, 1]
                ab = calibrator(pc, y[ca])
                foldrec["models"].append(dict(method=METHODS[m], fit_seconds=time.perf_counter() - t0, platt_slope=float(ab[0]), platt_intercept=float(ab[1])))
                for j, rate in enumerate(RATES):
                    for r in range(REPS if rate else 1):
                        corrupted = x.iloc[te].copy().astype({c: "float64" for c in x if c not in categorical})
                        if rate:
                            corrupted = corrupted.mask(masks[r] < rate)
                        p = model.predict_proba(corrupted)[:, 1]
                        raw[m, j, r] = p
                        cal[m, j, r] = apply_calibration(p, ab)
                    if not rate:
                        raw[m, j, 1:] = raw[m, j, 0]
                        cal[m, j, 1:] = cal[m, j, 0]
                print(f"{name} fold {f + 1}/5 {METHODS[m]} fitted and scored", flush=True)
            records.append(foldrec)
            raw_parts.append(raw); cal_parts.append(cal)
            test_parts.append((te, np.full(len(te), f)))
        ti = np.concatenate([a[0] for a in test_parts]); fold = np.concatenate([a[1] for a in test_parts])
        assert len(np.unique(ti)) == len(ti)
        np.savez_compressed(OUT / f"{name}_predictions.npz", raw=np.concatenate(raw_parts, axis=-1), calibrated=np.concatenate(cal_parts, axis=-1), y=y[ti], row_id=ids[ti], fold=fold)
        info.update(test_n=len(ti), test_positive=int(y[ti].sum()), folds=records)
        manifest["datasets"][name] = info
    manifest["code_sha256"] = digest(__file__)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    report()


def bootstrap_ci(values, fold, name, seed=SEED, n_boot=2000):
    """Conditional paired bootstrap, keeping fitted models and action sets fixed.

    Bank: circular moving blocks of length 100 within each chronological window.
    Shop: row bootstrap inside each held-out fold. Not a retraining uncertainty CI.
    """
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(fold == f) for f in np.unique(fold)]
    means = []
    for _ in range(n_boot):
        total = 0.0
        for ii in groups:
            n = len(ii)
            if name == "bank":
                starts = rng.integers(0, n, int(np.ceil(n / 100)))
                jj = (starts[:, None] + np.arange(100)[None, :]).ravel()[:n] % n
            else:
                jj = rng.integers(0, n, n)
            total += values[ii[jj]].sum()
        means.append(1000 * total / len(values))
    return np.quantile(means, [0.025, 0.975]).tolist()


def row_util(d, model, rate_index, calibrated=True, threshold=COST, capacity=CAPACITY, cost=COST):
    raw = d["raw"][model, rate_index]
    p = d["calibrated" if calibrated else "raw"][model, rate_index]
    acts = np.array([action_by_fold(pp, rr, d["row_id"], d["fold"], capacity, threshold) for pp, rr in zip(p, raw)])
    return (acts * (d["y"][None, :] - cost)).mean(axis=0), acts.mean(axis=0)


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    provenance_path = OUT / "manifest.json"
    provenance = json.loads(provenance_path.read_text())
    if "code_sha256" in provenance:
        provenance["training_code_sha256"] = provenance.pop("code_sha256")
    provenance["report_code_sha256"] = digest(__file__)
    provenance_path.write_text(json.dumps(provenance, indent=2))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    palette = ["#777777", "#0072B2", "#009E73", "#D55E00"]
    datasets = {n: dict(np.load(OUT / f"{n}_predictions.npz")) for n in SOURCES}
    table1 = []; table2 = []; detailed = {"contrasts": {}, "scenarios": [], "invariance": {}}
    for name, d in datasets.items():
        label = "Bank" if name == "bank" else "Shop"
        for m, method in enumerate(METHODS):
            p = d["calibrated"][m, 0, 0]
            u, a = row_util(d, m, 0)
            ci = bootstrap_ci(u, d["fold"], name)
            table1.append(dict(dataset=label, model=method, auc=roc_auc_score(d["y"], p),
                               average_precision=average_precision_score(d["y"], p),
                               brier=brier_score_loss(d["y"], p), utility=1000 * u.mean(),
                               ci_low=ci[0], ci_high=ci[1], selected_percent=100 * a.mean()))
        for calibrated, economic in itertools.product([False, True], repeat=2):
            row = dict(dataset=label, calibration="Yes" if calibrated else "No", cost_rule="Yes" if economic else "No")
            for j, suffix in [(0, "clean"), (2, "missing30")]:
                u, a = row_util(d, 3, j, calibrated, COST if economic else 0.5)
                ps = d["calibrated" if calibrated else "raw"][3, j]
                row[f"brier_{suffix}"] = float(np.mean([(p - d["y"]) ** 2 for p in ps]))
                row[f"utility_{suffix}"] = 1000 * u.mean()
                row[f"selected_{suffix}"] = 100 * a.mean()
            table2.append(row)
        for j, rate in enumerate(RATES):
            u_cal, _ = row_util(d, 3, j, True)
            u_raw, _ = row_util(d, 3, j, False)
            ci = bootstrap_ci(u_cal - u_raw, d["fold"], name)
            detailed["contrasts"][f"{name}_calibration_{rate}"] = dict(delta=1000 * (u_cal - u_raw).mean(), ci=ci)
            for m, method in enumerate(METHODS):
                for c in [0.05, 0.10, 0.20, 0.40]:
                    for q in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
                        u, a = row_util(d, m, j, True, c, q, c)
                        detailed["scenarios"].append(dict(dataset=name, method=method, missing_rate=rate,
                                                         cost=c, capacity=q, utility=1000 * u.mean(), selected=float(a.mean())))
        # Numerically verify the ranking-invariance proposition in every fold/mask.
        checked = 0; matched = 0
        for m, j, r, f in itertools.product(range(4), range(4), range(REPS), range(5)):
            if j == 0 and r > 0:
                continue
            ii = np.flatnonzero(d["fold"] == f)
            p, z = d["calibrated"][m, j, r, ii], d["raw"][m, j, r, ii]
            aa = choose(p, z, d["row_id"][ii]); bb = choose(z, z, d["row_id"][ii])
            k = int(len(ii) * CAPACITY)
            if aa.sum() == k and bb.sum() == k:
                checked += 1
                matched += int(np.array_equal(aa, bb))
        assert checked == matched
        detailed["invariance"][name] = dict(both_fill_capacity=checked, identical_sets=matched,
                                            total_fold_mask_model_checks=4 * (1 + 3 * REPS) * 5,
                                            note="duplicates of clean predictions skipped; diagnostic checks are not independent replications")
        # Explicit paired comparison to every capacity-matched baseline.
        uf, _ = row_util(d, 3, 0)
        for m in range(3):
            ub, _ = row_util(d, m, 0)
            detailed["contrasts"][f"{name}_boost_vs_{m}"] = dict(delta=1000 * (uf - ub).mean(), ci=bootstrap_ci(uf - ub, d["fold"], name))
    t1 = pd.DataFrame(table1); t2 = pd.DataFrame(table2)
    t1.to_csv(OUT / "table1.csv", index=False, float_format="%.10f")
    t2.to_csv(OUT / "table2.csv", index=False, float_format="%.10f")
    (OUT / "analysis.json").write_text(json.dumps(detailed, indent=2))

    def save(fig, stem, png_dpi=600, tight=True):
        for fmt, dpi in [("pdf", None), ("png", png_dpi)]:
            buffer = io.BytesIO()
            fig.savefig(buffer, format=fmt, dpi=dpi, bbox_inches="tight" if tight else None)
            payload = buffer.getvalue()
            assert len(payload) > 1000, f"Empty figure: {stem}.{fmt}"
            if fmt == "pdf":
                assert payload.rstrip().endswith(b"%%EOF"), f"Incomplete PDF: {stem}"
            (OUT / f"{stem}.{fmt}").write_bytes(payload)
        plt.close(fig)

    # Figure 1: budget trade-off and Brier/utility decoupling, both empirical.
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    grid = np.linspace(0.02, 0.50, 25)
    for ax, (name, d), letter in zip(axes, datasets.items(), "ab"):
        for m, method in enumerate(METHODS):
            values = [1000 * row_util(d, m, 0, True, COST, float(q))[0].mean() for q in grid]
            ax.plot(100 * grid, values, color=palette[m], label=method, lw=1.7)
        ax.axvline(20, color="#bbbbbb", lw=0.8, ls="--")
        ax.axhline(0, color="#bbbbbb", lw=0.7)
        ax.set(title=f"({letter}) {'Bank chronological holdouts' if name == 'bank' else 'Shop stratified holdouts'}", xlabel="Capacity (% of cases)", ylabel="Screening utility per 1,000 cases")
        ax.grid(alpha=.18)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, "figure1")

    # Figure 2: strictly same fitted predictor and four decision-layer variants.
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
    fig.subplots_adjust(left=.09, right=.985, bottom=.19, top=.88, wspace=.33)
    variants = [(False, False, "Raw + 0.5 rule", "#777777", "--"),
                (True, False, "Calibrated + 0.5 rule", "#0072B2", "--"),
                (False, True, "Raw + cost rule", "#009E73", "-"),
                (True, True, "Calibrated + cost rule", "#D55E00", "-")]
    for ax, (name, d), letter in zip(axes, datasets.items(), "ab"):
        for cal, eco, label, color, style in variants:
            values = [1000 * row_util(d, 3, j, cal, COST if eco else 0.5)[0].mean() for j in range(4)]
            ax.plot(100 * np.array(RATES), values, label=label, color=color, ls=style, marker="o", ms=3.5, lw=1.6)
        ax.set(title=f"({letter}) {'Bank' if name == 'bank' else 'Shop'} gradient boosting", xlabel="Test predictor cells masked (%)", ylabel="Utility per 1,000 cases")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.18)
        if name == "shop":
            ax.text(.04, .70, "Cost-rule curves overlap", transform=ax.transAxes, fontsize=7, color="#555555")
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    save(fig, "figure2")

    # Separate graphical abstract, deterministic vector shapes and measured numbers.
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"], "pdf.use14corefonts": False})
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off"); ax.set(xlim=(0, 10), ylim=(0, 4))
    boxes = [(0.12, 1.3, 2.6, "OBSERVED BUSINESS DATA", "Bank marketing + online shopping\nSeparate training and holdouts\nTest-only missingness"),
             (3.35, 1.3, 2.8, "EXPLICIT DECISION RULES", "Probability calibration\nBenefit minus action cost\nLimited action capacity"),
             (6.8, 1.3, 3.0, "DECISION QUALITY", "Measure utility and calibration\nCompare matched baselines\nReport uncertainty and failures")]
    for x, y, width, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, y), width, 1.8, boxstyle="round,pad=0.02", fc="#f2f5f7", ec="#405769", lw=1))
        ax.text(x + width/2, y+1.45, title, ha="center", va="center", fontsize=9, weight="bold", color="#213547")
        ax.text(x + width/2, y+.75, body, ha="center", va="center", fontsize=10, linespacing=1.6)
    for start, end in [(2.78, 3.27), (6.23, 6.72)]:
        ax.add_patch(FancyArrowPatch((start, 2.2), (end, 2.2), arrowstyle="-|>", mutation_scale=16, lw=1.4, color="#405769"))
    ax.text(5, 3.65, "Does better calibration produce better business decisions?", ha="center", fontsize=15, weight="bold")
    vals = []
    for label in ["Bank", "Shop"]:
        rr = t2[(t2.dataset==label) & (t2.calibration=="Yes") & (t2.cost_rule=="Yes")].iloc[0]
        vals.append(f"{label}: {rr.utility_clean:.1f} utility units per 1,000 cases")
    ax.text(5, .93, "Gradient boosting with calibration and cost-aware rules", ha="center", fontsize=10)
    ax.text(5, .61, "  |  ".join(vals), ha="center", fontsize=11, color="#164d69")
    ax.text(5, .20, "Retrospective benchmark | assumed costs | no causal profit or readiness claim", ha="center", fontsize=10)
    # Keep the publisher's 2.5:1 aspect ratio instead of tight cropping.
    save(fig, "graphical_abstract", png_dpi=300, tight=False)
    print(t1.to_string(index=False), flush=True)
    print(t2.to_string(index=False), flush=True)


def self_test():
    rng = np.random.default_rng(SEED)
    for n in [5, 7]:
        for _ in range(20):
            p = rng.uniform(0.01, .99, n); ids = np.arange(n); c = .3; q = .5; k = int(n*q)
            a = choose(p, p, ids, q, c)
            actual = np.dot(a, p-c)
            exhaustive = max(np.dot(s, p-c) for s in itertools.product([0,1], repeat=n) if sum(s)<=k)
            assert abs(actual-exhaustive)<1e-12
            # Monotone transformation must preserve top-k membership if cutoff inactive.
            pp = expit(1.7 * logit(p) - .2)
            assert np.array_equal(choose(p,p,ids,q,0), choose(pp,p,ids,q,0))
    assert choose(np.ones(7)*.4,np.ones(7)*.4,np.arange(7),.5,.1).tolist()==[1,1,1,0,0,0,0]
    assert choose(np.array([.1,.09]),np.array([.1,.09]),np.arange(2),1,.1).sum()==0
    for n in SOURCES:
        x,y,ids,cat,bi,info=read_data(n)
        seen=[]
        for tr,ca,te in splits(n,y):
            assert len(set(tr)&set(ca))==len(set(tr)&set(te))==len(set(ca)&set(te))==0
            seen.extend(te.tolist())
        assert len(set(seen))==len(seen)
        assert not ({"duration","PageValues","y","Revenue"}&set(x))
    print("PASS: exact policy optimality, monotone invariance, ties, zero gain, excluded fields, disjoint splits and unique test records.")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true");parser.add_argument("--report",action="store_true");parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test: self_test()
    if args.run: run()
    elif args.report: report()
    elif not args.self_test: parser.print_help()
