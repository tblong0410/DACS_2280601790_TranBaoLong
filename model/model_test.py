from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve, balanced_accuracy_score,
)

MODEL_NAMES = ("lightgbm", "random_forest", "xgboost")
DISPLAY_NAMES = {
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
BASE = Path(__file__).resolve().parent

FEATURE_GROUPS = (
    ("Byte histogram", 256),
    ("Byte entropy", 256),
    ("Strings", 104),
    ("General file", 10),
    ("PE headers", 62),
    ("Sections", 255),
    ("Imports", 1280),
    ("Exports", 128),
    ("Data directories", 30),
)
assert sum(count for _, count in FEATURE_GROUPS) == 2381


def store_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_png(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=175, bbox_inches="tight")
    plt.close(fig)


def load_dataset(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Khong tim thay BODMAS: {path}")
    print(f"[DATA] Dang doc {path} ...", flush=True)
    with np.load(path, allow_pickle=False) as pack:
        if not {"X", "y"}.issubset(pack.files):
            raise ValueError(f"File npz phai chua X va y; thuc te: {pack.files}")
        X = np.asarray(pack["X"], dtype=np.float32, order="C")
        y = np.asarray(pack["y"])
    if X.ndim != 2 or X.shape[1] != 2381:
        raise ValueError(f"Can EMBER v2: 2381 chieu; nhan X={X.shape}")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError(f"Nhan y khong dung: X={X.shape}, y={y.shape}")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("BODMAS test phai co nhan 0=benign, 1=malware (khong tu bo mau)")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("Can ca hai lop 0 va 1 de tinh ROC-AUC")
    for start in range(0, len(X), 8192):
        if not np.isfinite(X[start:start + 8192]).all():
            raise ValueError(f"X co NaN/Inf gan dong {start}")
    y = y.astype(np.int8, copy=False)
    print(f"[DATA] {len(y):,} mau, {X.shape[1]} features | benign={(y==0).sum():,} | malware={(y==1).sum():,}", flush=True)
    return X, y


def load_metadata(path: Path, n_rows: int):
    if not path.is_file():
        print("[META] Khong co metadata.csv - bo qua phan theo thoi gian.")
        return None, None
    frame = pd.read_csv(path, low_memory=False)
    if len(frame) != n_rows:
        print(f"[META] SO DONG KHONG KHOP ({len(frame)} vs {n_rows}); bo qua metadata de tranh gan sai mau.")
        return None, None
    print(f"[META] {list(frame.columns)}")
    lower = {str(col).lower().strip(): col for col in frame.columns}
    sha_col = next((col for key, col in lower.items() if "sha256" in key or "sha_256" in key), None)
    date_col = next((col for key, col in lower.items()
                     if any(token in key for token in ("timestamp", "first_seen", "firstseen", "appeared", "date", "time"))), None)
    if date_col:
        series = pd.to_datetime(frame[date_col], errors="coerce", utc=True)
        if series.notna().mean() >= 0.9:
            frame["__test_month"] = series.dt.strftime("%Y-%m")
            print(f"[META] Co thoi gian tu cot: {date_col}")
        else:
            print(f"[META] Khong doc duoc >90% cot ngay {date_col}, bo qua phan thoi gian.")
    return frame, sha_col


def batched_proba(model, X: np.ndarray, batch_size: int) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model khong co predict_proba")
    labels = list(model.classes_)
    if 1 not in labels:
        raise ValueError(f"Model classes_={labels} khong chua nhan malware=1")
    idx = labels.index(1)
    probs = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        p = model.predict_proba(X[start:stop])
        if p.ndim != 2 or p.shape[1] <= idx:
            raise ValueError(f"predict_proba co shape bat thuong: {p.shape}")
        probs[start:stop] = p[:, idx]
    if not np.isfinite(probs).all() or (probs < -1e-7).any() or (probs > 1 + 1e-7).any():
        raise ValueError("Model tra ve xac suat khong hop le")
    return np.clip(probs, 0, 1)


def calculate_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray, seconds: float) -> dict:
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {
        "n_samples": int(len(y)),
        "benign": int(np.sum(y == 0)),
        "malware": int(np.sum(y == 1)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "fpr": fp / (tn + fp) if tn + fp else 0.0,
        "fnr": fn / (tp + fn) if tp + fn else 0.0,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "inference_seconds": float(seconds),
    }


def plot_model_metrics(metrics: dict, name: str, out: Path) -> None:
    names = ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Avg Precision")
    values = [metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")]
    fig, ax = plt.subplots(figsize=(9.8, 5.5))
    bars = ax.bar(names, values, color="#2366a7")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score (0-1)")
    ax.set_title(f"{name} - BODMAS evaluation")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=12)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val+.015, f"{val:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    save_png(fig, out / "evaluation_metrics.png")


def plot_errors(metrics: dict, name: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    bars = ax.bar(["False positive (FP)", "False negative (FN)"],
                  [metrics["fp"], metrics["fn"]], color=["#e09c35", "#c75252"])
    ax.bar_label(bars, padding=4, fmt="%d")
    ax.set_ylim(0, max(1, metrics["fp"], metrics["fn"]) * 1.18)
    ax.set_ylabel("Number of files")
    ax.set_title(f"{name} - classification errors")
    fig.tight_layout()
    save_png(fig, out / "classification_errors.png")


def plot_curves(y: np.ndarray, score: np.ndarray, metrics: dict, name: str, out: Path) -> None:
    fpr, tpr, _ = roc_curve(y, score)
    fig, ax = plt.subplots(figsize=(6.6, 5.5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {metrics['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{name} - ROC curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=.2)
    save_png(fig, out / "roc_curve.png")

    precision, recall, _ = precision_recall_curve(y, score)
    fig, ax = plt.subplots(figsize=(6.6, 5.5))
    ax.plot(recall, precision, lw=2, label=f"AP = {metrics['average_precision']:.4f}")
    ax.axhline(np.mean(y), color="gray", ls="--", label=f"Malware base rate = {np.mean(y):.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title=f"{name} - Precision-Recall curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=.2)
    save_png(fig, out / "precision_recall_curve.png")


def save_importances(model, name: str, out: Path) -> None:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        print(f"[WARN] {name}: model khong co feature_importances_, bo qua")
        return
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) != 2381:
        print(f"[WARN] {name}: feature_importances_ co {len(values)} chieu, bo qua")
        return
    if not np.isfinite(values).all():
        print(f"[WARN] {name}: feature_importances_ co gia tri khong hop le, bo qua")
        return

    group_labels = ["unknown"] * 2381
    group_records = []
    offset = 0
    for group_name, count in FEATURE_GROUPS:
        chunk = values[offset:offset+count]
        group_labels[offset:offset+count] = [group_name] * count
        group_records.append({"feature_group": group_name, "n_features": count,
                              "importance_sum": float(chunk.sum()),
                              "importance_mean": float(chunk.mean())})
        offset += count

    idx = np.arange(2381)
    df = pd.DataFrame({
        "feature_index_0_based": idx,
        "feature_id": [f"f{i:04d}" for i in idx],
        "feature_group": group_labels,
        "importance_native": values,
    }).sort_values("importance_native", ascending=False)
    df.to_csv(out / "feature_importance.csv", index=False, encoding="utf-8-sig")
    groups = pd.DataFrame(group_records)
    groups.to_csv(out / "feature_importance_groups.csv", index=False, encoding="utf-8-sig")

    top = df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.4, 7))
    labels = [f"{r.feature_id} ({r.feature_group})" for r in top.itertuples(index=False)]
    ax.barh(labels, top["importance_native"].to_numpy(), color="#31828e")
    ax.set(title=f"{name} - top 20 important features", xlabel="Native feature importance (within model)")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    save_png(fig, out / "feature_importance_top20.png")

    ordered = groups.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(ordered["feature_group"], ordered["importance_mean"], color="#7e6baf")
    ax.set(title=f"{name} - average importance by feature group", xlabel="Mean native importance per feature")
    fig.tight_layout()
    save_png(fig, out / "feature_importance_groups.png")


def save_monthly_metrics(meta, y, pred, score, name: str, out: Path) -> None:
    if meta is None or "__test_month" not in meta:
        return
    frame = pd.DataFrame({"month": meta["__test_month"], "y": y, "pred": pred, "score": score})
    records = []
    for month, group in frame.groupby("month", sort=True):
        if pd.isna(month) or len(group) < 10:
            continue
        t = group["y"].to_numpy()
        p = group["pred"].to_numpy()
        pos = int((t == 1).sum())
        neg = int((t == 0).sum())
        recall = float(((t == 1) & (p == 1)).sum() / pos) if pos else np.nan
        fpr = float(((t == 0) & (p == 1)).sum() / neg) if neg else np.nan
        records.append({"month": month, "n_samples": len(group), "n_malware": pos,
                        "n_benign": neg, "recall": recall, "fpr": fpr})
    if not records:
        return
    monthly = pd.DataFrame(records)
    monthly.to_csv(out / "monthly_metrics.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(monthly["month"], monthly["recall"], marker="o", ms=3, label="Malware recall")
    ax.plot(monthly["month"], monthly["fpr"], marker="o", ms=3, label="False positive rate")
    ax.set(xlabel="First-seen month", ylabel="Rate", title=f"{name} - BODMAS temporal performance")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=.25)
    ax.legend()
    ax.tick_params(axis="x", rotation=55)
    fig.tight_layout()
    save_png(fig, out / "monthly_recall_fpr.png")


def evaluate_model(name: str, model_path: Path, X, y, metadata, sha_col, out_dir: Path,
                   threshold: float, batch_size: int) -> dict:
    if not model_path.is_file():
        raise FileNotFoundError(f"Thieu model: {model_path}. Hay train truoc hoac kiem tra thu muc models.")
    title = DISPLAY_NAMES[name]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[MODEL] {title}: nap {model_path.name}", flush=True)
    model = joblib.load(model_path)  # Only load YOUR OWN trusted joblib files.
    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None and int(n_features) != X.shape[1]:
        raise ValueError(f"{title}: model {n_features} features, BODMAS {X.shape[1]} features")
    start = time.perf_counter()
    score = batched_proba(model, X, batch_size)
    seconds = time.perf_counter() - start
    pred = (score >= threshold).astype(np.int8)
    metrics = calculate_metrics(y, pred, score, seconds)
    metrics.update({"model": title, "dataset": "BODMAS external test (2019-2020)",
                    "threshold": threshold, "source_model": str(model_path.resolve())})
    store_json(out_dir / "metrics.json", metrics)

    report = classification_report(y, pred, labels=[0, 1], target_names=["Benign", "Malware"],
                                   zero_division=0)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    report_dict = classification_report(y, pred, labels=[0, 1], target_names=["Benign", "Malware"],
                                        zero_division=0, output_dict=True)
    pd.DataFrame(report_dict).T.to_csv(out_dir / "classification_report.csv", encoding="utf-8-sig")
    cm = confusion_matrix(y, pred, labels=[0, 1])
    pd.DataFrame(cm, index=["Actual benign", "Actual malware"],
                 columns=["Pred benign", "Pred malware"]).to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malware"]).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"{title} - BODMAS confusion matrix")
    save_png(fig, out_dir / "confusion_matrix.png")
    plot_model_metrics(metrics, title, out_dir)
    plot_errors(metrics, title, out_dir)
    plot_curves(y, score, metrics, title, out_dir)
    save_importances(model, title, out_dir)

    predictions = pd.DataFrame({"index": np.arange(len(y)), "y_true": y, "y_pred": pred,
                                "p_malware": score, "correct": pred == y})
    if metadata is not None and sha_col is not None:
        predictions.insert(1, "sha256", metadata[sha_col].to_numpy())
    if metadata is not None and "__test_month" in metadata:
        predictions["month"] = metadata["__test_month"].to_numpy()
    predictions.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    save_monthly_metrics(metadata, y, pred, score, title, out_dir)
    print(f"[DONE] {title}: accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
          f"recall={metrics['recall']:.4f}, FPR={metrics['fpr']:.4f} | {seconds:.2f}s", flush=True)
    del predictions, model, score, pred
    gc.collect()
    return metrics


def comparison_plots(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [("accuracy", "Accuracy"), ("precision", "Precision"),
               ("recall", "Recall"), ("f1", "F1"),
               ("roc_auc", "ROC-AUC"), ("average_precision", "Avg Precision")]
    xs = np.arange(len(df))
    width = .13
    fig, ax = plt.subplots(figsize=(13, 6))
    for j, (key, label) in enumerate(metrics):
        ax.bar(xs + (j - 2.5) * width, df[key], width=width, label=label)
    ax.set_xticks(xs, df["model"])
    ax.set_ylim(0, 1.05)
    ax.set(ylabel="Score (0-1)", title="EMBER2018-trained models: External test on BODMAS")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, bbox_to_anchor=(.5, -0.13), loc="upper center")
    fig.tight_layout()
    save_png(fig, out_dir / "comparison_metrics.png")

    fig, ax = plt.subplots(figsize=(9, 5.3))
    width = .33
    ax.bar(xs-width/2, df["fpr"] * 100, width=width, label="FPR (%)", color="#e09c35")
    ax.bar(xs+width/2, df["fnr"] * 100, width=width, label="FNR (%)", color="#c75252")
    ax.set_xticks(xs, df["model"])
    ax.set(ylabel="Percent (%)", title="Error-rate comparison on BODMAS")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_png(fig, out_dir / "comparison_error_rates.png")

    fig, ax = plt.subplots(figsize=(9, 5.3))
    ax.bar(xs-width/2, df["fp"], width=width, label="False positives", color="#e09c35")
    ax.bar(xs+width/2, df["fn"], width=width, label="False negatives", color="#c75252")
    ax.set_xticks(xs, df["model"])
    ax.set(ylabel="Number of files", title="Classification errors across 3 models")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_png(fig, out_dir / "comparison_fp_fn.png")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate 3 EMBER2018 v2 models on BODMAS")
    p.add_argument("--data", type=Path, default=BASE.parent / "BODMAS" / "bodmas.npz")
    p.add_argument("--metadata", type=Path, default=None, help="Defaults to bodmas_metadata.csv next to --data")
    p.add_argument("--models", type=Path, default=BASE / "models")
    p.add_argument("--output", type=Path, default=BASE / "output_test" / "bodmas")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold fixed BEFORE examining test results")
    p.add_argument("--batch-size", type=int, default=8192)
    args = p.parse_args()
    if not 0 < args.threshold < 1:
        p.error("--threshold phai nam trong khoang (0, 1)")
    if args.batch_size < 1:
        p.error("--batch-size phai >= 1")
    metadata_path = args.metadata if args.metadata is not None else args.data.parent / "bodmas_metadata.csv"
    print("=== EXTERNAL TEST: EMBER2018 -> BODMAS ===")
    print("Models are used read-only; no fitting, no retraining, no threshold tuning.")
    for model_name in MODEL_NAMES:
        path = args.models / f"{model_name}.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Chua co model: {path} (can model da huan luyen)")
    X, y = load_dataset(args.data)
    meta, sha_col = load_metadata(metadata_path, len(y))
    args.output.mkdir(parents=True, exist_ok=True)
    store_json(args.output / "run_info.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Original BODMAS; independent external test of EMBER2018 models",
        "data_file": str(args.data.resolve()),
        "metadata_file": str(metadata_path.resolve()) if meta is not None else None,
        "training_dataset": "EMBER2018 500k-sample cache (as described in original model/common.py)",
        "split": "BODMAS used only for inference; NO train/test random split",
        "test_sample_count": int(len(y)),
        "test_label_counts": {"benign": int((y==0).sum()), "malware": int((y==1).sum())},
        "n_features": int(X.shape[1]), "feature_version": "EMBER v2 (expected)",
        "threshold": args.threshold, "batch_size": args.batch_size,
        "warning": "2381 shape alone does not prove identical feature ordering; verify extraction pipeline. This external test does not replace EMBER2018 official test.",
    })
    metrics_list = []
    for model_name in MODEL_NAMES:
        result = evaluate_model(model_name, args.models / f"{model_name}.joblib", X, y,
                                meta, sha_col, args.output / model_name, args.threshold, args.batch_size)
        metrics_list.append(result)

    df = pd.DataFrame(metrics_list)
    columns = ["model", "n_samples", "accuracy", "balanced_accuracy", "precision", "recall",
               "f1", "roc_auc", "average_precision", "specificity", "fpr", "fnr",
               "tn", "fp", "fn", "tp", "inference_seconds", "threshold"]
    df[columns].to_csv(args.output / "comparison.csv", index=False, encoding="utf-8-sig")
    comparison_plots(df, args.output)

    display = df[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "fpr", "fnr"]]
    print("\n" + display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    best = df.loc[df["f1"].idxmax(), "model"]
    lines = [
        "# External test: EMBER2018 -> BODMAS", "",
        f"- Samples: {len(y):,}; benign: {(y==0).sum():,}; malware: {(y==1).sum():,}",
        f"- Fixed prediction threshold: {args.threshold}",
        "- No fit() or threshold calibration was performed on BODMAS.",
        "- This is EXTERNAL test, **not** the official EMBER2018 test split.",
        "- Feature-version / feature-order consistency must be checked beyond dimensionality.",
        "- Native feature importances should only be compared within each model.",
        "", "## Results (percent)", "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append("| " + str(r["model"]) + " | " + " | ".join(
            f"{100*r[key]:.2f}%" for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "fpr", "fnr")) + " |")
    lines.extend(["", f"- Highest F1 in this test: **{best}** (not necessarily best for deployment).",
                  "- Inspect FP/FN, calibration, and temporal trends before drawing conclusions.",
                  "- If the team previously tuned a threshold on EMBER2018 validation, use that locked threshold and report it; do not tune using BODMAS.",
                  ""])
    (args.output / "REPORT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OUTPUT] {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
