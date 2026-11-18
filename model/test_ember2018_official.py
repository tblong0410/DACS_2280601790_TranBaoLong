from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAMES = ("lightgbm", "random_forest", "xgboost")
DISPLAY_NAMES = {"lightgbm": "LightGBM", "random_forest": "Random Forest", "xgboost": "XGBoost"}
FEATURE_DIM = 2381
FEATURE_GROUPS = (
    ("Byte histogram", 256), ("Byte entropy", 256), ("Strings", 104),
    ("General file", 10), ("PE headers", 62), ("Sections", 255),
    ("Imports", 1280), ("Exports", 128), ("Data directories", 30),
)
assert sum(n for _, n in FEATURE_GROUPS) == FEATURE_DIM


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_plot(fig, target: Path) -> None:
    fig.savefig(target, dpi=175, bbox_inches="tight")
    plt.close(fig)


def cache_source_info(source: Path) -> dict:
    stat = source.stat()
    return {"test_file": str(source.resolve()), "source_bytes": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns, "feature_version": 2,
            "feature_dim": FEATURE_DIM}


def read_cache(source: Path, cache_dir: Path, force: bool = False):
    """Stream JSONL twice into memory-mapped NPY arrays, keeping RAM usage low."""
    if not source.is_file():
        raise FileNotFoundError(
            f"Khong tim thay official test: {source}\n"
            "Hay giai nen EMBER2018 v2 va kiem tra ten file test_features.jsonl.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    x_path = cache_dir / "X.npy"
    y_path = cache_dir / "y.npy"
    meta_path = cache_dir / "cache_info.json"
    provenance = cache_source_info(source)

    if not force and x_path.is_file() and y_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if all(meta.get(k) == v for k, v in provenance.items()):
                X = np.load(x_path, mmap_mode="r", allow_pickle=False)
                y = np.load(y_path, mmap_mode="r", allow_pickle=False)
                if (X.ndim == 2 and X.shape[1] == FEATURE_DIM and y.ndim == 1
                        and len(X) == len(y) == meta["labeled_count"]
                        and np.isin(y, [0, 1]).all()):
                    print(f"[CACHE] Tai su dung cache official test: {len(y):,} mau", flush=True)
                    return X, y, meta
            print("[CACHE] Cache cu khac nguon/khong hop le, tao lai.", flush=True)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            print("[CACHE] Cache bi loi, tao lai.", flush=True)

    # Deliberately count the OFFICIAL test file, not any train_features_*.jsonl.
    print(f"[DATA] Dem mau co nhan trong {source}...", flush=True)
    labeled, unlabeled, empty, total = 0, 0, 0, 0
    with source.open("r", encoding="utf-8-sig") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                empty += 1
                continue
            total += 1
            try:
                label = json.loads(line)["label"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"JSON/label loi dong {lineno}: {exc}") from exc
            if label in (0, 1):
                labeled += 1
            elif label == -1:
                unlabeled += 1
            else:
                raise ValueError(f"Nhan khong hop le tai dong {lineno}: {label!r}")
    if labeled < 2:
        raise ValueError("Official test khong du mau co nhan 0/1.")

    # Import only when cache needs to be generated; this is the same extractor
    # used by the team's create_cache.py script.
    try:
        from ember.features import PEFeatureExtractor
    except ImportError as exc:
        raise ImportError(
            "Chua import duoc 'ember.features.PEFeatureExtractor'. "
        ) from exc

    extractor = PEFeatureExtractor(feature_version=2, print_feature_warning=False)
    if extractor.dim != FEATURE_DIM:
        raise RuntimeError(f"Extractor EMBER v2 can {FEATURE_DIM} chieu, thuc te: {extractor.dim}")

    print(f"[DATA] {total:,} ban ghi; {labeled:,} co nhan; {unlabeled:,} unlabeled; {empty:,} dong trong", flush=True)
    tmp_x = cache_dir / "X.partial.npy"
    tmp_y = cache_dir / "y.partial.npy"
    Xw = yw = None
    try:
        Xw = open_memmap(tmp_x, mode="w+", dtype=np.float32, shape=(labeled, FEATURE_DIM))
        yw = open_memmap(tmp_y, mode="w+", dtype=np.int8, shape=(labeled,))
        pos = 0
        benign = malware = 0
        t0 = time.perf_counter()
        with source.open("r", encoding="utf-8-sig") as handle:
            for lineno, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                label = rec.get("label")
                if label == -1:
                    continue
                try:
                    vec = np.asarray(extractor.process_raw_features(rec), dtype=np.float32)
                except Exception as exc:
                    raise RuntimeError(f"Khong vector hoa duoc dong {lineno}: {exc}") from exc
                if vec.shape != (FEATURE_DIM,) or not np.isfinite(vec).all():
                    raise ValueError(f"Vector khong hop le dong {lineno}: {vec.shape}")
                Xw[pos] = vec
                yw[pos] = int(label)
                benign += int(label == 0)
                malware += int(label == 1)
                pos += 1
                if pos % 10000 == 0:
                    print(f"[CACHE] Da vector hoa {pos:,}/{labeled:,}", flush=True)
        if pos != labeled or not benign or not malware:
            raise ValueError(f"So mau/nhan khong hop le: {pos}/{labeled}, benign={benign}, malware={malware}")
        Xw.flush()
        yw.flush()
        del Xw, yw
        Xw = yw = None
        gc.collect()
        os.replace(tmp_x, x_path)
        os.replace(tmp_y, y_path)
        meta = {**provenance, "original_records": total, "labeled_count": labeled,
                "unlabeled_skipped": unlabeled, "empty_lines_skipped": empty,
                "benign_count": benign, "malware_count": malware,
                "vectorize_seconds": round(time.perf_counter() - t0, 2),
                "source_split": "EMBER2018 official test, NOT train validation"}
        save_json(meta_path, meta)
        print(f"[CACHE] Da luu: {cache_dir.resolve()}", flush=True)
    finally:
        if Xw is not None:
            del Xw
        if yw is not None:
            del yw
        gc.collect()
        # Clean up incomplete temporary cache without touching a previously valid one.
        for p in (tmp_x, tmp_y):
            try:
                p.unlink(missing_ok=True)
            except PermissionError:
                pass

    X = np.load(x_path, mmap_mode="r", allow_pickle=False)
    y = np.load(y_path, mmap_mode="r", allow_pickle=False)
    return X, y, meta


def predict_probs(model, X, batch_size: int) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model khong ho tro predict_proba")
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"classes_={classes}: khong co nhan malware=1")
    positive_index = classes.index(1)
    result = np.empty(X.shape[0], dtype=np.float64)
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        raw = model.predict_proba(X[start:end])
        if raw.ndim != 2 or raw.shape[1] <= positive_index:
            raise ValueError("predict_proba tra ve ket qua sai shape")
        result[start:end] = raw[:, positive_index]
    if not np.isfinite(result).all() or np.any((result < -1e-6) | (result > 1 + 1e-6)):
        raise ValueError("Model dua ra xac suat ngoai [0,1] hoac NaN")
    return np.clip(result, 0, 1)


def calc_metrics(y, pred, prob, seconds, threshold: float) -> dict:
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {
        "dataset": "EMBER2018 official test",
        "n_samples": int(len(y)), "n_benign": int((y == 0).sum()),
        "n_malware": int((y == 1).sum()), "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "average_precision": float(average_precision_score(y, prob)),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "inference_seconds": float(seconds),
    }


def save_importance(model, title: str, out_dir: Path) -> None:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        print(f"[WARN] {title}: model khong co feature_importances_", flush=True)
        return
    imp = np.asarray(values, dtype=np.float64).ravel()
    if imp.shape != (FEATURE_DIM,) or not np.isfinite(imp).all():
        print(f"[WARN] {title}: feature importances khong hop le, bo qua", flush=True)
        return
    feature_groups = []
    group_rows = []
    offset = 0
    for group, n in FEATURE_GROUPS:
        values = imp[offset:offset + n]
        feature_groups.extend([group] * n)
        group_rows.append({"group": group, "feature_count": n,
                           "total_native_importance": float(values.sum()),
                           "mean_native_importance": float(values.mean())})
        offset += n
    df = pd.DataFrame({
        "feature_index_0_based": np.arange(FEATURE_DIM),
        "feature_id": [f"f{i:04d}" for i in range(FEATURE_DIM)],
        "feature_group": feature_groups,
        "native_importance": imp,
    }).sort_values("native_importance", ascending=False)
    df.to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    groups_df = pd.DataFrame(group_rows)
    groups_df.to_csv(out_dir / "feature_importance_groups.csv", index=False, encoding="utf-8-sig")

    top = df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([f"{r.feature_id} ({r.feature_group})" for r in top.itertuples(index=False)],
            top.native_importance.to_numpy())
    ax.set(xlabel="Native feature importance", title=f"{title}: top 20 important features")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    save_plot(fig, out_dir / "feature_importance_top20.png")

    group_plot = groups_df.sort_values("mean_native_importance")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(group_plot.group, group_plot.mean_native_importance)
    ax.set(xlabel="Mean native importance per feature",
           title=f"{title}: feature groups (mean per feature)")
    fig.tight_layout()
    save_plot(fig, out_dir / "feature_importance_groups.png")


def save_individual_plots(y, pred, prob, metrics: dict, title: str, out_dir: Path) -> None:
    cm = confusion_matrix(y, pred, labels=[0, 1])
    pd.DataFrame(cm, index=["Actual benign", "Actual malware"],
                 columns=["Predicted benign", "Predicted malware"]).to_csv(
                     out_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6.7, 5.7))
    ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malware"]).plot(
        ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"{title}: official EMBER2018 confusion matrix")
    save_plot(fig, out_dir / "confusion_matrix.png")

    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "AP"]
    vals = [metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals)
    ax.set(ylim=(0, 1.10), ylabel="Score", title=f"{title}: official test metrics")
    ax.bar_label(bars, labels=[f"{x:.3f}" for x in vals], padding=2)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    save_plot(fig, out_dir / "evaluation_metrics.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["False positive (FP)", "False negative (FN)"], [metrics["fp"], metrics["fn"]])
    ax.bar_label(bars, padding=2)
    ax.set(ylabel="Number of samples", title=f"{title}: classification errors")
    fig.tight_layout()
    save_plot(fig, out_dir / "classification_errors.png")

    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{title}: ROC curve")
    ax.grid(alpha=.2)
    ax.legend()
    save_plot(fig, out_dir / "roc_curve.png")

    precision, recall, _ = precision_recall_curve(y, prob)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(recall, precision, label=f"Average Precision = {metrics['average_precision']:.4f}")
    ax.axhline(float(np.mean(y)), ls="--", color="gray", label="Malware prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title=f"{title}: precision-recall curve")
    ax.grid(alpha=.2)
    ax.legend()
    save_plot(fig, out_dir / "precision_recall_curve.png")

    report = classification_report(y, pred, labels=[0, 1], target_names=["Benign", "Malware"],
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out_dir / "classification_report.csv", encoding="utf-8-sig")
    (out_dir / "classification_report.txt").write_text(
        classification_report(y, pred, labels=[0, 1], target_names=["Benign", "Malware"],
                              zero_division=0), encoding="utf-8")


def evaluate_one(model_key: str, model_dir: Path, X, y, output_dir: Path,
                 threshold: float, batch_size: int) -> dict:
    model_path = model_dir / f"{model_key}.joblib"
    title = DISPLAY_NAMES[model_key]
    print(f"\n[MODEL] {title}: {model_path}", flush=True)
    model = joblib.load(model_path)  # Load only team-owned, trusted joblib files.
    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None and int(n_features) != FEATURE_DIM:
        raise ValueError(f"{title}: model co {n_features} features, expected {FEATURE_DIM}")
    t0 = time.perf_counter()
    prob = predict_probs(model, X, batch_size)
    seconds = time.perf_counter() - t0
    pred = (prob >= threshold).astype(np.int8)
    metrics = calc_metrics(y, pred, prob, seconds, threshold)
    metrics.update({"model": title, "model_path": str(model_path.resolve())})
    out = output_dir / model_key
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "metrics.json", metrics)
    save_individual_plots(y, pred, prob, metrics, title, out)
    save_importance(model, title, out)
    pd.DataFrame({"sample_index": np.arange(len(y)), "actual_label": y,
                  "predicted_label": pred, "malware_probability": prob,
                  "correct": pred == y}).to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    print(f"[DONE] {title}: accuracy={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} "
          f"recall={metrics['recall']:.4f} FPR={metrics['fpr']:.4f} "
          f"inference={seconds:.1f}s", flush=True)
    del model, prob, pred
    gc.collect()
    return metrics


def plot_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    names = [("accuracy", "Accuracy"), ("precision", "Precision"),
             ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC-AUC"),
             ("average_precision", "AP")]
    xs = np.arange(len(df))
    width = .13
    fig, ax = plt.subplots(figsize=(13.5, 6))
    for i, (col, label) in enumerate(names):
        ax.bar(xs + (i - 2.5) * width, df[col], width, label=label)
    ax.set_xticks(xs, df.model)
    ax.set_ylim(0, 1.06)
    ax.set(ylabel="Score (0-1)", title="3 models: EMBER2018 official test")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, -.10))
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    save_plot(fig, output_dir / "comparison_metrics.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    w = .33
    ax.bar(xs - w/2, df.fpr * 100, w, label="FPR (%)")
    ax.bar(xs + w/2, df.fnr * 100, w, label="FNR (%)")
    ax.set_xticks(xs, df.model)
    ax.set(ylabel="Percent", title="EMBER2018 official test: FPR / FNR")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    save_plot(fig, output_dir / "comparison_error_rates.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(xs - w/2, df.fp, w, label="False positives")
    ax.bar(xs + w/2, df.fn, w, label="False negatives")
    ax.set_xticks(xs, df.model)
    ax.set(ylabel="Number of samples", title="EMBER2018 official test: FP / FN counts")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    save_plot(fig, output_dir / "comparison_fp_fn.png")


def compare_with_validation(df: pd.DataFrame, validation_outputs: Path, output_dir: Path) -> bool:
    names = {"LightGBM": "lightgbm", "Random Forest": "random_forest", "XGBoost": "xgboost"}
    keys = {
        "accuracy": "accuracy", "precision": "precision_malware",
        "recall": "recall_malware", "f1": "f1_malware",
        "roc_auc": "roc_auc", "fpr": "false_positive_rate",
    }
    rows = []
    for _, entry in df.iterrows():
        path = validation_outputs / names[entry["model"]] / "metrics.json"
        if not path.exists():
            continue
        try:
            validation = json.loads(path.read_text(encoding="utf-8"))
            for test_key, val_key in keys.items():
                if val_key in validation:
                    rows.append({"model": entry["model"], "metric": test_key,
                                 "validation": float(validation[val_key]),
                                 "official_test": float(entry[test_key]),
                                 "delta_test_minus_val": float(entry[test_key] - validation[val_key])})
        except (ValueError, OSError, TypeError) as exc:
            print(f"[WARN] Khong doc duoc {path}: {exc}")
    if not rows:
        return False
    val_df = pd.DataFrame(rows)
    val_df.to_csv(output_dir / "validation_vs_official.csv", index=False, encoding="utf-8-sig")
    accuracy = val_df[val_df.metric == "accuracy"].copy()
    if not accuracy.empty:
        xs = np.arange(len(accuracy))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(xs - .18, accuracy.validation * 100, .36, label="Validation (random train split)")
        ax.bar(xs + .18, accuracy.official_test * 100, .36, label="Official EMBER2018 test")
        ax.set_xticks(xs, accuracy.model)
        ax.set(ylabel="Accuracy (%)", title="Validation vs independent official test")
        ax.legend()
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        save_plot(fig, output_dir / "validation_vs_official_accuracy.png")
    return True


def main():
    parser = argparse.ArgumentParser(description="Official EMBER2018 v2 test for 3 saved models; no training")
    parser.add_argument("--test-file", type=Path,
                        default=BASE_DIR.parent / "ember2018" / "test_features.jsonl")
    parser.add_argument("--models-dir", type=Path, default=BASE_DIR / "models")
    parser.add_argument("--cache-dir", type=Path, default=BASE_DIR / "cache" / "ember2018_official_test")
    parser.add_argument("--output-dir", type=Path,
                        default=BASE_DIR / "output_test" / "ember2018_official")
    parser.add_argument("--validation-dir", type=Path, default=BASE_DIR / "outputs")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold phai trong khoang (0, 1)")
    if args.batch_size < 1:
        parser.error("--batch-size phai >= 1")
    missing = [args.models_dir / f"{name}.joblib" for name in MODEL_NAMES
               if not (args.models_dir / f"{name}.joblib").is_file()]
    if missing:
        raise FileNotFoundError("Thieu model da train:\n" + "\n".join(str(p) for p in missing))

    print("=" * 70)
    print("EMBER2018 OFFICIAL TEST (NOT 80/20 VALIDATION)")
    print("No fit(), no model overwrite, no threshold search.")
    print("=" * 70)
    X, y, meta = read_cache(args.test_file, args.cache_dir, args.rebuild_cache)
    print(f"[TEST] X={X.shape} | benign={(y==0).sum():,} | malware={(y==1).sum():,}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "run_info.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "EMBER2018 official test_features.jsonl (v2)",
        "test_source": str(args.test_file.resolve()),
        "train_origin": "EMBER2018 train_features_0..5.jsonl only (500k sample cache)",
        "random_validation_is_not_official_test": True,
        "feature_version": 2, "feature_dim": FEATURE_DIM,
        "total_test_samples": int(len(y)),
        "benign_count": int((y == 0).sum()), "malware_count": int((y == 1).sum()),
        "threshold_fixed_in_advance": args.threshold,
        "batch_size": args.batch_size,
        "test_cache_metadata": meta,
        "note": "Keep test held out: do not tune or retrain using these results.",
    })

    rows = []
    for name in MODEL_NAMES:
        rows.append(evaluate_one(name, args.models_dir, X, y, output_dir,
                                 args.threshold, args.batch_size))
    df = pd.DataFrame(rows)
    ordered = ["model", "n_samples", "accuracy", "precision", "recall", "f1",
               "roc_auc", "average_precision", "balanced_accuracy", "specificity",
               "fpr", "fnr", "tn", "fp", "fn", "tp", "inference_seconds", "threshold"]
    df[ordered].to_csv(output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    plot_comparison(df, output_dir)
    found_val = compare_with_validation(df, args.validation_dir, output_dir)
    print("\n" + df[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "fpr", "fnr"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    lines = [
        "# EMBER2018 — Official Test Results", "",
        "These numbers come from the **official `test_features.jsonl`** held-out split.",
        "The models were previously trained on EMBER2018 training data; no retraining was performed.", "",
        f"- Samples: {len(y):,}; benign: {(y==0).sum():,}; malware: {(y==1).sum():,}",
        f"- Fixed threshold: {args.threshold} (selected before test)",
        "- Feature extractor: EMBER v2, 2381 dimensions", "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append("| " + r["model"] + " | " + " | ".join(
            f"{100*float(r[k]):.2f}%" for k in ("accuracy", "precision", "recall", "f1",
                                                      "roc_auc", "fpr", "fnr")) + " |")
    lines.extend(["", "![Model comparison](comparison_metrics.png)",
                  "", "![FPR and FNR comparison](comparison_error_rates.png)",
                  "", "## Interpretation", "",
                  "- Compare Recall/FNR to quantify missed malware, and FPR to quantify false alarms.",
                  "- A difference from validation scores reflects evaluation on a separate held-out split.",
                  "- Native feature importance scales/methods differ between the 3 algorithms.",
                  "- Never tune a threshold, features or hyperparameters against the official test.",
                  ""])
    if found_val:
        lines.extend(["## Validation vs Official test", "",
                      "![Validation vs test](validation_vs_official_accuracy.png)", ""])
    (output_dir / "REPORT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OUTPUT] {output_dir.resolve()}\n[REPORT] {output_dir / 'REPORT_SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
