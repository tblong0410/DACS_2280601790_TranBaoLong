
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve
)


# =========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


CACHE_PATH = (
    BASE_DIR / "cache" / "ember_v2_sample_500000_seed_42.npz"
)

OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"


# =========================================================
# 2. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU
# =========================================================

def load_data():

    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy cache: {CACHE_PATH}"
        )

    print("=" * 65)
    print("ĐANG ĐỌC DỮ LIỆU EMBER")
    print("=" * 65)

    with np.load(CACHE_PATH) as data:

        print("Các mảng trong cache:", data.files)

        X = data["X"]
        y = data["y"]

    if X.ndim != 2:
        raise ValueError("X phải là ma trận 2 chiều.")

    if y.ndim != 1:
        raise ValueError("y phải là vector 1 chiều.")

    if len(X) != len(y):
        raise ValueError(
            "Số mẫu X và y không khớp nhau."
        )

    mask = np.isin(y, [0, 1])

    X = np.asarray(X[mask], dtype=np.float32)
    y = np.asarray(y[mask], dtype=np.int8)

    if len(X) == 0:
        raise ValueError("Không có mẫu hợp lệ.")

    if not np.isfinite(X).all():
        raise ValueError(
            "Dữ liệu X chứa NaN hoặc Infinity. "
            "Hãy kiểm tra và xử lý trước khi train."
        )

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            "Tập dữ liệu cần có đủ hai nhãn 0 và 1."
        )

    labels, counts = np.unique(y, return_counts=True)

    print("\nTHÔNG TIN DỮ LIỆU")
    print("Tổng số mẫu:", f"{len(y):,}")
    print("Số đặc trưng:", X.shape[1])

    print("\nPhân bố nhãn:")

    for label, count in zip(labels, counts):

        name = "Lành tính" if label == 0 else "Mã độc"

        print(f"{name}: {count:,}")

    # Chia tập Train và Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nKÍCH THƯỚC SAU KHI CHIA")
    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)

    return X_train, X_val, y_train, y_val


# =========================================================
# 3. HÀM ĐÁNH GIÁ VÀ LƯU KẾT QUẢ
# =========================================================

def evaluate_and_save(
    name,
    model,
    X_val,
    y_val,
    train_seconds
):

    output_dir = OUTPUT_DIR / name

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 65)
    print("ĐÁNH GIÁ MÔ HÌNH:", name)
    print("=" * 65)

    # -----------------------------------------------------
    # 3.1. Dự đoán
    # -----------------------------------------------------

    y_pred = model.predict(X_val)

    probabilities = model.predict_proba(X_val)

    # Lấy xác suất tương ứng với lớp mã độc = 1
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "Mô hình không có lớp mã độc (label = 1)."
        )

    y_proba = probabilities[:, classes.index(1)]

    # -----------------------------------------------------
    # 3.2. Ma trận nhầm lẫn
    # -----------------------------------------------------

    cm = confusion_matrix(
        y_val,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # -----------------------------------------------------
    # 3.3. Tính toán các chỉ số
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_val, y_pred
    )

    precision = precision_score(
        y_val, y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val, y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val, y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_val, y_proba
    )

    pr_auc = average_precision_score(
        y_val, y_proba
    )

    # Khả năng nhận diện đúng file lành tính
    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0 else 0
    )

    # Tỷ lệ cảnh báo nhầm
    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0 else 0
    )

    # Tỷ lệ bỏ sót mã độc
    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0 else 0
    )

    # -----------------------------------------------------
    # 3.4. Tổng hợp kết quả
    # -----------------------------------------------------

    metrics = {
        "model": name,
        "accuracy": float(accuracy),
        "precision_malware": float(precision),
        "recall_malware": float(recall),
        "f1_malware": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc_average_precision": float(pr_auc),
        "specificity": float(specificity),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "train_seconds": float(train_seconds)
    }

    print("\nCÁC CHỈ SỐ ĐÁNH GIÁ")

    for key, value in metrics.items():

        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
        else:
            print(f"{key:30s}: {value}")

    # -----------------------------------------------------
    # 3.5. Classification Report
    # -----------------------------------------------------

    print("\nBÁO CÁO PHÂN LOẠI")

    report = classification_report(
        y_val,
        y_pred,
        labels=[0, 1],
        target_names=["Lành tính", "Mã độc"],
        zero_division=0
    )

    print(report)

    report_dict = classification_report(
        y_val,
        y_pred,
        labels=[0, 1],
        target_names=["Lành tính", "Mã độc"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).transpose()

    report_df.to_csv(
        output_dir / "classification_report.csv",
        encoding="utf-8-sig"
    )

    # =====================================================
    # 4. VẼ CONFUSION MATRIX
    # =====================================================

    fig, ax = plt.subplots(figsize=(8, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Lành tính", "Mã độc"]
    )

    display.plot(
        ax=ax,
        values_format="d",
        colorbar=True,
        cmap="Blues"
    )

    ax.set_title(
        f"Confusion Matrix - {name}",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Nhãn dự đoán",
        fontsize=12
    )

    ax.set_ylabel(
        "Nhãn thực tế",
        fontsize=12
    )

    plt.tight_layout()

    fig.savefig(
        output_dir / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # Lưu ma trận nhầm lẫn thành CSV
    cm_df = pd.DataFrame(
        cm,
        index=["Thực tế: Lành tính", "Thực tế: Mã độc"],
        columns=["Dự đoán: Lành tính", "Dự đoán: Mã độc"]
    )

    cm_df.to_csv(
        output_dir / "confusion_matrix.csv",
        encoding="utf-8-sig"
    )

    # =====================================================
    # 5. BIỂU ĐỒ CÁC CHỈ SỐ ĐÁNH GIÁ
    # =====================================================

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
        "PR-AUC"
    ]

    metric_values = [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.bar(
        metric_names,
        metric_values
    )

    ax.set_ylim(0, 1.10)

    ax.set_title(
        f"Evaluation Metrics - {name}",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel("Chỉ số đánh giá")
    ax.set_ylabel("Giá trị")

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    for bar, value in zip(bars, metric_values):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()

    fig.savefig(
        output_dir / "evaluation_metrics.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # =====================================================
    # 6. ROC CURVE
    # =====================================================

    fpr, tpr, thresholds = roc_curve(
        y_val,
        y_proba
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC-AUC = {roc_auc:.4f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_title(
        f"ROC Curve - {name}",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    fig.savefig(
        output_dir / "roc_curve.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # =====================================================
    # 7. PRECISION - RECALL CURVE
    # =====================================================

    precisions, recalls, pr_thresholds = precision_recall_curve(
        y_val,
        y_proba
    )

    baseline = np.mean(y_val)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        recalls,
        precisions,
        linewidth=2,
        label=f"Average Precision = {pr_auc:.4f}"
    )

    ax.axhline(
        y=baseline,
        linestyle="--",
        label=f"Baseline = {baseline:.4f}"
    )

    ax.set_title(
        f"Precision-Recall Curve - {name}",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    fig.savefig(
        output_dir / "precision_recall_curve.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # =====================================================
    # 8. VẼ THÊM BIỂU ĐỒ FP VÀ FN
    # =====================================================

    error_names = [
        "False Positive",
        "False Negative"
    ]

    error_values = [
        fp,
        fn
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        error_names,
        error_values
    )

    ax.set_title(
        f"Classification Errors - {name}",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_ylabel("Số lượng mẫu")
    ax.set_xlabel("Loại lỗi")

    for bar, value in zip(bars, error_values):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom"
        )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    fig.savefig(
        output_dir / "classification_errors.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # =====================================================
    # 9. LƯU CÁC CHỈ SỐ JSON
    # =====================================================

    with (output_dir / "metrics.json").open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=4
        )

    # =====================================================
    # 10. LƯU MÔ HÌNH
    # =====================================================

    model_path = MODEL_DIR / f"{name}.joblib"

    joblib.dump(
        model,
        model_path
    )

    # =====================================================
    # 11. THÔNG BÁO KẾT QUẢ
    # =====================================================

    print("\n" + "=" * 65)
    print("HOÀN TẤT ĐÁNH GIÁ:", name)
    print("=" * 65)

    print("Mô hình:", model_path)
    print("Thư mục kết quả:", output_dir)

    print("\nCác file đã lưu:")
    print("- metrics.json")
    print("- classification_report.csv")
    print("- confusion_matrix.csv")
    print("- confusion_matrix.png")
    print("- evaluation_metrics.png")
    print("- roc_curve.png")
    print("- precision_recall_curve.png")
    print("- classification_errors.png")

    return metrics
