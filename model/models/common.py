from pathlib import Path
import json
import numpy as np
import joblib
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
    ConfusionMatrixDisplay
)

BASE_DIR = Path(__file__).resolve().parent

CACHE_PATH = (
    BASE_DIR / "cache" / "ember_v2_sample_5000_seed_42.npz"
)


def load_data():
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy cache: {CACHE_PATH}"
        )

    with np.load(CACHE_PATH) as data:
        print("Các mảng trong cache:", data.files)

        X = data["X"]
        y = data["y"]

    # Chỉ sử dụng nhãn nhị phân 0 và 1
    mask = np.isin(y, [0, 1])
    X = np.asarray(X[mask], dtype=np.float32)
    y = np.asarray(y[mask], dtype=np.int8)

    print("Tổng số mẫu:", len(y))
    print("Số đặc trưng:", X.shape[1])
    print("Phân bố nhãn:", {
        int(label): int(count)
        for label, count in zip(
            *np.unique(y, return_counts=True)
        )
    })

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def evaluate_and_save(name, model, X_val, y_val, train_seconds):
    output_dir = BASE_DIR / "outputs" / name
    model_dir = BASE_DIR / "models"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_val, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision_malware": float(
            precision_score(y_val, y_pred, zero_division=0)
        ),
        "recall_malware": float(
            recall_score(y_val, y_pred, zero_division=0)
        ),
        "f1_malware": float(
            f1_score(y_val, y_pred, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_val, y_proba)),
        "pr_auc": float(average_precision_score(y_val, y_proba)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "train_seconds": float(train_seconds)
    }

    print("\n===== KẾT QUẢ", name, "=====")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nBáo cáo phân loại:")
    print(classification_report(
        y_val,
        y_pred,
        labels=[0, 1],
        target_names=["Lành tính", "Mã độc"],
        zero_division=0
    ))

    with (output_dir / "metrics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    # Lưu ma trận nhầm lẫn
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(
            y_val, y_pred, labels=[0, 1]
        ),
        display_labels=["Lành tính", "Mã độc"]
    )

    display.plot(values_format="d")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    # Lưu mô hình
    model_path = model_dir / f"{name}.joblib"
    joblib.dump(model, model_path)

    print("\nĐã lưu mô hình:", model_path)
    print("Đã lưu báo cáo:", output_dir)