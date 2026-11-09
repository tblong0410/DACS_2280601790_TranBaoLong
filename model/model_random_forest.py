import time

from sklearn.ensemble import RandomForestClassifier

from common import load_data, evaluate_and_save


X_train, X_val, y_train, y_val = load_data()

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=30,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

print("\nBắt đầu huấn luyện Random Forest...")

start = time.time()
model.fit(X_train, y_train)
train_seconds = time.time() - start

print(f"Huấn luyện xong sau {train_seconds:.2f} giây")

evaluate_and_save(
    "random_forest",
    model,
    X_val,
    y_val,
    train_seconds
)