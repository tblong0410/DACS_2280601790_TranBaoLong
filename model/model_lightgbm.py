import time

from lightgbm import LGBMClassifier

from common import load_data, evaluate_and_save


X_train, X_val, y_train, y_val = load_data()

model = LGBMClassifier(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

print("\nBắt đầu huấn luyện LightGBM...")

start = time.time()
model.fit(X_train, y_train)
train_seconds = time.time() - start

print(f"Huấn luyện xong sau {train_seconds:.2f} giây")

evaluate_and_save(
    "lightgbm",
    model,
    X_val,
    y_val,
    train_seconds
)