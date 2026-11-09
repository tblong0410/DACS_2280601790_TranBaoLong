import time

from xgboost import XGBClassifier

from common import load_data, evaluate_and_save


X_train, X_val, y_train, y_val = load_data()

model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

print("\nBắt đầu huấn luyện XGBoost...")

start = time.time()
model.fit(X_train, y_train)
train_seconds = time.time() - start

print(f"Huấn luyện xong sau {train_seconds:.2f} giây")

evaluate_and_save(
    "xgboost",
    model,
    X_val,
    y_val,
    train_seconds
)