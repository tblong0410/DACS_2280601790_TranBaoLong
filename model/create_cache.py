
import json
import random
import time

import numpy as np

from pathlib import Path
from collections import Counter
from tqdm import tqdm
from ember.features import PEFeatureExtractor


DATA_DIR = Path(r"D:\DACS\ember2018")

BASE_DIR = Path(__file__).resolve().parent

CACHE_DIR = BASE_DIR / "cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = (
    CACHE_DIR / "ember_v2_sample_500000_seed_42.npz"
)

MAX_SAMPLES = 500000
RANDOM_STATE = 42
FEATURE_VERSION = 2

TRAIN_FILES = [
    DATA_DIR / f"train_features_{i}.jsonl"
    for i in range(6)
]


if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Không tìm thấy thư mục dữ liệu: {DATA_DIR}"
    )

for file_path in TRAIN_FILES:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {file_path}"
        )


print("Thư mục dữ liệu:", DATA_DIR)
print("Số file train:", len(TRAIN_FILES))
print("Số mẫu cần lấy:", f"{MAX_SAMPLES:,}")
print("Feature Version:", FEATURE_VERSION)



rng = random.Random(RANDOM_STATE)

selected_refs = []

total_labeled = 0
bad_json = 0

file_line_counts = []

start_time = time.time()

for file_index, file_path in enumerate(TRAIN_FILES):

    print(f"\nĐang đọc: {file_path.name}")

    file_line_count = 0

    with file_path.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(
            tqdm(f, desc=file_path.name, unit="dòng")
        ):

            file_line_count += 1

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                bad_json += 1
                continue

            label = record.get("label")

            if label not in (0, 1):
                continue

            total_labeled += 1

            reference = (file_index, line_number)

            if len(selected_refs) < MAX_SAMPLES:

                selected_refs.append(reference)

            else:

                j = rng.randrange(total_labeled)

                if j < MAX_SAMPLES:
                    selected_refs[j] = reference

    file_line_counts.append(file_line_count)

print("\nHoàn thành lấy mẫu.")
print("Tổng mẫu có nhãn:", f"{total_labeled:,}")
print("Số mẫu được chọn:", f"{len(selected_refs):,}")
print("Số dòng JSON lỗi:", bad_json)

if len(selected_refs) < MAX_SAMPLES:
    raise RuntimeError(
        f"Chỉ tìm được {len(selected_refs)} mẫu, "
        f"không đủ {MAX_SAMPLES} mẫu."
    )



extractor = PEFeatureExtractor(
    feature_version=FEATURE_VERSION,
    print_feature_warning=True
)

print("Số chiều đặc trưng:", extractor.dim)

if extractor.dim != 2381:
    raise RuntimeError(
        f"Số chiều không đúng: {extractor.dim}"
    )


selected_by_file = {
    i: set()
    for i in range(len(TRAIN_FILES))
}

for file_index, line_number in selected_refs:
    selected_by_file[file_index].add(line_number)


X = np.empty(
    (MAX_SAMPLES, extractor.dim),
    dtype=np.float32
)

y = np.empty(
    MAX_SAMPLES,
    dtype=np.int8
)

written = 0
bad_features = 0

feature_label_counts = Counter()



for file_index, file_path in enumerate(TRAIN_FILES):

    selected_lines = selected_by_file[file_index]

    if not selected_lines:
        continue

    print(f"\nĐang vector hóa: {file_path.name}")

    with file_path.open("r", encoding="utf-8") as f:

        iterator = enumerate(
            tqdm(
                f,
                total=file_line_counts[file_index],
                desc="Vector hóa",
                unit="dòng"
            )
        )

        for line_number, line in iterator:

            if line_number not in selected_lines:
                continue

            try:

                record = json.loads(line)

                label = int(record["label"])

                vector = extractor.process_raw_features(
                    record
                )

                vector = np.asarray(
                    vector,
                    dtype=np.float32
                )

                if vector.shape != (extractor.dim,):
                    raise ValueError(
                        f"Sai kích thước vector: {vector.shape}"
                    )

                if not np.isfinite(vector).all():
                    raise ValueError(
                        "Vector chứa NaN hoặc Infinity."
                    )

                X[written] = vector
                y[written] = label

                written += 1

                feature_label_counts[label] += 1

            except Exception as e:

                bad_features += 1

                if bad_features <= 5:

                    print(
                        f"\nLỗi mẫu tại {file_path.name}, "
                        f"dòng {line_number}: "
                        f"{type(e).__name__}: {e}"
                    )



print("\n" + "=" * 65)
print("KẾT QUẢ VECTOR HÓA")
print("=" * 65)

print("Số mẫu thành công:", f"{written:,}")
print("Số mẫu lỗi:", f"{bad_features:,}")

if written == 0:
    raise RuntimeError(
        "Không vector hóa được mẫu nào."
    )

if written < MAX_SAMPLES:
    print(
        f"CẢNH BÁO: Chỉ vector hóa thành công "
        f"{written:,}/{MAX_SAMPLES:,} mẫu."
    )

X = X[:written].copy()
y = y[:written].copy()

print("\nKích thước X:", X.shape)
print("Kích thước y:", y.shape)

print("\nPhân bố nhãn:")

for label, count in sorted(feature_label_counts.items()):

    name = "Lành tính" if label == 0 else "Mã độc"

    print(f"{name}: {count:,}")

if set(np.unique(y)) != {0, 1}:
    raise RuntimeError(
        "Dữ liệu không có đủ hai lớp lành tính và mã độc."
    )

if not np.isfinite(X).all():
    raise RuntimeError(
        "Ma trận X còn chứa NaN hoặc Infinity."
    )



np.savez_compressed(
    CACHE_PATH,
    X=X,
    y=y
)

print("Đã lưu cache:", CACHE_PATH)

print(
    "Dung lượng:",
    round(CACHE_PATH.stat().st_size / (1024 ** 2), 2),
    "MB"
)



with np.load(CACHE_PATH) as data:

    X_check = data["X"]
    y_check = data["y"]

print("\nKIỂM TRA CUỐI CÙNG")

print("X:", X_check.shape)
print("y:", y_check.shape)

print(
    "Phân bố nhãn:",
    {
        int(label): int(count)
        for label, count in zip(
            *np.unique(y_check, return_counts=True)
        )
    }
)

print(
    "Tổng thời gian:",
    round(time.time() - start_time, 2),
    "giây"
)

