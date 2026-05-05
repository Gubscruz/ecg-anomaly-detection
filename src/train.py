from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import numpy as np
import wfdb
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.layers import (
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Input,
    LSTM,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ECG arrhythmia model.")
    parser.add_argument(
        "--params",
        default="params.yaml",
        help="Path to the DVC params YAML file.",
    )
    return parser.parse_args()


def load_params(params_path: Path) -> dict[str, Any]:
    with params_path.open("r", encoding="utf-8") as stream:
        params = yaml.safe_load(stream) or {}

    if "train" not in params:
        raise ValueError("params.yaml must define a top-level `train` section.")
    if "outputs" not in params:
        raise ValueError("params.yaml must define a top-level `outputs` section.")

    return params


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def zscore(signal: np.ndarray) -> np.ndarray:
    std = float(np.std(signal))
    if np.isclose(std, 0.0):
        raise ValueError("Cannot normalize a signal with zero standard deviation.")
    return ((signal - float(np.mean(signal))) / std).astype(np.float32)


def prepare_windows(
    data_dir: Path,
    records: list[str],
    target_symbols: set[str],
    window_size: int,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if window_size % 2 != 0:
        raise ValueError("train.window_size must be an even number.")

    half_window = window_size // 2
    x_list: list[np.ndarray] = []
    y_list: list[str] = []

    for record_name in records:
        record_base = data_dir / record_name
        if not (record_base.with_suffix(".hea").exists() and record_base.with_suffix(".atr").exists()):
            raise FileNotFoundError(
                f"Missing WFDB files for record {record_name!r} in {data_dir}."
            )

        record = wfdb.rdrecord(str(record_base))
        annotation = wfdb.rdann(str(record_base), "atr")
        signal = record.p_signal[:, 0].astype(np.float32)
        if normalize:
            signal = zscore(signal)

        for index, symbol in enumerate(annotation.symbol):
            if symbol not in target_symbols:
                continue

            peak = int(annotation.sample[index])
            start = peak - half_window
            end = peak + half_window
            if start < 0 or end > len(signal):
                continue

            x_list.append(signal[start:end])
            y_list.append(symbol)

    if not x_list:
        raise ValueError("No ECG windows were created. Check train.records and train.target_symbols.")

    x = np.asarray(x_list, dtype=np.float32).reshape((-1, window_size, 1))
    y = np.asarray(y_list)
    return x, y


def build_model(train_params: dict[str, Any], class_count: int) -> tf.keras.Model:
    model_params = train_params.get("model", {})
    architecture = model_params.get("architecture", "cnn_bilstm")
    window_size = int(train_params["window_size"])

    if architecture == "cnn_bilstm":
        model = Sequential(
            [
                Input(shape=(window_size, 1)),
                Conv1D(
                    filters=int(model_params.get("conv_filters", 32)),
                    kernel_size=int(model_params.get("kernel_size", 5)),
                    activation=model_params.get("conv_activation", "relu"),
                ),
                MaxPooling1D(pool_size=int(model_params.get("pool_size", 2))),
                Bidirectional(
                    LSTM(
                        int(model_params.get("lstm_units", 64)),
                        return_sequences=False,
                    )
                ),
                Dropout(float(model_params.get("dropout", 0.4))),
                Dense(
                    int(model_params.get("dense_units", 32)),
                    activation=model_params.get("dense_activation", "relu"),
                ),
                Dense(class_count, activation="softmax"),
            ]
        )
    elif architecture == "lstm":
        model = Sequential(
            [
                Input(shape=(window_size, 1)),
                LSTM(int(model_params.get("lstm_units", 64)), return_sequences=False),
                Dropout(float(model_params.get("dropout", 0.2))),
                Dense(
                    int(model_params.get("dense_units", 32)),
                    activation=model_params.get("dense_activation", "relu"),
                ),
                Dense(class_count, activation="softmax"),
            ]
        )
    else:
        raise ValueError(f"Unsupported train.model.architecture: {architecture!r}")

    model.compile(
        optimizer=train_params.get("optimizer", "adam"),
        loss=train_params.get("loss", "categorical_crossentropy"),
        metrics=["accuracy"],
    )
    return model


def maybe_apply_smote(
    x_train: np.ndarray,
    y_train_indices: np.ndarray,
    train_params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if not bool(train_params.get("apply_smote", False)):
        return x_train, y_train_indices

    x_train_flat = x_train.reshape((x_train.shape[0], -1))
    smote = SMOTE(
        sampling_strategy=train_params.get("smote_sampling_strategy", "auto"),
        random_state=int(train_params.get("random_state", 42)),
        k_neighbors=int(train_params.get("smote_k_neighbors", 5)),
    )
    x_resampled, y_resampled = smote.fit_resample(x_train_flat, y_train_indices)
    return x_resampled.reshape((-1, int(train_params["window_size"]), 1)), y_resampled


def compute_class_weights(enabled: bool, y_train_indices: np.ndarray) -> dict[int, float] | None:
    if not enabled:
        return None

    classes = np.unique(y_train_indices)
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_indices,
    )
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def write_history_csv(history: tf.keras.callbacks.History, history_path: Path) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = sorted(history.history.keys())

    with history_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", *metric_names])
        writer.writeheader()
        for epoch_index in range(len(next(iter(history.history.values()), []))):
            row = {"epoch": epoch_index + 1}
            for name in metric_names:
                row[name] = float(history.history[name][epoch_index])
            writer.writerow(row)


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def write_metrics(
    metrics_path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    class_distribution: Counter[str],
    train_size: int,
    test_size: int,
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "samples": int(train_size + test_size),
        "train_samples": int(train_size),
        "test_samples": int(test_size),
        "class_distribution": {
            str(label): int(count)
            for label, count in sorted(
                class_distribution.items(),
                key=lambda item: str(item[0]),
            )
        },
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in labels
        },
        "confusion_matrix": matrix,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(to_builtin(metrics), stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    args = parse_args()
    params_path = resolve_path(args.params)
    params = load_params(params_path)
    train_params = params["train"]
    output_params = params["outputs"]

    random_state = int(train_params.get("random_state", 42))
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    data_dir = resolve_path(train_params["data_dir"])
    records = [str(record) for record in train_params["records"]]
    target_symbols = {str(symbol) for symbol in train_params["target_symbols"]}

    x, y = prepare_windows(
        data_dir=data_dir,
        records=records,
        target_symbols=target_symbols,
        window_size=int(train_params["window_size"]),
        normalize=bool(train_params.get("normalize", True)),
    )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded, num_classes=len(label_encoder.classes_))

    x_train, x_test, y_train, y_test, y_train_indices, y_test_indices = train_test_split(
        x,
        y_categorical,
        y_encoded,
        test_size=float(train_params.get("test_size", 0.2)),
        random_state=random_state,
        stratify=y_encoded,
    )

    x_train, y_train_indices = maybe_apply_smote(x_train, y_train_indices, train_params)
    y_train = to_categorical(y_train_indices, num_classes=len(label_encoder.classes_))

    class_weights = compute_class_weights(
        bool(train_params.get("class_weight", False)),
        y_train_indices,
    )

    model = build_model(train_params, class_count=len(label_encoder.classes_))
    history = model.fit(
        x_train,
        y_train,
        epochs=int(train_params.get("epochs", 15)),
        batch_size=int(train_params.get("batch_size", 64)),
        validation_data=(x_test, y_test),
        class_weight=class_weights,
        verbose=int(train_params.get("verbose", 1)),
    )

    model_path = resolve_path(output_params["model_path"])
    encoder_path = resolve_path(output_params["encoder_path"])
    history_path = resolve_path(output_params["history_path"])
    metrics_path = resolve_path(output_params["metrics_path"])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    with encoder_path.open("wb") as stream:
        pickle.dump(label_encoder, stream)

    y_pred_probabilities = model.predict(x_test, verbose=0)
    y_pred_indices = np.argmax(y_pred_probabilities, axis=1)

    write_history_csv(history, history_path)
    write_metrics(
        metrics_path=metrics_path,
        y_true=y_test_indices,
        y_pred=y_pred_indices,
        labels=[str(label) for label in label_encoder.classes_],
        class_distribution=Counter(y),
        train_size=len(x_train),
        test_size=len(x_test),
    )

    print(f"Saved model to {model_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved label encoder to {encoder_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved metrics to {metrics_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
