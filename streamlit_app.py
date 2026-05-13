from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
import wfdb


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_DATA_DIR = "data/mit-bih-arrhythmia-database-1.0.0"
TARGET_LABELS = ["N", "A", "L", "R", "V"]
WINDOW_SIZE = 360
HALF_WINDOW = WINDOW_SIZE // 2
NUMBER_SPLIT_RE = re.compile(r"[\s,;]+")


def normalize_api_url(api_base_url: str) -> str:
    return api_base_url.strip().rstrip("/")


def parse_signal_text(value: str) -> list[float]:
    parts = [part for part in NUMBER_SPLIT_RE.split(value.strip()) if part]
    signal = [float(part) for part in parts]
    if len(signal) != WINDOW_SIZE:
        raise ValueError(f"Expected {WINDOW_SIZE} values, got {len(signal)}.")
    return signal


def load_mit_bih_window(
    data_dir: str,
    record_id: str,
    target_label: str,
) -> tuple[list[float], dict[str, Any]]:
    record_base = Path(data_dir) / record_id.strip()
    record = wfdb.rdrecord(str(record_base))
    annotation = wfdb.rdann(str(record_base), "atr")

    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol != target_label:
            continue
        if HALF_WINDOW < sample < len(record.p_signal) - HALF_WINDOW:
            window = record.p_signal[sample - HALF_WINDOW : sample + HALF_WINDOW, 0]
            return window.astype(float).tolist(), {
                "record": record_id,
                "true_label": symbol,
                "rpeak": int(sample),
            }

    raise ValueError(
        f"No complete {WINDOW_SIZE}-point window with label {target_label!r} "
        f"was found in record {record_id!r}."
    )


def get_health(api_base_url: str, timeout: float) -> dict[str, Any]:
    response = requests.get(f"{api_base_url}/health", timeout=timeout)
    response.raise_for_status()
    return response.json()


def predict(api_base_url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    response = requests.post(f"{api_base_url}/predict", json=payload, timeout=timeout)
    try:
        response_payload = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"API returned non-JSON response: {response.text}") from exc

    if not response.ok:
        message = response_payload.get("error") or response.text
        raise RuntimeError(f"API error {response.status_code}: {message}")

    return response_payload


def render_predictions(response_payload: dict[str, Any]) -> None:
    predictions = response_payload.get("predictions", [])
    if not predictions:
        st.warning("The API response did not include predictions.")
        st.json(response_payload)
        return

    summary_rows = []
    probability_rows = []
    for prediction in predictions:
        summary_rows.append(
            {
                "index": prediction.get("index"),
                "label": prediction.get("label"),
                "description": prediction.get("description"),
                "confidence": prediction.get("confidence"),
                "source": prediction.get("source"),
                "rpeak": prediction.get("rpeak"),
            }
        )
        for label, probability in (prediction.get("probabilities") or {}).items():
            probability_rows.append(
                {
                    "window": prediction.get("index"),
                    "label": label,
                    "probability": probability,
                }
            )

    st.subheader("Prediction")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    if probability_rows:
        st.subheader("Class probabilities")
        probabilities = pd.DataFrame(probability_rows)
        for window_index, window_probabilities in probabilities.groupby("window"):
            st.caption(f"Window {window_index}")
            chart_data = window_probabilities.set_index("label")["probability"]
            st.bar_chart(chart_data)

    with st.expander("Raw API response"):
        st.json(response_payload)


def build_payload(source_mode: str, normalize: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata: dict[str, Any] = {}

    if source_mode == "MIT-BIH sample":
        data_dir = st.text_input("Data directory", value=DEFAULT_DATA_DIR)
        record_id = st.text_input("Record ID", value="100")
        target_label = st.selectbox("Target label to sample", TARGET_LABELS, index=0)
        signal, metadata = load_mit_bih_window(data_dir, record_id, target_label)
        return {"signal": signal, "normalize": normalize}, metadata

    if source_mode == "Manual signal":
        signal_text = st.text_area(
            "360 signal values",
            height=220,
            placeholder="Paste 360 numeric values separated by commas, spaces, or new lines.",
        )
        return {"signal": parse_signal_text(signal_text), "normalize": normalize}, metadata

    raw_json = st.text_area(
        "Request JSON",
        height=260,
        value=json.dumps({"signal": [0.0] * WINDOW_SIZE, "normalize": normalize}, indent=2),
    )
    payload = json.loads(raw_json)
    if not isinstance(payload, dict):
        raise ValueError("Raw JSON request must be a JSON object.")
    return payload, metadata


def main() -> None:
    st.set_page_config(page_title="ECG API Client", layout="wide")
    st.title("ECG API Client")

    with st.sidebar:
        st.header("API")
        api_base_url = normalize_api_url(
            st.text_input("Base URL", value=DEFAULT_API_BASE_URL)
        )
        timeout = st.number_input("Timeout (seconds)", min_value=1, max_value=120, value=60)

        if st.button("Check health", use_container_width=True):
            try:
                st.success("API is reachable.")
                st.json(get_health(api_base_url, timeout))
            except Exception as exc:
                st.error(str(exc))

    source_mode = st.radio(
        "Request source",
        ["MIT-BIH sample", "Manual signal", "Raw JSON"],
        horizontal=True,
    )
    normalize = st.checkbox("Ask API to normalize signal", value=True)

    try:
        payload, metadata = build_payload(source_mode, normalize)
    except Exception as exc:
        st.error(str(exc))
        return

    if metadata:
        st.info(
            "Loaded sample "
            f"record={metadata['record']} "
            f"label={metadata['true_label']} "
            f"rpeak={metadata['rpeak']}"
        )

    with st.expander("Request payload", expanded=False):
        st.json(payload)

    if st.button("Send prediction request", type="primary"):
        with st.spinner("Calling prediction API..."):
            try:
                response_payload = predict(api_base_url, payload, timeout)
            except Exception as exc:
                st.error(str(exc))
                return

        render_predictions(response_payload)


if __name__ == "__main__":
    main()
