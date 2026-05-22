from __future__ import annotations

import json
import re
import time
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


@st.cache_data(show_spinner=False)
def list_record_ids(data_dir: str) -> list[str]:
    base_dir = Path(data_dir)
    if not base_dir.exists():
        return []

    return sorted({path.stem for path in base_dir.glob("*.hea")})


@st.cache_data(show_spinner=False)
def load_mit_bih_record(
    data_dir: str,
    record_id: str,
) -> tuple[list[float], list[dict[str, Any]], dict[str, Any]]:
    record_base = Path(data_dir) / record_id.strip()
    record = wfdb.rdrecord(str(record_base))
    annotation = wfdb.rdann(str(record_base), "atr")

    signal = record.p_signal[:, 0].astype(float).tolist()
    annotations = [
        {"sample": int(sample), "label": symbol}
        for sample, symbol in zip(annotation.sample, annotation.symbol)
    ]
    metadata = {
        "record": record_id,
        "length": len(signal),
        "sampling_frequency": float(record.fs),
        "lead": record.sig_name[0] if getattr(record, "sig_name", None) else "MLII",
    }
    return signal, annotations, metadata


def find_window_annotation(
    annotations: list[dict[str, Any]],
    window_start: int,
    window_end: int,
) -> dict[str, Any] | None:
    candidates = [
        annotation
        for annotation in annotations
        if window_start <= annotation["sample"] < window_end
    ]
    if not candidates:
        return None
    return candidates[-1]


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


def render_live_predictions(history: list[dict[str, Any]]) -> None:
    if not history:
        st.info("No windows were classified yet.")
        return

    summary_frame = pd.DataFrame(history)
    latest = summary_frame.iloc[-1].to_dict()

    st.subheader("Live status")
    status_columns = st.columns(4)
    status_columns[0].metric("Windows classified", len(summary_frame))
    status_columns[1].metric("Latest label", latest.get("label", "-"))
    status_columns[2].metric("Confidence", f"{float(latest.get('confidence', 0.0)):.3f}")
    status_columns[3].metric("Samples ingested", int(latest.get("window_end", 0)))

    st.dataframe(
        summary_frame[
            [
                "step",
                "window_start",
                "window_end",
                "label",
                "confidence",
                "truth_label",
                "rpeak",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    confidence_frame = summary_frame[["step", "confidence"]].set_index("step")
    st.line_chart(confidence_frame, use_container_width=True)

    with st.expander("Raw live history"):
        st.json(history)


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


def run_live_simulation(
    api_base_url: str,
    data_dir: str,
    record_id: str,
    normalize: bool,
    chunk_size: int,
    frame_delay_seconds: float,
    max_steps: int,
) -> None:
    signal, annotations, record_metadata = load_mit_bih_record(data_dir, record_id)

    if len(signal) < WINDOW_SIZE:
        st.error(
            f"Record {record_id!r} has only {len(signal)} samples, which is not enough for a {WINDOW_SIZE}-point window."
        )
        return

    chunk_size = max(1, int(chunk_size))
    max_steps = max(1, int(max_steps))
    frame_delay_seconds = max(0.0, float(frame_delay_seconds))

    window_end = WINDOW_SIZE
    step = 0
    history: list[dict[str, Any]] = []

    chart_placeholder = st.empty()
    status_placeholder = st.empty()
    table_placeholder = st.empty()
    progress_bar = st.progress(0)

    st.caption(
        f"Streaming record {record_metadata['record']} ({record_metadata['lead']}, {record_metadata['sampling_frequency']:.0f} Hz). "
        f"New samples arrive in chunks of {chunk_size} and each complete 360-point window is sent to the API."
    )

    while window_end <= len(signal) and step < max_steps:
        window_start = window_end - WINDOW_SIZE
        window = signal[window_start:window_end]
        payload = {"signal": window, "normalize": normalize}
        window_annotation = find_window_annotation(annotations, window_start, window_end)

        try:
            response_payload = predict(api_base_url, payload, timeout=60)
        except Exception as exc:
            status_placeholder.error(str(exc))
            break

        prediction = (response_payload.get("predictions") or [{}])[0]
        history.append(
            {
                "step": step + 1,
                "window_start": window_start,
                "window_end": window_end,
                "label": prediction.get("label"),
                "confidence": prediction.get("confidence"),
                "truth_label": None if window_annotation is None else window_annotation["label"],
                "rpeak": None if window_annotation is None else window_annotation["sample"],
            }
        )

        live_trace = pd.DataFrame(
            {
                "sample": list(range(window_start, window_end)),
                "amplitude": window,
            }
        ).set_index("sample")

        chart_placeholder.line_chart(live_trace, use_container_width=True)
        status_placeholder.success(
            " | ".join(
                [
                    f"step={step + 1}",
                    f"window={window_start}:{window_end}",
                    f"pred={prediction.get('label')} ({float(prediction.get('confidence', 0.0)):.3f})",
                    f"truth={history[-1]['truth_label'] or 'n/a'}",
                ]
            )
        )
        table_placeholder.dataframe(
            pd.DataFrame(history)[
                [
                    "step",
                    "window_start",
                    "window_end",
                    "label",
                    "confidence",
                    "truth_label",
                    "rpeak",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        step += 1
        progress_bar.progress(min(step / max_steps, 1.0))
        if frame_delay_seconds > 0 and step < max_steps and window_end + chunk_size <= len(signal):
            time.sleep(frame_delay_seconds)
        window_end += chunk_size

    if history:
        st.divider()
        render_live_predictions(history)


def main() -> None:
    st.set_page_config(page_title="ECG API Client", layout="wide")
    st.title("ECG API Client")
    st.caption("Send a single ECG window or simulate a live sensor stream from MIT-BIH data.")

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

    normalize = st.checkbox("Ask API to normalize signal", value=True)

    app_mode = st.radio(
        "Mode",
        ["Single request", "Live sensor simulation"],
        horizontal=True,
    )

    if app_mode == "Live sensor simulation":
        live_data_dir = st.text_input("Data directory", value=DEFAULT_DATA_DIR)
        available_records = list_record_ids(live_data_dir)
        if not available_records:
            st.error(
                "No MIT-BIH records were found in the selected data directory. Check the path before starting the simulation."
            )
            return

        live_record_id = st.selectbox("Record ID", available_records, index=0)
        chunk_size = st.slider("Incoming samples per tick", min_value=1, max_value=36, value=6)
        frame_delay_seconds = st.slider(
            "Delay between ticks (seconds)", min_value=0.0, max_value=2.0, value=0.25, step=0.05
        )
        max_steps = st.number_input("Classification steps", min_value=1, max_value=500, value=30)

        if st.button("Start live simulation", type="primary", use_container_width=True):
            with st.spinner("Streaming ECG samples and classifying each new window..."):
                run_live_simulation(
                    api_base_url=api_base_url,
                    data_dir=live_data_dir,
                    record_id=live_record_id,
                    normalize=normalize,
                    chunk_size=chunk_size,
                    frame_delay_seconds=frame_delay_seconds,
                    max_steps=max_steps,
                )
        return

    source_mode = st.radio(
        "Request source",
        ["MIT-BIH sample", "Manual signal", "Raw JSON"],
        horizontal=True,
    )

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
