from __future__ import annotations

import argparse
import errno
import importlib.util
import json
import math
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "modelo_arritmia_final_v3.h5"
WINDOW_SIZE = 360
HALF_WINDOW = WINDOW_SIZE // 2

# LabelEncoder in the notebook was fit on ["N", "V", "L", "R", "A"] and sorts
# alphabetically, so the model outputs follow this order.
LABELS = ("A", "L", "N", "R", "V")
LABEL_DESCRIPTIONS = {
    "A": "Batimento atrial prematuro",
    "L": "Bloqueio de ramo esquerdo",
    "N": "Batimento normal",
    "R": "Bloqueio de ramo direito",
    "V": "Contracao ventricular prematura",
}

NUMBER_SPLIT_RE = re.compile(r"[\s,;]+")


class PredictionError(Exception):
    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


class ArrhythmiaPredictor:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._model = None

    @property
    def tensorflow_available(self) -> bool:
        return importlib.util.find_spec("tensorflow") is not None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self):
        if not self.tensorflow_available:
            raise PredictionError(
                "TensorFlow nao esta instalado. Instale as dependencias com "
                "`pip install -r requirements.txt` e rode a aplicacao novamente.",
                HTTPStatus.SERVICE_UNAVAILABLE,
            )

        if not self.model_path.exists():
            raise PredictionError(
                f"Modelo nao encontrado em {self.model_path}",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        if self._model is None:
            import tensorflow as tf

            self._model = tf.keras.models.load_model(self.model_path, compile=False)

        return self._model

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        windows, metadata = build_windows(payload)
        model = self._load_model()

        batch = windows.reshape((windows.shape[0], WINDOW_SIZE, 1))
        raw_predictions = model.predict(batch, verbose=0)
        probabilities = np.asarray(raw_predictions, dtype=np.float64)

        if probabilities.ndim != 2 or probabilities.shape[1] != len(LABELS):
            raise PredictionError(
                "A saida do modelo nao combina com as 5 classes esperadas.",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        predictions = []
        for index, row in enumerate(probabilities):
            class_index = int(np.argmax(row))
            label = LABELS[class_index]
            predictions.append(
                {
                    "index": index,
                    "label": label,
                    "description": LABEL_DESCRIPTIONS[label],
                    "confidence": round(float(row[class_index]), 6),
                    "probabilities": {
                        label_name: round(float(probability), 6)
                        for label_name, probability in zip(LABELS, row)
                    },
                    **metadata[index],
                }
            )

        return {
            "model": str(self.model_path),
            "window_size": WINDOW_SIZE,
            "labels": list(LABELS),
            "predictions": predictions,
        }


def parse_numeric_sequence(value: Any, field_name: str) -> np.ndarray:
    if isinstance(value, str):
        parts = [part for part in NUMBER_SPLIT_RE.split(value.strip()) if part]
        value = parts

    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise PredictionError(f"`{field_name}` deve conter apenas numeros.") from exc

    if array.ndim != 1:
        raise PredictionError(f"`{field_name}` deve ser uma lista plana de numeros.")

    if array.size == 0:
        raise PredictionError(f"`{field_name}` nao pode estar vazio.")

    if not np.all(np.isfinite(array)):
        raise PredictionError(f"`{field_name}` possui NaN ou infinito.")

    return array


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    mean = float(np.mean(signal))
    std = float(np.std(signal))

    if math.isclose(std, 0.0, abs_tol=1e-8):
        raise PredictionError("Nao foi possivel normalizar: desvio padrao igual a zero.")

    return ((signal - mean) / std).astype(np.float32)


def as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "sim", "yes", "y"}
    return bool(value)


def build_windows(payload: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    normalize = as_bool(payload.get("normalize"), default=True)

    if "windows" in payload:
        try:
            windows = np.asarray(payload["windows"], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise PredictionError("`windows` deve ser uma matriz numerica.") from exc

        if windows.ndim != 2 or windows.shape[1] != WINDOW_SIZE:
            raise PredictionError("`windows` deve ter formato [n, 360].")

        if not np.all(np.isfinite(windows)):
            raise PredictionError("`windows` possui NaN ou infinito.")

        if normalize:
            windows = np.stack([normalize_signal(window) for window in windows])

        metadata = [{"source": "windows"} for _ in range(windows.shape[0])]
        return windows.astype(np.float32), metadata

    if "window" in payload:
        payload = {**payload, "signal": payload["window"]}

    if "signal" not in payload:
        raise PredictionError("Envie `signal`, `window` ou `windows` no JSON.")

    signal = parse_numeric_sequence(payload["signal"], "signal")

    if signal.size == WINDOW_SIZE:
        window = normalize_signal(signal) if normalize else signal.astype(np.float32)
        return window.reshape(1, WINDOW_SIZE), [{"source": "signal"}]

    if "rpeaks" not in payload:
        raise PredictionError(
            "`signal` deve ter exatamente 360 pontos ou vir acompanhado de `rpeaks`."
        )

    rpeaks = parse_rpeaks(payload["rpeaks"])
    normalized_signal = normalize_signal(signal) if normalize else signal.astype(np.float32)
    windows = []
    metadata = []

    for peak in rpeaks:
        start = peak - HALF_WINDOW
        end = peak + HALF_WINDOW
        if start < 0 or end > normalized_signal.size:
            raise PredictionError(
                f"R-peak {peak} nao permite uma janela completa de 360 pontos."
            )
        windows.append(normalized_signal[start:end])
        metadata.append({"source": "rpeaks", "rpeak": peak, "start": start, "end": end})

    return np.stack(windows).astype(np.float32), metadata


def parse_rpeaks(value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        raise PredictionError("`rpeaks` deve ser uma lista nao vazia de inteiros.")

    peaks = []
    for item in value:
        if not isinstance(item, (int, np.integer)):
            raise PredictionError("`rpeaks` deve conter apenas inteiros.")
        peaks.append(int(item))
    return peaks


INDEX_HTML = """<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Detector de Arritmia</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f3;
      --ink: #1d2433;
      --muted: #617084;
      --panel: #ffffff;
      --accent: #146c94;
      --line: #d7dde5;
      --ok: #0f766e;
      --err: #b42318;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }

    main {
      width: min(1120px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 32px 0;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 24px;
    }

    h1 {
      margin: 0;
      font-size: clamp(1.7rem, 3vw, 2.5rem);
      line-height: 1.1;
      letter-spacing: 0;
    }

    .status {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 12px;
      background: var(--panel);
      color: var(--muted);
      font-size: 0.9rem;
      white-space: nowrap;
    }

    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 0.85fr);
      gap: 18px;
      align-items: stretch;
    }

    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 560px;
      display: flex;
      flex-direction: column;
    }

    .section-head {
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    h2 {
      margin: 0;
      font-size: 1rem;
      letter-spacing: 0;
    }

    textarea {
      width: 100%;
      flex: 1;
      min-height: 390px;
      resize: vertical;
      border: 0;
      outline: 0;
      padding: 18px;
      font: 0.95rem/1.55 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
        "Liberation Mono", monospace;
      color: var(--ink);
      background: #fbfcfd;
    }

    button {
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      padding: 10px 14px;
      cursor: pointer;
    }

    button:disabled {
      cursor: wait;
      opacity: 0.6;
    }

    .actions {
      padding: 14px 18px;
      border-top: 1px solid var(--line);
      display: flex;
      gap: 10px;
      justify-content: flex-end;
    }

    pre {
      margin: 0;
      flex: 1;
      overflow: auto;
      padding: 18px;
      white-space: pre-wrap;
      word-break: break-word;
      font: 0.95rem/1.55 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
        "Liberation Mono", monospace;
      color: var(--ink);
      background: #fbfcfd;
    }

    .hint {
      color: var(--muted);
      font-size: 0.9rem;
    }

    .ok { color: var(--ok); }
    .err { color: var(--err); }

    @media (max-width: 860px) {
      main { width: min(100vw - 20px, 720px); padding: 18px 0; }
      header { align-items: flex-start; flex-direction: column; }
      .status { white-space: normal; }
      .workspace { grid-template-columns: 1fr; }
      section { min-height: 420px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Detector de Arritmia</h1>
      <div id="status" class="status">Carregando status...</div>
    </header>

    <div class="workspace">
      <section>
        <div class="section-head">
          <h2>Entrada</h2>
          <span class="hint">JSON</span>
        </div>
        <textarea id="payload" spellcheck="false" placeholder='{"signal": [360 valores de ECG]}'></textarea>
        <div class="actions">
          <button id="sampleButton" type="button">Exemplo</button>
          <button id="predictButton" type="button">Predizer</button>
        </div>
      </section>

      <section>
        <div class="section-head">
          <h2>Resultado</h2>
          <span class="hint">A, L, N, R, V</span>
        </div>
        <pre id="result">Aguardando dados.</pre>
      </section>
    </div>
  </main>

  <script>
    const payloadInput = document.querySelector("#payload");
    const result = document.querySelector("#result");
    const statusEl = document.querySelector("#status");
    const predictButton = document.querySelector("#predictButton");
    const sampleButton = document.querySelector("#sampleButton");

    function makeExampleSignal() {
      const values = [];
      for (let i = 0; i < 360; i += 1) {
        const baseline = 0.08 * Math.sin((2 * Math.PI * i) / 120);
        const qrs = Math.exp(-Math.pow((i - 180) / 10, 2));
        values.push(Number((baseline + qrs).toFixed(5)));
      }
      return { signal: values };
    }

    async function refreshStatus() {
      const response = await fetch("/health");
      const data = await response.json();
      const state = data.tensorflow_available ? "pronto" : "sem TensorFlow";
      statusEl.textContent = `Modelo: ${data.model_loaded ? "carregado" : state}`;
      statusEl.className = `status ${data.tensorflow_available ? "ok" : "err"}`;
    }

    sampleButton.addEventListener("click", () => {
      payloadInput.value = JSON.stringify(makeExampleSignal(), null, 2);
    });

    predictButton.addEventListener("click", async () => {
      predictButton.disabled = true;
      result.textContent = "Processando...";

      try {
        const parsed = JSON.parse(payloadInput.value);
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(parsed),
        });
        const data = await response.json();
        result.textContent = JSON.stringify(data, null, 2);
        await refreshStatus();
      } catch (error) {
        result.textContent = JSON.stringify({ error: String(error) }, null, 2);
      } finally {
        predictButton.disabled = false;
      }
    });

    refreshStatus().catch(() => {
      statusEl.textContent = "Status indisponivel";
      statusEl.className = "status err";
    });
  </script>
</body>
</html>
"""


def make_handler(predictor: ArrhythmiaPredictor):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ECGArrhythmiaApp/1.0"

        def do_GET(self):
            if self.path in {"/", "/index.html"}:
                self._send_bytes(
                    INDEX_HTML.encode("utf-8"),
                    HTTPStatus.OK,
                    "text/html; charset=utf-8",
                )
                return

            if self.path == "/health":
                self._send_json(
                    {
                        "ok": True,
                        "tensorflow_available": predictor.tensorflow_available,
                        "model_loaded": predictor.model_loaded,
                        "model_path": str(predictor.model_path),
                        "labels": list(LABELS),
                    }
                )
                return

            self._send_json({"error": "Rota nao encontrada."}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if self.path != "/predict":
                self._send_json({"error": "Rota nao encontrada."}, HTTPStatus.NOT_FOUND)
                return

            try:
                payload = self._read_json()
                response = predictor.predict(payload)
                self._send_json(response)
            except PredictionError as exc:
                self._send_json({"error": str(exc)}, exc.status)
            except json.JSONDecodeError:
                self._send_json({"error": "JSON invalido."}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._send_json(
                    {"error": f"Erro inesperado: {exc}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def do_OPTIONS(self):
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def _read_json(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise PredictionError("O corpo da requisicao deve ser um objeto JSON.")
            return payload

        def _send_json(
            self,
            payload: dict[str, Any],
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self._send_bytes(body, status, "application/json; charset=utf-8")

        def _send_bytes(
            self,
            body: bytes,
            status: HTTPStatus,
            content_type: str,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ECG arrhythmia prediction app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = ArrhythmiaPredictor(args.model)
    handler = make_handler(predictor)

    try:
        server = ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(
                f"Porta {args.port} ja esta em uso. "
                f"Tente: python app.py --port {args.port + 1}"
            )
            return
        raise

    print(f"Servidor iniciado em http://{args.host}:{args.port}")
    print(f"Modelo: {args.model}")
    server.serve_forever()


if __name__ == "__main__":
    main()
