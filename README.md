# Project Template (Cookiecutter)

Minimal Cookiecutter template for creating clean and organized machine learning projects

## Aplicacao de predicao

O app simples esta em `app.py` e usa o modelo salvo em
`models/modelo_arritmia_final_v3.h5`.

```bash
source ecg-env/bin/activate
pip install -r requirements.txt
python app.py
```

Depois acesse `http://127.0.0.1:8000` ou envie JSON direto para a API:

Se a porta 8000 ja estiver em uso, rode com outra porta:

```bash
python app.py --port 8001
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data-binary @payload.json
```

O `payload.json` deve conter uma das entradas abaixo.

Formatos aceitos:

- `{"signal": [...]}` com exatamente 360 pontos.
- `{"windows": [[...], [...]]}` para varias janelas de 360 pontos.
- `{"signal": [...], "rpeaks": [1000, 1500]}` para recortar janelas de 360 pontos ao redor dos R-peaks.

Por padrao o app aplica normalizacao Z-score antes da predicao, igual ao notebook.
Use `"normalize": false` se os dados ja estiverem normalizados.

## Structure

The template generates a project with the following layout:

```
ecg/
├── .gitignore
├── README.md
├── data/
│   └── .gitkeep        # placeholder for raw and processed data
├── models/
│   └── .gitkeep        # placeholder for saved models
├── notebooks/
│   └── .gitkeep        # placeholder for Jupyter notebooks
└── src/
    ├── __init__.py
    ├── process.py      # data processing script
    └── train.py        # model training script
```

## cookiecutter.json

The configuration file allows you to set:

* `directory_name` — name of the generated project folder
  (and any additional fields you may add later)

## Usage

Generate a new project:

```bash
cookiecutter https://github.com/<your_username>/<your_template_repo>.git
```
