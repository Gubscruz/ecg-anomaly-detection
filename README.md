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

## Cliente Streamlit

O arquivo `streamlit_app.py` fornece uma interface simples para enviar
requisicoes para a API existente.

Em um terminal, rode a API:

```bash
source ecg-env/bin/activate
python app.py --port 8000
```

Em outro terminal, rode o cliente:

```bash
source ecg-env/bin/activate
streamlit run streamlit_app.py
```

No app Streamlit, use a URL base `http://127.0.0.1:8000`. Se a API estiver em
outra porta, altere esse campo na barra lateral.

## Experimentos com DVC

O projeto esta configurado para versionar o dataset MIT-BIH usado no treino,
os parametros do experimento, o modelo treinado e as metricas geradas.

```bash
source ecg-env/bin/activate
pip install -r requirements.txt
```

Arquivos principais:

- `params.yaml`: parametros editaveis do treino.
- `dvc.yaml`: pipeline DVC que executa `src/train.py`.
- `data/mit-bih-arrhythmia-database-1.0.0.dvc`: hash/versionamento do dataset.
- `models/modelo_arritmia_final_v3.h5`: modelo gerado pelo pipeline.
- `models/label_encoder.pkl`: encoder das classes gerado pelo pipeline.
- `metrics/train.json`: metricas finais.
- `metrics/history.csv`: historico de treino para plots do DVC.

Para executar e registrar um run:

```bash
dvc exp run
```

Para comparar runs e parametros:

```bash
dvc exp show
dvc exp diff
dvc metrics show
```

Para rodar o pipeline sem criar um experimento separado:

```bash
dvc repro
```

## Dashboard com DagsHub + MLflow

O treino tambem envia parametros, metricas, historico de epocas e artefatos
para MLflow quando `tracking.mlflow.enabled` esta ativo em `params.yaml`.
Para usar o dashboard compartilhado do DagsHub:

1. Crie ou importe este repositorio em `https://dagshub.com`.
2. No DagsHub, gere um token em **User Settings > Tokens**.
3. Configure as credenciais no terminal, sem commitar secrets:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<owner>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<dagshub-username>
export MLFLOW_TRACKING_PASSWORD=<dagshub-token>
```

4. Rode o treino normalmente pelo DVC:

```bash
dvc exp run
```

Depois abra o dashboard em:

```text
https://dagshub.com/<owner>/<repo>/experiments
```

Cada run registra:

- parametros de `params.yaml`;
- metricas finais de `metrics/train.json`;
- curvas de treino/validacao de `metrics/history.csv`;
- modelo `.h5`, label encoder, `dvc.yaml`, `dvc.lock` e arquivo `.dvc` do dataset.

Tambem e possivel configurar o repositorio direto em `params.yaml`:

```yaml
tracking:
  mlflow:
    dagshub:
      repo_owner: "<owner>"
      repo_name: "<repo>"
```

Se quiser rodar sem logging no MLflow:

```bash
dvc exp run -S tracking.mlflow.enabled=false
```

Se quiser compartilhar os dados/modelos via remote, configure um destino e envie
o cache:

```bash
dvc remote add -d storage <remote-url>
dvc push
```

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
