# Project Template (Cookiecutter)

Minimal Cookiecutter template for creating clean and organized machine learning projects

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