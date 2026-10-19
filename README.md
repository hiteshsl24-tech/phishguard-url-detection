# PhishGuard — Phishing URL Detection

A Streamlit web app that engineers simple URL features and trains a Random Forest classifier to distinguish labeled legitimate and potentially phishing URLs.

## Run in VS Code (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Features

- Upload a labeled URL CSV to train a baseline model.
- Inspect the held-out accuracy, classification report, and confusion matrix.
- Analyze one URL.
- Upload a CSV of URLs and download predictions.

## Dataset format

The bundled `data/sample_urls.csv` has columns `url` and `label`. Label values may be `phishing` / `legitimate` or `1` / `0`. Upload your own properly sourced dataset from the app sidebar.

## Limitations

The sample rows are synthetic and use reserved/test-style domains. Lexical URL patterns alone cannot establish whether a website is malicious. Do not use the predictions as the only security control. Metrics on this synthetic data are not evidence of performance on real phishing URLs.
