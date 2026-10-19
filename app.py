from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "sample_urls.csv"
FEATURE_COLUMNS = [
    "url_length", "hostname_length", "path_length", "query_length", "dot_count",
    "hyphen_count", "at_symbol", "digit_count", "subdomain_count", "has_ip",
    "uses_https", "suspicious_keyword_count", "punycode",
]
SUSPICIOUS_WORDS = (
    "login", "verify", "update", "secure", "account", "password", "wallet",
    "confirm", "signin", "bank", "urgent", "recover", "bonus", "gift",
)


def extract_url_features(url: str) -> dict[str, float]:
    """Extract simple lexical URL features; these are indicators, not proof of a threat."""
    raw = str(url).strip()
    candidate = raw if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw) else f"http://{raw}"
    parsed = urlparse(candidate)
    host = (parsed.hostname or "").lower()
    try:
        ipaddress.ip_address(host)
        has_ip = 1
    except ValueError:
        has_ip = 0
    return {
        "url_length": len(raw),
        "hostname_length": len(host),
        "path_length": len(parsed.path or ""),
        "query_length": len(parsed.query or ""),
        "dot_count": raw.count("."),
        "hyphen_count": raw.count("-"),
        "at_symbol": int("@" in raw),
        "digit_count": sum(char.isdigit() for char in raw),
        "subdomain_count": max(0, len(host.split(".")) - 2),
        "has_ip": has_ip,
        "uses_https": int(parsed.scheme.lower() == "https"),
        "suspicious_keyword_count": sum(word in raw.lower() for word in SUSPICIOUS_WORDS),
        "punycode": int("xn--" in host),
    }


def normalize_label(value: object) -> int | None:
    text = str(value).strip().lower()
    if text in {"1", "phishing", "malicious", "bad", "suspicious", "spam", "unsafe"}:
        return 1
    if text in {"0", "legitimate", "benign", "safe", "good", "ham", "legit"}:
        return 0
    return None


def get_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def prepare_dataset(df: pd.DataFrame, url_column: str, label_column: str) -> tuple[pd.DataFrame, pd.Series]:
    cleaned = df[[url_column, label_column]].dropna().copy()
    cleaned[url_column] = cleaned[url_column].astype(str).str.strip()
    cleaned["_target"] = cleaned[label_column].map(normalize_label)
    cleaned = cleaned[(cleaned[url_column] != "") & cleaned["_target"].notna()]
    X = pd.DataFrame([extract_url_features(value) for value in cleaned[url_column]])
    y = cleaned["_target"].astype(int)
    return X, y


st.set_page_config(page_title="PhishGuard | URL Risk Analyzer", page_icon="🛡️", layout="wide")
st.title("🛡️ PhishGuard")
st.subheader("Phishing URL Detection with Machine Learning")
st.write("Train a baseline model on labeled URLs, inspect evaluation metrics, and screen URLs. A model prediction is a risk signal—not a guarantee that a URL is safe or malicious.")
st.info("The bundled CSV contains synthetic demonstration examples on reserved/test-style domains. Replace it with a properly sourced, labeled dataset before drawing real-world conclusions.")

with st.sidebar:
    st.header("1. Dataset")
    uploaded = st.file_uploader("Upload a labeled URL CSV", type=["csv"], help="Expected columns: url and label. Labels can be phishing/legitimate or 1/0.")
    if uploaded:
        raw_df = pd.read_csv(uploaded)
    else:
        raw_df = pd.read_csv(DATA_PATH)
        st.caption("Using bundled sample_urls.csv")

    st.write("Dataset preview")
    st.dataframe(raw_df.head(6), use_container_width=True, hide_index=True)
    url_guess = get_column(raw_df, ["url", "link", "website", "uri"])
    label_guess = get_column(raw_df, ["label", "class", "result", "target", "status"])
    column_names = list(raw_df.columns)
    if not column_names:
        st.error("The CSV has no columns.")
        st.stop()
    url_col = st.selectbox("URL column", column_names, index=column_names.index(url_guess) if url_guess in column_names else 0)
    label_col = st.selectbox("Label column", column_names, index=column_names.index(label_guess) if label_guess in column_names else min(1, len(column_names) - 1))

try:
    X, y = prepare_dataset(raw_df, url_col, label_col)
except Exception as exc:
    st.error(f"Could not prepare this dataset: {exc}")
    st.stop()

if len(X) < 12 or y.nunique() < 2 or y.value_counts().min() < 2:
    st.error("Provide at least 12 valid rows and examples from both classes. Use the label values phishing/legitimate or 1/0.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=160, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

m1, m2, m3 = st.columns(3)
m1.metric("Valid labeled URLs", f"{len(X):,}")
m2.metric("Test accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
m3.metric("Phishing examples", f"{int(y.sum()):,}")

with st.expander("Evaluation details", expanded=False):
    report = classification_report(y_test, y_pred, labels=[0, 1], target_names=["Legitimate", "Phishing"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    st.write("Confusion matrix — rows are actual classes, columns are predicted classes.")
    st.dataframe(pd.DataFrame(cm, index=["Actual legitimate", "Actual phishing"], columns=["Predicted legitimate", "Predicted phishing"]), use_container_width=True)

st.divider()
st.header("2. Check a URL")
url_input = st.text_input("URL to analyze", placeholder="https://example.com/account")
if st.button("Analyze URL", type="primary"):
    if not url_input.strip():
        st.warning("Enter a URL first.")
    else:
        features = pd.DataFrame([extract_url_features(url_input)])[FEATURE_COLUMNS]
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        phishing_probability = float(probabilities[list(model.classes_).index(1)])
        if prediction == 1:
            st.error(f"Model assessment: potentially suspicious — estimated phishing probability {phishing_probability:.1%}")
        else:
            st.success(f"Model assessment: likely legitimate according to this model — estimated phishing probability {phishing_probability:.1%}")
        st.caption("Do not open unknown links based only on this result. The model may be wrong, particularly when trained on small or synthetic datasets.")
        st.dataframe(features, use_container_width=True)

st.divider()
st.header("3. Batch URL screening")
batch_file = st.file_uploader("Upload CSV with a URL column", type=["csv"], key="batch_upload")
if batch_file:
    batch_df = pd.read_csv(batch_file)
    batch_url_col = get_column(batch_df, ["url", "link", "website", "uri"])
    if not batch_url_col:
        st.error("Could not find a URL column. Name it url, link, website, or uri.")
    else:
        features = pd.DataFrame([extract_url_features(v) for v in batch_df[batch_url_col].fillna("").astype(str)])
        batch_df["prediction"] = ["potentially_phishing" if p == 1 else "likely_legitimate" for p in model.predict(features[FEATURE_COLUMNS])]
        batch_df["phishing_probability"] = model.predict_proba(features[FEATURE_COLUMNS])[:, list(model.classes_).index(1)]
        st.dataframe(batch_df.head(25), use_container_width=True, hide_index=True)
        st.download_button("Download URL screening results", batch_df.to_csv(index=False).encode("utf-8"), file_name="phishguard_predictions.csv", mime="text/csv")
