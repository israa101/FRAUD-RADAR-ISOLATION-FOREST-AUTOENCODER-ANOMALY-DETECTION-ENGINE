"""
Credit Card Fraud Detection — Isolation Forest + Autoencoder
Neon Green Streamlit App
------------------------------------------------
Run locally with:
    streamlit run app.py

This app LOADS the models saved by the notebook (no retraining):
    isolation_forest_model.pkl   (final tuned Isolation Forest)
    model_features.pkl           (feature columns the model expects)
    autoencoder_model.keras      (optional)
    autoencoder_scaler.pkl       (optional)
    autoencoder_threshold.pkl    (optional)
Keep them in the same folder as app.py. Upload creditcard.csv from the
sidebar to evaluate the saved models on real data.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.inspection import permutation_importance

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Radar | Anomaly Detection",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# NEON GREEN THEME (custom CSS)
# ----------------------------------------------------------------------
NEON = "#39ff14"
NEON_SOFT = "#8dff7a"
BG_DARK = "#050807"
BG_PANEL = "#0c130f"

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Share Tech Mono', monospace;
        }}

        .stApp {{
            background: radial-gradient(circle at 10% 0%, #0d1a10 0%, {BG_DARK} 55%, #000000 100%);
            color: #d9ffe0;
        }}

        /* Titles */
        h1, h2, h3, h4 {{
            font-family: 'Orbitron', sans-serif !important;
            color: {NEON} !important;
            text-shadow: 0 0 6px {NEON}, 0 0 18px rgba(57,255,20,0.55);
            letter-spacing: 1px;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #061109 0%, #030502 100%);
            border-right: 1px solid rgba(57,255,20,0.35);
        }}
        section[data-testid="stSidebar"] * {{
            color: {NEON_SOFT} !important;
        }}

        /* Metric cards */
        div[data-testid="stMetric"] {{
            background: {BG_PANEL};
            border: 1px solid rgba(57,255,20,0.45);
            border-radius: 14px;
            padding: 14px 10px;
            box-shadow: 0 0 14px rgba(57,255,20,0.12), inset 0 0 20px rgba(57,255,20,0.04);
        }}
        div[data-testid="stMetricValue"] {{
            color: {NEON} !important;
            text-shadow: 0 0 8px rgba(57,255,20,0.7);
        }}
        div[data-testid="stMetricLabel"] {{
            color: {NEON_SOFT} !important;
        }}

        /* Buttons */
        .stButton>button, .stDownloadButton>button {{
            background: linear-gradient(135deg, #0b2a10, #06150a);
            color: {NEON};
            border: 1px solid {NEON};
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(57,255,20,0.35);
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 1px;
            transition: 0.2s ease-in-out;
        }}
        .stButton>button:hover, .stDownloadButton>button:hover {{
            box-shadow: 0 0 22px rgba(57,255,20,0.85);
            color: #ffffff;
            border-color: #ffffff;
        }}

        /* Tabs */
        button[data-baseweb="tab"] {{
            color: {NEON_SOFT} !important;
            font-family: 'Orbitron', sans-serif;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {NEON} !important;
            border-bottom: 3px solid {NEON} !important;
            text-shadow: 0 0 8px rgba(57,255,20,0.7);
        }}

        /* Dataframe */
        .stDataFrame {{
            border: 1px solid rgba(57,255,20,0.3);
            border-radius: 10px;
        }}

        /* Alert boxes */
        div[data-testid="stAlert"] {{
            border: 1px solid rgba(57,255,20,0.4);
            border-radius: 10px;
        }}

        /* Neon divider */
        .neon-divider {{
            height: 1px;
            background: linear-gradient(90deg, transparent, {NEON}, transparent);
            box-shadow: 0 0 10px {NEON};
            margin: 0.6rem 0 1.2rem 0;
        }}

        .fraud-badge {{
            display:inline-block;
            padding: 6px 16px;
            border-radius: 999px;
            font-family: 'Orbitron', sans-serif;
            font-weight: 700;
            letter-spacing: 1px;
        }}
        .badge-safe {{
            color: {NEON};
            border: 1px solid {NEON};
            box-shadow: 0 0 14px rgba(57,255,20,0.6);
        }}
        .badge-fraud {{
            color: #ff4d6d;
            border: 1px solid #ff4d6d;
            box-shadow: 0 0 14px rgba(255,77,109,0.6);
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG_DARK,
    plot_bgcolor=BG_DARK,
    font=dict(color="#d9ffe0", family="Share Tech Mono"),
    colorway=[NEON, "#ff4d6d", NEON_SOFT, "#00e5ff"],
    xaxis=dict(gridcolor="rgba(57,255,20,0.12)", zerolinecolor="rgba(57,255,20,0.25)"),
    yaxis=dict(gridcolor="rgba(57,255,20,0.12)", zerolinecolor="rgba(57,255,20,0.25)"),
)

# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_synthetic_data(n_normal: int = 20000, n_fraud: int = 60, seed: int = 42) -> pd.DataFrame:
    """Synthetic stand-in with the same schema as the Kaggle dataset,
    used only when no real creditcard.csv is provided."""
    rng = np.random.default_rng(seed)
    n_features = 28

    normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, n_features))
    fraud = rng.normal(loc=0.0, scale=1.0, size=(n_fraud, n_features))
    # Push a few components to make fraud "anomalous", mirroring V14/V17 patterns
    fraud[:, 13] -= rng.uniform(3, 7, size=n_fraud)   # V14-like
    fraud[:, 16] -= rng.uniform(3, 6, size=n_fraud)   # V17-like
    fraud[:, 11] += rng.uniform(2, 5, size=n_fraud)   # V12-like

    amount_normal = np.round(rng.exponential(scale=60, size=n_normal), 2)
    amount_fraud = np.round(rng.exponential(scale=25, size=n_fraud), 2)

    time_normal = rng.integers(0, 172800, size=n_normal)
    time_fraud = rng.integers(0, 172800, size=n_fraud)

    cols = [f"V{i}" for i in range(1, n_features + 1)]
    df_normal = pd.DataFrame(normal, columns=cols)
    df_normal["Time"] = time_normal
    df_normal["Amount"] = amount_normal
    df_normal["Class"] = 0

    df_fraud = pd.DataFrame(fraud, columns=cols)
    df_fraud["Time"] = time_fraud
    df_fraud["Amount"] = amount_fraud
    df_fraud["Class"] = 1

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["Hour"] = (df["Time"] // 3600) % 24
    ordered = ["Time"] + cols + ["Amount", "Hour", "Class"]
    return df[ordered]


@st.cache_data(show_spinner=False)
def load_uploaded(file_bytes: bytes) -> pd.DataFrame:
    """Read the uploaded CSV, clean it and keep numeric columns only."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    df = df.drop_duplicates().reset_index(drop=True)

    # Keep only numeric columns (the model can't use text columns)
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        df = df.drop(columns=non_numeric)

    # Fill missing values with the median so training never crashes
    df = df.fillna(df.median(numeric_only=True))

    if "Time" in df.columns:
        df["Hour"] = (df["Time"] // 3600) % 24
    return df


APP_DIR = os.path.dirname(os.path.abspath(__file__))
IF_MODEL_PATH = os.path.join(APP_DIR, "isolation_forest_model.pkl")
IF_FEATURES_PATH = os.path.join(APP_DIR, "model_features.pkl")
AE_MODEL_PATH = os.path.join(APP_DIR, "autoencoder_model.keras")
AE_SCALER_PATH = os.path.join(APP_DIR, "autoencoder_scaler.pkl")
AE_THRESHOLD_PATH = os.path.join(APP_DIR, "autoencoder_threshold.pkl")


@st.cache_resource(show_spinner=False)
def load_isolation_forest():
    """Load the final saved Isolation Forest + its feature list."""
    model = joblib.load(IF_MODEL_PATH)
    features = list(joblib.load(IF_FEATURES_PATH))
    return model, features


@st.cache_resource(show_spinner=False)
def load_autoencoder():
    """Load the saved Autoencoder bundle. Returns None if files/TensorFlow are missing."""
    if not all(os.path.exists(p) for p in (AE_MODEL_PATH, AE_SCALER_PATH, AE_THRESHOLD_PATH)):
        return None
    try:
        from tensorflow.keras.models import load_model
        ae = load_model(AE_MODEL_PATH, compile=False)
        scaler = joblib.load(AE_SCALER_PATH)
        thr = float(joblib.load(AE_THRESHOLD_PATH))
        return ae, scaler, thr
    except Exception:
        return None


def make_data_signature(df: pd.DataFrame, data_source: str) -> str:
    """Small fingerprint of the dataset, used to know when data changed."""
    h = pd.util.hash_pandas_object(df.head(500), index=False).sum()
    return f"{data_source}|{df.shape[0]}|{df.shape[1]}|{h}"


# ----------------------------------------------------------------------
# SIDEBAR — DATA SOURCE + HYPERPARAMETERS
# ----------------------------------------------------------------------
st.sidebar.markdown("## 🟢 CONTROL PANEL")
st.sidebar.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

st.sidebar.markdown("### 📂 Dataset")
uploaded_file = st.sidebar.file_uploader("Upload creditcard.csv", type=["csv"])

if uploaded_file is not None:
    try:
        df = load_uploaded(uploaded_file.getvalue())
        data_source = f"Uploaded: {uploaded_file.name}"
    except Exception as e:
        st.sidebar.error(f"Could not read the file: {e}")
        st.error("The uploaded file could not be read. Please upload a valid CSV file.")
        st.stop()
else:
    df = generate_synthetic_data()
    data_source = "Synthetic demo dataset"
    st.sidebar.caption("No file uploaded — using a synthetic demo dataset with the same schema (Time, V1–V28, Amount, Class).")

st.sidebar.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown("### 🧠 Saved Models")
if not (os.path.exists(IF_MODEL_PATH) and os.path.exists(IF_FEATURES_PATH)):
    st.sidebar.error("isolation_forest_model.pkl / model_features.pkl not found next to app.py")
    st.error("Saved model files were not found. Run the notebook's save cell and put the .pkl files in the same folder as app.py.")
    st.stop()

model, model_features = load_isolation_forest()
ae_bundle = load_autoencoder()

st.sidebar.success("✅ Isolation Forest loaded")
st.sidebar.caption(
    f"n_estimators={model.n_estimators} · max_samples={model.max_samples} · "
    f"max_features={model.max_features} · bootstrap={model.bootstrap}"
)
if ae_bundle is not None:
    st.sidebar.success("✅ Autoencoder loaded")
else:
    st.sidebar.warning("Autoencoder files not found (optional) — run the notebook's Autoencoder save cell to enable it.")

st.sidebar.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
st.sidebar.caption("Models are loaded from disk — nothing is retrained here.")


# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🟢 FRAUD RADAR</h1>"
    "<p style='text-align:center; color:#8dff7a; letter-spacing:3px;'>ISOLATION FOREST + AUTOENCODER · ANOMALY DETECTION ENGINE</p>",
    unsafe_allow_html=True,
)
st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# PREP (use the saved feature list) + SCORE with the loaded models
# ----------------------------------------------------------------------
missing = [c for c in model_features if c not in df.columns]
if missing:
    st.error(f"The dataset is missing columns the saved model expects: {missing}")
    st.stop()

if len(df) < 50:
    st.error("The dataset is too small (need at least 50 rows).")
    st.stop()

has_labels = "Class" in df.columns and df["Class"].nunique() > 1

X = df[model_features]
if has_labels:
    y = df["Class"].astype(int)
    # Same split as the notebook (80/20, random_state=42, stratified) -> same test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
else:
    y = None
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    y_train = y_test = None

data_sig = make_data_signature(df, data_source)

st.success(f"✅ Using the saved Isolation Forest — evaluated on **{data_source}** · "
           f"{len(X_test):,} test rows · {len(model_features)} features")

test_scores = -model.decision_function(X_test)
test_pred = np.where(model.predict(X_test) == -1, 1, 0)

# Autoencoder scores (if available)
ae_scores = ae_pred = None
if ae_bundle is not None:
    ae_model, ae_scaler, ae_threshold = ae_bundle
    def ae_error(frame: pd.DataFrame) -> np.ndarray:
        scaled = ae_scaler.transform(frame[model_features])
        recon = ae_model.predict(scaled, verbose=0)
        return np.mean(np.square(scaled - recon), axis=1)
    ae_scores = ae_error(X_test)
    ae_pred = (ae_scores > ae_threshold).astype(int)


# ----------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------
tab_overview, tab_perf, tab_ae, tab_importance, tab_predict = st.tabs(
    ["📊 Overview", "🎯 Performance", "🤖 Autoencoder", "🧬 Feature Importance", "🔎 Score a Transaction"]
)

# ---------------- OVERVIEW ----------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data source", data_source)
    c2.metric("Total transactions", f"{len(df):,}")
    if "Class" in df.columns:
        c3.metric("Fraud cases", f"{int(df['Class'].sum()):,}")
        c4.metric("Fraud rate", f"{df['Class'].mean()*100:.3f}%")
    else:
        c3.metric("Fraud cases", "n/a")
        c4.metric("Fraud rate", "n/a")

    st.markdown("### Class Distribution")
    if "Class" in df.columns:
        counts = df["Class"].value_counts().rename({0: "Legit", 1: "Fraud"})
        fig = px.bar(
            x=counts.index, y=counts.values, log_y=True, text=counts.values,
            labels={"x": "Class", "y": "Count (log scale)"},
        )
        fig.update_traces(marker_color=[NEON, "#ff4d6d"])
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")

    if "Amount" in df.columns:
        st.markdown("### Amount Distribution (log1p)")
        plot_df = pd.DataFrame({"log1p_Amount": np.log1p(df["Amount"].clip(lower=0))})
        if "Class" in df.columns:
            plot_df["Class"] = df["Class"].astype(str)
        fig2 = px.histogram(plot_df, x="log1p_Amount", nbins=50,
                            color="Class" if "Class" in plot_df.columns else None,
                            barmode="overlay")
        fig2.update_layout(**PLOTLY_LAYOUT)
        fig2.update_layout(xaxis_title="log1p(Amount)")
        st.plotly_chart(fig2, width="stretch")

    if "Hour" in df.columns and "Class" in df.columns:
        st.markdown("### Fraud Cases by Hour of Day")
        hourly = df[df["Class"] == 1]["Hour"].value_counts().sort_index()
        fig3 = px.bar(x=hourly.index, y=hourly.values, labels={"x": "Hour", "y": "Fraud count"})
        fig3.update_traces(marker_color=NEON)
        fig3.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig3, width="stretch")

    with st.expander("Preview raw data"):
        st.dataframe(df.head(50), width="stretch")


# ---------------- PERFORMANCE ----------------
with tab_perf:
    if not has_labels:
        st.info("Upload a dataset that includes the `Class` column to see evaluation metrics.")
    else:
        pr_auc = average_precision_score(y_test, test_scores)
        roc_auc = roc_auc_score(y_test, test_scores)
        cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PR-AUC", f"{pr_auc:.4f}")
        c2.metric("ROC-AUC", f"{roc_auc:.4f}")
        c3.metric("Fraud Recall", f"{(tp/(tp+fn) if (tp+fn) else 0):.2%}")
        c4.metric("Fraud Precision", f"{(tp/(tp+fp) if (tp+fp) else 0):.2%}")

        colA, colB = st.columns(2)

        with colA:
            st.markdown("#### Confusion Matrix")
            cm_fig = px.imshow(
                cm, text_auto=True, color_continuous_scale=["#050807", NEON],
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Normal", "Fraud"], y=["Normal", "Fraud"],
            )
            cm_fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(cm_fig, width="stretch")

        with colB:
            st.markdown("#### ROC & Precision-Recall Curves")
            fpr, tpr, _ = roc_curve(y_test, test_scores)
            prec, rec, _ = precision_recall_curve(y_test, test_scores)

            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=NEON, width=3), name="ROC"))
            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                          line=dict(color="gray", dash="dash"), name="Random"))
            roc_fig.update_layout(**PLOTLY_LAYOUT)
            roc_fig.update_layout(
            xaxis_title="FPR",
            yaxis_title="TPR",
            title="ROC Curve",
            height=280,
            margin=dict(t=40, b=10)
            )
            st.plotly_chart(roc_fig, width="stretch")

            pr_fig = go.Figure()
            pr_fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", line=dict(color=NEON_SOFT, width=3), name="PR"))
            pr_fig.update_layout(**PLOTLY_LAYOUT)
            pr_fig.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            title="Precision-Recall Curve",
            height=280,
            margin=dict(t=40, b=10)
            )
            st.plotly_chart(pr_fig, width="stretch")

        st.markdown("#### Classification Report")
        report = classification_report(
            y_test, test_pred, labels=[0, 1], target_names=["Normal", "Fraud"],
            digits=4, output_dict=True, zero_division=0,
        )
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.style.format("{:.4f}"), width="stretch")


# ---------------- AUTOENCODER ----------------
with tab_ae:
    if ae_bundle is None:
        st.info("Autoencoder files not found. Run the notebook's Autoencoder save cell "
                "(autoencoder_model.keras, autoencoder_scaler.pkl, autoencoder_threshold.pkl) "
                "and put them next to app.py.")
    elif not has_labels:
        st.info("Upload a dataset that includes the `Class` column to see Autoencoder metrics.")
    else:
        a_pr = average_precision_score(y_test, ae_scores)
        a_roc = roc_auc_score(y_test, ae_scores)
        acm = confusion_matrix(y_test, ae_pred, labels=[0, 1])
        atn, afp, afn, atp = acm.ravel()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PR-AUC", f"{a_pr:.4f}")
        c2.metric("ROC-AUC", f"{a_roc:.4f}")
        c3.metric("Fraud Recall", f"{(atp/(atp+afn) if (atp+afn) else 0):.2%}")
        c4.metric("Fraud Precision", f"{(atp/(atp+afp) if (atp+afp) else 0):.2%}")
        st.caption(f"Reconstruction-error threshold (from notebook): {ae_threshold:.4f}")

        colA, colB = st.columns(2)
        with colA:
            st.markdown("#### Confusion Matrix")
            acm_fig = px.imshow(
                acm, text_auto=True, color_continuous_scale=["#050807", NEON],
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Normal", "Fraud"], y=["Normal", "Fraud"],
            )
            acm_fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(acm_fig, width="stretch")
        with colB:
            st.markdown("#### Reconstruction Error Distribution")
            hist_df = pd.DataFrame({
                "error": np.clip(ae_scores, 0, np.percentile(ae_scores, 99.5)),
                "Class": np.where(y_test.values == 1, "Fraud", "Normal"),
            })
            hfig = px.histogram(hist_df, x="error", color="Class", nbins=100,
                                barmode="overlay", log_y=True,
                                color_discrete_map={"Normal": NEON, "Fraud": "#ff4d6d"})
            hfig.add_vline(x=ae_threshold, line_dash="dash", line_color="#ffffff")
            hfig.update_layout(**PLOTLY_LAYOUT)
            hfig.update_layout(xaxis_title="Reconstruction error", yaxis_title="Count (log)")
            st.plotly_chart(hfig, width="stretch")

        st.markdown("#### Classification Report")
        arep = pd.DataFrame(classification_report(
            y_test, ae_pred, labels=[0, 1], target_names=["Normal", "Fraud"],
            digits=4, output_dict=True, zero_division=0)).T
        st.dataframe(arep.style.format("{:.4f}"), width="stretch")


# ---------------- FEATURE IMPORTANCE ----------------
with tab_importance:
    if not has_labels:
        st.info("Upload a dataset that includes the `Class` column to compute feature importance.")
    else:
        st.markdown(
            "Permutation Importance is computed on the **test set** using PR-AUC as the scoring metric "
            "(Isolation Forest has no built-in `feature_importances_`). "
            "A higher value means the model relies more on that feature to catch fraud."
        )

        n_repeats = st.slider("Number of repeats (more = more accurate, but slower)", 1, 10, 3)
        max_rows = st.slider("Max test rows used (to keep it fast)", 2000, 50000, 10000, step=1000)

        if st.button("Compute Permutation Importance"):
            # Keep ALL fraud rows and sample the normal rows, so PR-AUC stays meaningful
            fraud_idx = y_test[y_test == 1].index
            normal_idx = y_test[y_test == 0].index
            n_normal_keep = max(max_rows - len(fraud_idx), 0)
            if len(normal_idx) > n_normal_keep:
                normal_idx = pd.Index(
                    np.random.default_rng(42).choice(normal_idx, size=n_normal_keep, replace=False)
                )
            sample_idx = fraud_idx.union(normal_idx)
            X_eval = X_test.loc[sample_idx]
            y_eval = y_test.loc[sample_idx]

            def pr_auc_scorer(estimator, Xs, ys):
                scores = -estimator.decision_function(Xs)
                return average_precision_score(ys, scores)

            try:
                with st.spinner("Permuting features..."):
                    perm = permutation_importance(
                        model, X_eval, y_eval, scoring=pr_auc_scorer,
                        n_repeats=n_repeats, random_state=42, n_jobs=1,
                    )
                st.session_state.imp_df = pd.DataFrame({
                    "Feature": X_eval.columns,
                    "Importance": perm.importances_mean,
                    "Std": perm.importances_std,
                }).sort_values("Importance", ascending=False)
                st.session_state.imp_sig = data_sig
            except Exception as e:
                st.error(f"Could not compute feature importance: {e}")

        # Show the last result (kept in session so it survives reruns) if it matches this data
        if "imp_df" in st.session_state and st.session_state.get("imp_sig") == data_sig:
            imp_df = st.session_state.imp_df.head(15)
            fig = px.bar(
                imp_df.sort_values("Importance"),
                x="Importance", y="Feature", orientation="h",
                error_x="Std",
            )
            fig.update_traces(marker_color=NEON)
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_layout(title="Top 15 Features (PR-AUC drop)", height=520)
            st.plotly_chart(fig, width="stretch")
            with st.expander("Show full table"):
                st.dataframe(st.session_state.imp_df, width="stretch")
        else:
            st.caption("Click the button above — permutation importance is compute-heavy so it isn't run automatically.")


# ---------------- SCORE A TRANSACTION ----------------
with tab_predict:
    st.markdown("### Score a single transaction")
    st.caption("Adjust the values below (defaults come from the feature medians of your data) "
               "and check whether the model flags it as anomalous.")

    medians = X[model_features].median(numeric_only=True)

    # Quick helpers so you can test the model without typing 30 numbers
    mode = st.radio(
        "Start from",
        ["Median (typical) values", "A real fraud example", "A real normal example"],
        horizontal=True,
        disabled=not has_labels,
    )

    base_values = medians.to_dict()
    if has_labels and mode != "Median (typical) values":
        target_class = 1 if mode == "A real fraud example" else 0
        pool = df[df["Class"] == target_class]
        if len(pool) > 0:
            base_values = pool.iloc[0][model_features].to_dict()

    # A new key prefix per data/mode, so number_inputs refresh their defaults
    key_prefix = f"{abs(hash((data_sig, mode))) % 10**8}"

    with st.form("score_form"):
        cols = st.columns(4)
        input_values = {}
        for i, feat in enumerate(model_features):
            with cols[i % 4]:
                default_val = float(base_values.get(feat, 0.0))
                if not np.isfinite(default_val):
                    default_val = 0.0
                input_values[feat] = st.number_input(
                    feat, value=default_val, format="%.4f", key=f"in_{key_prefix}_{feat}"
                )
        submitted = st.form_submit_button("⚡ Score Transaction", width="stretch")

    if submitted:
        row = pd.DataFrame([input_values])[model_features]
        score = float(-model.decision_function(row)[0])
        pred = int(model.predict(row)[0])
        is_fraud = pred == -1

        # Threshold the model itself uses to call something an anomaly
        threshold = float(-model.offset_)

        ref_scores = np.concatenate([test_scores, [score]])
        min_ref, max_ref = float(ref_scores.min()), float(ref_scores.max())
        gauge_val = float(np.clip((score - min_ref) / (max_ref - min_ref + 1e-9), 0, 1) * 100)

        colL, colR = st.columns([1, 1])
        with colL:
            badge_class = "badge-fraud" if is_fraud else "badge-safe"
            badge_text = "⚠ ANOMALY — LIKELY FRAUD" if is_fraud else "✓ NORMAL TRANSACTION"
            st.markdown(f"<span class='fraud-badge {badge_class}'>{badge_text}</span>", unsafe_allow_html=True)
            st.metric("Anomaly score (higher = more suspicious)", f"{score:.4f}")
            st.caption(f"Decision threshold: {threshold:.4f} — scores above it are flagged as anomalies.")

        with colR:
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gauge_val,
                number={"suffix": "%", "font": {"color": NEON}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": NEON_SOFT},
                    "bar": {"color": "#ff4d6d" if is_fraud else NEON},
                    "bgcolor": BG_PANEL,
                    "borderwidth": 1,
                    "bordercolor": NEON,
                },
                title={"text": "Relative Suspicion Level", "font": {"color": NEON_SOFT}},
            ))
            gauge_fig.update_layout(**PLOTLY_LAYOUT)
            gauge_fig.update_layout(height=260, margin=dict(t=40, b=10))
            st.plotly_chart(gauge_fig, width="stretch")

        if ae_bundle is not None:
            ae_err = float(ae_error(row)[0])
            ae_flag = ae_err > ae_threshold
            st.markdown("#### Autoencoder verdict")
            b_class = "badge-fraud" if ae_flag else "badge-safe"
            b_text = "⚠ ANOMALY (Autoencoder)" if ae_flag else "✓ NORMAL (Autoencoder)"
            st.markdown(f"<span class='fraud-badge {b_class}'>{b_text}</span>", unsafe_allow_html=True)
            st.metric("Reconstruction error", f"{ae_err:.4f}", delta=f"threshold {ae_threshold:.4f}", delta_color="off")

st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
st.caption("Fraud Radar · Isolation Forest + Autoencoder demo app · Built for the Credit Card Fraud Detection assignment.")