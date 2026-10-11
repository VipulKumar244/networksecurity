import io
import os
import pickle

import pandas as pd
import streamlit as st

from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.utils.ml_utils.model.estimator import NetworkModel


MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_model", "model.pkl")

st.set_page_config(
    page_title="Network Security Classifier",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    if not isinstance(model, NetworkModel):
        raise TypeError("The saved model is not a NetworkModel instance")
    return model


st.title("🛡️ Network Security Classifier")
st.write(
    "Upload a CSV of network records to classify them with the trained "
    "phishing detection model."
)

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        source_df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        st.stop()

    st.subheader("Data preview")
    st.caption(f"{len(source_df):,} rows · {len(source_df.columns)} columns")
    st.dataframe(source_df.head(10), use_container_width=True)

    if st.button("Run prediction", type="primary"):
        try:
            model = load_model()
            features = source_df.drop(columns=[TARGET_COLUMN], errors="ignore")
            predictions = model.predict(features)
            result_df = source_df.copy()
            result_df["predicted_column"] = predictions

            st.subheader("Prediction results")
            total_col, class_values_col = st.columns([1, 2])
            total_col.metric("Records classified", f"{len(result_df):,}")
            class_counts = pd.Series(predictions).value_counts().sort_index()
            for label, count in class_counts.items():
                class_values_col.metric(f"Predicted class {label}", f"{count:,}")

            st.dataframe(result_df, use_container_width=True)
            st.download_button(
                "Download predictions as CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="network_security_predictions.csv",
                mime="text/csv",
            )
        except FileNotFoundError:
            st.error("The trained model file is missing: final_model/model.pkl")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.divider()
st.caption("Demo interface for the NetworkSecurity machine learning project.")
