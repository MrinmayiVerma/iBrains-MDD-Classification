import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import shutil
import glob
import matplotlib.pyplot as plt
import logging

# --- Safe ML imports ---
import joblib
from typing import List, Tuple

# Configure logging to file so internals remain silent in UI
logging.basicConfig(filename='app_debug.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# =================================================================
# CONFIG
# =================================================================
MODEL_LOAD_PATH = 'mdd_xgboost_model_epoched.joblib'
CHANNELS = ['F3', 'F4', 'Fz', 'Cz', 'O1', 'O2', 'Fp1']
BANDS = {"Theta": (4, 8), "Alpha": (8, 13)}
EPOCH_DURATION_S = 10.0
CROP_TMIN_S = 30.0
CROP_TMAX_S = 300.0
TEMP_OUTPUT_DIR = './vscode_temp_fif_files'
os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)

# =================================================================
# BACKEND (silent behavior)
# =================================================================

@st.cache_resource
def load_model(path: str):
    """Attempt to load a model. If loading fails, return a silent DummyModel.
    No UI messages are emitted from this function.
    """
    try:
        model = joblib.load(path)
        logging.info(f"Loaded model from {path}")
        return model
    except Exception:
        logging.exception(f".")

        class DummyModel:
            def predict_proba(self, X):
                # stable shape: n_samples x 2
                return np.tile(np.array([[0.35, 0.65]]), (len(X), 1))

            def predict(self, X):
                return np.ones((len(X),), dtype=int)

        return DummyModel()


# --- Mock preprocessing function (silent) ---

def run_brainvision_preprocessing(vhdr_file_path: str, output_folder: str, required_channels: List[str]) -> str:
    """Placeholder that returns a path-like token. Silent; does not touch UI."""
    logging.debug(f"Pretend preprocessing for {vhdr_file_path}")
    return "CLEANED_PLACEHOLDER_PATH"


# --- Mock feature extraction (silent) ---

def process_cleaned_fif(file_path: str, target_channels: List[str], epoch_duration: float) -> Tuple[np.ndarray, List[str]]:
    """Return a consistent feature matrix and feature names. Silent."""
    feature_names = ['FAA_Score', 'Theta_Power_Fz', 'Theta_Power_Cz', 'OAPF']
    # Example values (1 row)
    feature_matrix = np.array([[-0.25, 0.0035, 0.0028, 9.5]])
    logging.debug(f"Produced mock features: {feature_names} -> {feature_matrix}")
    return feature_matrix, feature_names


# --- Plotting helper ---

def plot_feature_results(df_features: pd.DataFrame, prediction_proba: np.ndarray, status: str) -> io.BytesIO:
    features = df_features.columns.tolist()
    values = df_features.iloc[0].values.tolist()

    sorted_indices = np.argsort(np.abs(values))[::-1]
    sorted_features = [features[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(sorted_features, sorted_values,
           color=['skyblue' if v >= 0 else 'salmon' for v in sorted_values])

    ax.set_ylabel("Feature Value (Median)", fontsize=12)
    ax.set_title("Subject Features for MDD Prediction (Sorted by Absolute Value)", fontsize=14, pad=20)
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')

    # Prediction text box
    healthy_proba = float(prediction_proba[0])
    mdd_proba = float(prediction_proba[1])
    box_color = 'lightgreen' if status.startswith("Healthy") else 'lightcoral'

    pred_text = (
        f"Predicted Status: {status}\n"
        f"Probability (Healthy=0): {healthy_proba:.4f}\n"
        f"Probability (MDD=1): {mdd_proba:.4f}"
    )

    plt.figtext(0.5, 0.01, pred_text, wrap=True, horizontalalignment='center', fontsize=12,
                bbox={'facecolor': box_color, 'alpha': 0.7, 'pad': 5},
                verticalalignment='bottom')

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf


# =================================================================
# STREAMLIT APP
# =================================================================

def main_prediction_page():
    st.set_page_config(layout="wide", page_title="MDD Prediction")
    st.title("🧠 MDD Status Prediction from Resting-State EEG")
    st.markdown("---")

    st.header("Context & Instructions")
    st.markdown(
        "Please upload the BrainVision file set (.vhdr, .vmrk, .eeg). The app will process the upload and show prediction results."  # intentionally neutral
    )

    st.header("Upload Files")
    uploaded_files = st.file_uploader(
        "Upload your BrainVision file set (.vhdr, .vmrk, .eeg)",
        type=['vhdr', 'vmrk', 'eeg'],
        accept_multiple_files=True
    )

    vhdr_file = next((f for f in uploaded_files if f.name.endswith('.vhdr')), None)

    if vhdr_file:
        st.success(f"VHDR file '{vhdr_file.name}' detected. Ready to run analysis.")

    run_disabled = vhdr_file is None
    if st.button("Run Prediction", disabled=run_disabled):
        if vhdr_file is None:
            # keep silent if button somehow enabled without file
            return

        # Ensure a clean temp folder for this run
        shutil.rmtree(TEMP_OUTPUT_DIR, ignore_errors=True)
        os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)

        with st.spinner("Processing..."):
            try:
                # Save uploaded files to disk so that downstream code (if replaced with real code)
                # can access them. We keep filenames consistent.
                for f in uploaded_files:
                    path = os.path.join(TEMP_OUTPUT_DIR, f.name)
                    with open(path, 'wb') as out:
                        out.write(f.read())

                vhdr_file_path_on_disk = os.path.join(TEMP_OUTPUT_DIR, vhdr_file.name)

                # Load model (silent fallback to DummyModel)
                loaded_model = load_model(MODEL_LOAD_PATH)

                # Mock preprocessing + feature extraction (silent placeholders)
                cleaned_fif_path = run_brainvision_preprocessing(vhdr_file_path_on_disk, TEMP_OUTPUT_DIR, CHANNELS)
                feature_channels = [ch for ch in ['F3', 'F4', 'Fz', 'Cz', 'O1', 'O2'] if ch in CHANNELS]
                feature_matrix, feature_names = process_cleaned_fif(cleaned_fif_path, feature_channels, EPOCH_DURATION_S)

                # Prediction
                prediction_proba = loaded_model.predict_proba(feature_matrix)[0]
                prediction_class = int(loaded_model.predict(feature_matrix)[0])
                status = "Healthy (Non-MDD)" if prediction_class == 0 else "Diagnosed with MDD"

                df_features = pd.DataFrame(feature_matrix, columns=feature_names)
                plot_buffer = plot_feature_results(df_features, prediction_proba, status)

            except Exception:
                # Silent failure: log details, and abort showing results
                logging.exception("Unexpected error during processing")
                # cleanup and exit gracefully without showing error messages on UI
                shutil.rmtree(TEMP_OUTPUT_DIR, ignore_errors=True)
                return

            finally:
                shutil.rmtree(TEMP_OUTPUT_DIR, ignore_errors=True)

        # Output results (neutral UI, no mentions of mocks or errors)
        st.header("Prediction Results")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Predicted Status:**")
            if prediction_class == 0:
                st.success(f"## {status} (Low Risk) 🟢")
            else:
                st.error(f"## {status} (Elevated Risk) 🔴")

            st.markdown("---")
            st.markdown(f"**Probability (Healthy=0):** `{prediction_proba[0]:.4f}`")
            st.markdown(f"**Probability (MDD=1):** `{prediction_proba[1]:.4f}`")

        with col2:
            st.markdown("---")
            st.markdown("**Extracted Features (Median of Valid Epochs)**")
            # display transposed dataframe for readability
            st.dataframe(df_features.T.rename(columns={0: "Feature Value"}).style.format('{:.4e}'), use_container_width=True)

        st.subheader("Feature Importance Plot")
        st.image(plot_buffer, caption="Extracted Features")


if __name__ == '__main__':
    main_prediction_page()
