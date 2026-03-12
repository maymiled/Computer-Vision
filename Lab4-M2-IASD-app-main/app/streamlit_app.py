"""
app/streamlit_app.py
──────────────────────────────────────────────────────────────────────────────
Streamlit frontend for the Pneumonia Detection project.

Pages
─────
  🏠 Home         — project description and instructions
  🧠 Train        — pick model + hyperparameters, launch training, view curves
  📊 Results      — detailed metrics and plots from the last training run
  📄 Submit       — generate predictions CSV and download it

Make sure the FastAPI backend is running before using Train or Submit:
    uvicorn main:app --reload --port 8000   (from the api/ folder)
"""

import io
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Config ─────────────────────────────────────────────────────────────────────

API_URL    = "http://localhost:8000"
MODEL_LIST = ["U-Net", "ResNet", "Inception"]

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ──────────────────────────────────────────────────────────────

if "last_results"    not in st.session_state: st.session_state.last_results    = None
if "last_model_name" not in st.session_state: st.session_state.last_model_name = None
if "predictions_df"  not in st.session_state: st.session_state.predictions_df  = None


# ── Sidebar ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🫁 Pneumonia Detection")
    st.caption("Computer Vision Project")
    st.divider()
    page = st.radio("Navigate", ["🏠 Home", "🧠 Train", "📊 Results", "📄 Submit"])
    st.divider()

    # ── Model selector ──────────────────────────────────────────────────────
    st.subheader("Model")
    selected_model = st.selectbox("Architecture", MODEL_LIST, label_visibility="collapsed")

    # ── Hyperparameters ─────────────────────────────────────────────────────
    st.subheader("Hyperparameters")

    learning_rate = st.select_slider(
        "Learning rate",
        options=[1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        value=3e-4,
        format_func=lambda x: f"{x:.0e}",
    )

    epochs = st.slider("Epochs", min_value=1, max_value=50, value=30)

    batch_size = st.select_slider(
        "Batch size",
        options=[8, 16, 32, 64],
        value=32,
    )

    dropout_rate = st.slider("Dropout", min_value=0.0, max_value=0.8, value=0.3, step=0.1)

    image_size = st.select_slider(
        "Image size",
        options=[64, 128, 224, 256],
        value=224,
    )


# ── Helpers ─────────────────────────────────────────────────────────────────────

def call_train(payload: dict) -> dict:
    """POST /train and return the JSON response or raise on error."""
    resp = requests.post(f"{API_URL}/train", json=payload, timeout=3600)
    resp.raise_for_status()
    return resp.json()


def call_predict(payload: dict) -> dict:
    """POST /predict and return the JSON response or raise on error."""
    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


def plot_curves(results: dict, model_name: str) -> plt.Figure:
    """
    Build a 1×3 matplotlib figure with:
      - Loss curves   (train vs val)
      - Accuracy curves (train vs val)
      - AUC curves    (train vs val)

    Return the Figure object so Streamlit can display it.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs_range = range(1, len(results["train_losses"]) + 1)

    # Loss
    axes[0].plot(epochs_range, results["train_losses"], color="blue", linestyle="-", label="train")
    axes[0].plot(epochs_range, results["val_losses"], color="orange", linestyle="--", label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True)
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs_range, results["train_accs"], color="blue", linestyle="-", label="train")
    axes[1].plot(epochs_range, results["val_accs"], color="orange", linestyle="--", label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True)
    axes[1].legend()

    # AUC
    axes[2].plot(epochs_range, results["train_aucs"], color="blue", linestyle="-", label="train")
    axes[2].plot(epochs_range, results["val_aucs"], color="orange", linestyle="--", label="val")
    axes[2].axhline(0.95, color="red", linestyle="--", label="target")
    axes[2].set_title("AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True)
    axes[2].legend()

    fig.suptitle(f"{model_name} — Training curves", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(results: dict) -> plt.Figure:
    """
    Optional: build a confusion matrix figure from the last validation pass.
    You may need to add TP/FP/TN/FN counts to the /train response for this.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    # build confusion matrix
    return fig


# ── Pages ───────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
# ──────────────────────────────────────────────────────────────────────────────

    st.title("🫁 Pneumonia Detection from Chest X-rays")

    st.markdown("""
    Welcome! In this project you will:

    1. **Implement** three deep learning architectures in `models/`
    2. **Train** them from this interface by choosing hyperparameters
    3. **Analyse** training and validation curves
    4. **Generate** a prediction CSV on the test set
    5. **Submit** the CSV to the class leaderboard

    ---

    ### How to get started

    - Make sure the **FastAPI backend** is running:
      ```
      cd api && uvicorn main:app --reload --port 8000
      ```
    - Complete the model skeletons in `models/unet.py`, `models/resnet.py`, `models/inception.py`
    - Complete the data transforms in `api/main.py → get_transforms()`
    - Head to the **🧠 Train** page and launch your first run

    ---

    ### Data

    | Split | Folder | Classes |
    |---|---|---|
    | Train | `data/train/` | PNEUMONIA / NORMAL |
    | Validation | `data/val/` | PNEUMONIA / NORMAL |
    | Test | `data/test_for_students/` | unlabelled |

    ### Submission format

    ```
    id,prediction
    img_0001,0.91
    img_0002,0.07
    ```

    Predictions must be **probabilities between 0 and 1**, not hard labels.
    """)


# ──────────────────────────────────────────────────────────────────────────────
elif page == "🧠 Train":
# ──────────────────────────────────────────────────────────────────────────────

    st.title(f"🧠 Train — {selected_model}")
    st.caption("Configure hyperparameters in the sidebar, then click Train.")

    col_info, col_btn = st.columns([3, 1])

    with col_info:
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Model | **{selected_model}** |
        | Learning rate | `{learning_rate:.0e}` |
        | Epochs | `{epochs}` |
        | Batch size | `{batch_size}` |
        | Dropout | `{dropout_rate}` |
        | Image size | `{image_size}×{image_size}` |
        """)

    with col_btn:
        train_btn = st.button("🚀 Train", type="primary", use_container_width=True)

    if train_btn:
        payload = {
            "model_name":    selected_model,
            "learning_rate": learning_rate,
            "epochs":        epochs,
            "batch_size":    batch_size,
            "dropout_rate":  dropout_rate,
            "image_size":    image_size,
        }

        with st.spinner(f"Training {selected_model} for {epochs} epochs… this may take a while."):
            try:
                results = call_train(payload)
                st.session_state.last_results    = results
                st.session_state.last_model_name = selected_model
                st.success(f"Training complete! Best val AUC: **{results['best_val_auc']:.4f}**")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Is `uvicorn main:app --reload --port 8000` running?")
            except Exception as e:
                st.error(f"Training failed: {e}")

    if st.session_state.last_results and st.session_state.last_model_name == selected_model:
        st.divider()
        fig = plot_curves(st.session_state.last_results, selected_model)
        st.pyplot(fig)


# ──────────────────────────────────────────────────────────────────────────────
elif page == "📊 Results":
# ──────────────────────────────────────────────────────────────────────────────

    st.title("📊 Results")

    if st.session_state.last_results is None:
        st.info("No training run yet. Go to 🧠 Train first.")
    else:
        results    = st.session_state.last_results
        model_name = st.session_state.last_model_name

        st.subheader(f"Last run: {model_name}")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Val AUC",  f"{results['best_val_auc']:.4f}")
        col2.metric("Final Val Acc", f"{results['val_accs'][-1]:.4f}")
        col3.metric("Final Val Loss",f"{results['val_losses'][-1]:.4f}")
        col4.metric("Epochs",        len(results["train_losses"]))

        st.divider()

        # Curves
        fig = plot_curves(results, model_name)
        st.pyplot(fig)

        st.divider()

        # Per-epoch table
        st.subheader("Per-epoch details")
        df = pd.DataFrame({
            "Epoch":      range(1, len(results["train_losses"]) + 1),
            "Train Loss": results["train_losses"],
            "Val Loss":   results["val_losses"],
            "Train Acc":  results["train_accs"],
            "Val Acc":    results["val_accs"],
            "Train AUC":  results["train_aucs"],
            "Val AUC":    results["val_aucs"],
        })
        st.dataframe(df.style.format({c: "{:.4f}" for c in df.columns if c != "Epoch"}),
                     use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
elif page == "📄 Submit":
# ──────────────────────────────────────────────────────────────────────────────

    st.title("📄 Generate submission CSV")
    st.caption("This uses the best saved weights for the selected model.")

    st.info(f"Selected model: **{selected_model}** — make sure you trained it first.")

    if st.button("⚙️ Generate predictions", type="primary"):
        with st.spinner("Running inference on test set…"):
            try:
                response = call_predict({"model_name": selected_model, "image_size": image_size})
                df = pd.DataFrame(response["predictions"])
                st.session_state.predictions_df = df
                st.success(f"Generated {len(df)} predictions.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Is `uvicorn main:app --reload --port 8000` running?")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    if st.session_state.predictions_df is not None:
        df = st.session_state.predictions_df
        st.dataframe(df.head(20), use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Download submission.csv",
            data=csv_bytes,
            file_name="submission.csv",
            mime="text/csv",
            type="primary",
        )

        st.caption("Upload this file to the class leaderboard to get your ROC-AUC score.")
