
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import pandas as pd
import time

st.set_page_config(
    page_title="Cricket Player Recognition",
    layout="centered",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"

# ---------------- Custom Styling ----------------
st.markdown("""
    <style>
    .stApp {
        background-color: #0B1220;
        color: #E6EDF7;
    }
    section[data-testid="stSidebar"] {
        background-color: #0F1A2E;
    }
    h1, h2, h3 {
        color: #F5F7FA;
    }
    .title-text {
        font-size: 2.2rem;
        font-weight: 700;
        color: #F5F7FA;
        margin-bottom: 0px;
    }
    .subtitle-text {
        color: #8FA3C4;
        font-size: 1rem;
        margin-top: 0px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1B2A4A, #14203B);
        border: 1px solid #2E4374;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin-top: 20px;
    }
    .predicted-name {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4FA8FF;
        margin: 6px 0px;
    }
    .confidence-text {
        font-size: 1rem;
        color: #8FA3C4;
    }
    div[data-testid="stFileUploader"] {
        background-color: #101B33;
        border: 1px dashed #2E4374;
        border-radius: 12px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #2E63FF;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5em 1.5em;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1E4FE0;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))

model = load_model()
class_names = sorted(model.names.values())

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### About")
    st.write(
        "This app identifies Indian cricketers from a photo using a "
        "YOLO11 image classification model, trained on face-cropped "
        "images for higher accuracy."
    )
    st.markdown("---")
    st.markdown("### Players in this model")
    for name in class_names:
        st.markdown(f"- {name}")
    st.markdown("---")
    st.caption("Model: YOLO11n-cls  |  Runs on CPU")

# ---------------- Header ----------------
st.markdown('<p class="title-text">Cricket Player Recognition</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Upload a photo and the model will identify the player.</p>', unsafe_allow_html=True)
st.write("")

# ---------------- Upload ----------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.image(image, caption="Uploaded Image", width="stretch")

    with st.spinner("Analyzing image..."):
        start = time.time()
        result = model.predict(source=image, device="cpu", imgsz=224, verbose=False)[0]
        elapsed = (time.time() - start) * 1000

    probs = result.probs.data.cpu().numpy()
    names = result.names
    top5_idx = probs.argsort()[::-1][:5]

    top_name = names[top5_idx[0]]
    top_conf = probs[top5_idx[0]] * 100

    with col2:
        st.markdown(f"""
            <div class="prediction-card">
                <div class="confidence-text">Predicted Player</div>
                <div class="predicted-name">{top_name}</div>
                <div class="confidence-text">Confidence: {top_conf:.1f}%</div>
                <div class="confidence-text">Inference time: {elapsed:.0f} ms</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("#### Top 5 Predictions")
    df_top5 = pd.DataFrame({
        "Player": [names[i] for i in top5_idx],
        "Confidence (%)": [round(probs[i] * 100, 1) for i in top5_idx]
    }).set_index("Player")

    st.bar_chart(df_top5, color="#4FA8FF")
    st.dataframe(df_top5, use_container_width=True)

else:
    st.info("Upload a cricketer's photo to get started.")
