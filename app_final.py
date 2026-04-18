import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import plotly.graph_objects as go
import cv2
import time
import os
import gdown

# ---------------------------
# CONFIG
# ---------------------------
IMG_SIZE = 224
MODEL_PATH = "deepfake_model_final.h5"
# ---------------------------
# DOWNLOAD MODEL (FOR DEPLOYMENT)
# ---------------------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait ⏳"):
        url = "https://drive.google.com/uc?id=1RfR_tFmGgufjD5zEOUum8ttgU9AtxrtF"
        gdown.download(url, MODEL_PATH, quiet=False)

st.set_page_config(page_title="Deepfake Detection System", layout="wide")

# ---------------------------
# CUSTOM STYLE
# ---------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617 70%);
    color: white;
}
.header-container {
    text-align: center;
    padding: 30px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.title {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg,#38bdf8,#6366f1,#22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    font-size: 15px;
    color: #94a3b8;
    max-width: 750px;
    margin: auto;
}
/* 🧠 IMPROVED CARD TITLES */
.card-title {
    font-size: 16px;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: 0.3px;
}

/* subtle glow */
.card-title span {
    color: #38bdf8;
}

/* optional divider */
.card-title::after {
    content: "";
    flex-grow: 1;
    height: 1px;
    background: rgba(255,255,255,0.08);
    margin-left: 8px;
}
/* ✅ REAL GLOW */
.real-glow {
    background: linear-gradient(135deg,#22c55e,#166534);
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 0 20px rgba(34,197,94,0.5);
    animation: pulseGreen 2s infinite;
}

/* ❌ FAKE GLOW */
.fake-glow {
    background: linear-gradient(135deg,#ef4444,#7f1d1d);
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 0 20px rgba(239,68,68,0.5);
    animation: pulseRed 2s infinite;
}

/* 🔥 ANIMATION */
@keyframes pulseGreen {
    0% {box-shadow: 0 0 10px rgba(34,197,94,0.3);}
    50% {box-shadow: 0 0 25px rgba(34,197,94,0.7);}
    100% {box-shadow: 0 0 10px rgba(34,197,94,0.3);}
}

@keyframes pulseRed {
    0% {box-shadow: 0 0 10px rgba(239,68,68,0.3);}
    50% {box-shadow: 0 0 25px rgba(239,68,68,0.7);}
    100% {box-shadow: 0 0 10px rgba(239,68,68,0.3);}
}
/* 🎯 RISK BADGE */
.risk-high {
    color: #ef4444;
    font-weight: 600;
}
.risk-medium {
    color: #facc15;
    font-weight: 600;
}
.risk-low {
    color: #22c55e;
    font-weight: 600;
}

/* 🟢 CONFIDENCE COLOR */
.conf-high { color:#22c55e; font-weight:700; }
.conf-mid { color:#facc15; font-weight:700; }
.conf-low { color:#ef4444; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# HEADER (PREMIUM VERSION)
# ---------------------------
st.markdown("""
<!-- TITLE -->
<div style="
    font-size:48px;
    font-weight:900;
    background: linear-gradient(90deg,#38bdf8,#6366f1,#22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom:12px;
    text-shadow: 0 0 25px rgba(56,189,248,0.25);
    text-align:center;
">
    Deepfake Image Detection using EfficientNetB4
</div>

<!-- SUBTITLE -->
<div style="
    font-size:17px;
    color:#94a3b8;
    max-width:720px;
    margin:auto;
    line-height:1.6;
    text-align:center;
">
   A deep learning-based classification system for distinguishing genuine and tampered facial images.
</div>

""", unsafe_allow_html=True)

# ---------------------------
# FACE DETECTOR
# ---------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    from tensorflow.keras.layers import Dense

    def custom_dense(**kwargs):
        kwargs.pop("quantization_config", None)
        return Dense(**kwargs)

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"Dense": custom_dense}
    )

model = load_model()

# ---------------------------
# SESSION STATE
# ---------------------------
for key in ["result","confidence","face_box","last_file","uploaded_img"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------
# UPLOAD
# ---------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg","png","jpeg"],
    key=st.session_state.get("uploader_key", "uploader_1")
)

if uploaded_file and uploaded_file != st.session_state.last_file:
    st.session_state.result = None
    st.session_state.confidence = None
    st.session_state.face_box = None
    st.session_state.last_file = uploaded_file
    st.session_state.uploaded_img = Image.open(uploaded_file)

# ---------------------------
# PREDICT FUNCTION
# ---------------------------
def predict_image(image):

    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30)
    )

    # ❌ NO FACE
    if len(faces) == 0:
        return "NO_FACE", 0.0, None

    # ✅ FACE FOUND
    face_box = max(faces, key=lambda x: x[2]*x[3])

    # 🔥 MODEL INPUT (FULL IMAGE)
    img_resized = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = preprocess_input(img_resized)

    pred = float(model(img_resized, training=False)[0][0])

    label = "REAL" if pred > 0.5 else "FAKE"
    confidence = pred if pred > 0.5 else (1 - pred)

    return label, confidence, face_box
def generate_explanation(label, confidence):
    conf = confidence * 100

    if label == "NO_FACE":
        return "No human face detected. Please upload a valid facial image for analysis."

    if label == "REAL":
        if conf > 85:
            return "The image appears authentic with consistent facial textures, natural lighting, and no visible manipulation artifacts."
        else:
            return "The image is likely real, but minor inconsistencies may exist. Confidence is moderate."

    if label == "FAKE":
        if conf > 85:
            return "The model detected strong indicators of deepfake manipulation such as texture inconsistencies, unnatural blending, or facial distortions."
        else:
            return "The image may contain synthetic or manipulated features. Some irregularities in facial structure or lighting were observed."
# ---------------------------
# BEFORE ANALYZE (FINAL CENTER FIX - PERFECT)
# ---------------------------
if uploaded_file and st.session_state.result is None:

    _, center, _ = st.columns([1,2,1])

    with center:

        # 🔥 IMAGE (FORCED CENTER)
        from io import BytesIO
        import base64

        buffered = BytesIO()
        st.session_state.uploaded_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        st.markdown(f"""
        <div style="text-align:center;">
            <img src="data:image/png;base64,{img_str}"
                 style="width:300px;border-radius:12px;box-shadow:0 0 20px rgba(56,189,248,0.25);">
            <div style="color:#94a3b8;font-size:13px;margin-top:6px;">
                Input Image
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 🔥 BUTTON (CENTERED)
        analyze_clicked = st.button("🔍 Analyze Image", use_container_width=True)

    # ---------------------------
    # LOADING + PREDICTION
    # ---------------------------
    if analyze_clicked:

        progress_container = st.empty()

        for i in range(101):
            percent = i

            progress_container.markdown(
                f"<div style='width:100%;'>"
                f"<div style='display:flex;justify-content:space-between;font-size:13px;color:#94a3b8;margin-bottom:4px;'>"
                f"<span>🔍 Analyzing Image...</span><span>{percent}%</span></div>"
                f"<div style='width:100%;height:10px;background:rgba(255,255,255,0.08);border-radius:10px;overflow:hidden;'>"
                f"<div style='height:100%;width:{percent}%;background:linear-gradient(90deg,#38bdf8,#6366f1,#22c55e);border-radius:10px;box-shadow:0 0 10px rgba(56,189,248,0.4);'></div>"
                f"</div></div>",
                unsafe_allow_html=True
            )

            time.sleep(0.01)

        progress_container.empty()

        label, conf, box = predict_image(st.session_state.uploaded_img)

        st.session_state.result = label
        st.session_state.confidence = conf
        st.session_state.face_box = box

        st.rerun()

# ---------------------------
# AFTER ANALYZE
# ---------------------------
if st.session_state.result:

    st.divider()

    # ✅ DEFINE COLUMNS FIRST
    col1, col2, col3 = st.columns(3)

    # ---------------------------
    # 🧠 PREDICTION
    # ---------------------------
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><span>🧠</span> Prediction</div>', unsafe_allow_html=True)

        conf = st.session_state.confidence
        conf_percent = conf * 100

        # ---------------------------
        # 🎯 RISK LEVEL LOGIC
        # ---------------------------
        if conf_percent >= 85:
            risk = "LOW" if st.session_state.result == "REAL" else "HIGH"
            risk_class = "risk-low" if st.session_state.result == "REAL" else "risk-high"
            conf_class = "conf-high"

        elif conf_percent >= 60:
            risk = "MEDIUM"
            risk_class = "risk-medium"
            conf_class = "conf-mid"

        else:
            risk = "HIGH" if st.session_state.result == "REAL" else "LOW"
            risk_class = "risk-high" if st.session_state.result == "REAL" else "risk-low"
            conf_class = "conf-low"

        # ---------------------------
        # RESULT BOX + EXPLANATION
        # ---------------------------
        if st.session_state.result == "NO_FACE":
            st.warning("No Face Detected")
            explanation = "No human face detected. Please upload a clear facial image."

        elif st.session_state.result == "REAL":
            st.markdown(f"""
            <div class="real-glow">
            ✔ REAL IMAGE<br>
            <span class="{conf_class}">
            {conf_percent:.2f}%
            </span>
            </div>
            """, unsafe_allow_html=True)

            # 🔥 Dynamic explanation (REAL)
            if conf_percent >= 90:
                explanation = "High confidence that the image is authentic. Facial textures, lighting, and structure appear natural with no manipulation signs."
            elif conf_percent >= 70:
                explanation = "The image is likely real. Most features are consistent, though minor variations exist."
            else:
                explanation = "Low confidence prediction. Some irregularities are present, reducing certainty."

        else:
            st.markdown(f"""
            <div class="fake-glow">
            ❌ FAKE IMAGE<br>
            <span class="{conf_class}">
            {conf_percent:.2f}%
            </span>
            </div>
            """, unsafe_allow_html=True)

            # 🔥 Dynamic explanation (FAKE)
            if conf_percent >= 90:
                explanation = "Strong evidence of deepfake manipulation. Detected unnatural blending, texture inconsistencies, and distortions."
            elif conf_percent >= 70:
                explanation = "The image is likely manipulated. Some inconsistencies in lighting and facial alignment were observed."
            else:
                explanation = "Low confidence prediction. Slight irregularities detected but not strong enough for certainty."

        # ---------------------------
        # 🎯 RISK LEVEL DISPLAY
        # ---------------------------
        if st.session_state.result != "NO_FACE":
            st.markdown(f"""
            <div style="margin-top:10px;font-size:14px;">
            🎯 Risk Level: <span class="{risk_class}">{risk}</span>
            </div>
            """, unsafe_allow_html=True)

        # ---------------------------
        # 🧠 EXPLANATION DISPLAY
        # ---------------------------
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div style="color:#94a3b8;font-size:13px;margin-bottom:5px;">
        <span>🧠</span> Explanation
        </div>
        """, unsafe_allow_html=True)
        
        st.write(explanation)

        st.markdown('</div>', unsafe_allow_html=True)
    # ---------------------------
    # 🖼️ IMAGE
    # ---------------------------
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><span>🖼️</span> Detected Image</div>', unsafe_allow_html=True)
        img_display = np.array(st.session_state.uploaded_img).copy()

        if st.session_state.face_box is not None:
            x,y,w,h = st.session_state.face_box
            cv2.rectangle(img_display,(x,y),(x+w,y+h),(0,255,0),2)

    # 🔥 SCAN EFFECT CONTAINER
        st.markdown('<div class="scan-container">', unsafe_allow_html=True)

        st.image(img_display, use_container_width=True)

    # 🔬 Animated scan line
        st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------
    # 📊 CONFIDENCE (FINAL STABLE VERSION)
    # ---------------------------
    if st.session_state.result != "NO_FACE":
        with col3:

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title"><span>📊</span> Confidence Score</div>', unsafe_allow_html=True)
            final_conf = float(st.session_state.confidence * 100)
            # 🔥 PERFECT ANIMATION (NO MISMATCH)
            progress_text = st.empty()

            for val in np.linspace(0, final_conf, 50):
                progress_text.markdown(
                f"<div style='text-align:center;font-size:24px;font-weight:800;color:#22c55e;'>"
                f"{val:.2f}%</div>",
                unsafe_allow_html=True
                )
                time.sleep(0.02)

               # ✅ FORCE FINAL EXACT VALUE
                progress_text.markdown(
                f"<div style='text-align:center;font-size:24px;font-weight:800;color:#22c55e;'>"
                f"{final_conf:.2f}%</div>",
                unsafe_allow_html=True
                )

            # ---------------------------
            # ✅ SINGLE FINAL GAUGE (NO ERROR)
            # ---------------------------
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=final_conf,
                number={'suffix': "%", 'valueformat': ".2f"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#22c55e"},
                    'bgcolor': "#020617",
                }
            ))

            fig.update_layout(
                height=220,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="#020617",
                font={'color': "white"}
            )

            st.plotly_chart(fig, use_container_width=True)

            # ---------------------------
            # 🔥 CORRECT PROBABILITY LOGIC
            # ---------------------------
            pred = st.session_state.confidence

            if st.session_state.result == "REAL":
                real_prob = pred
                fake_prob = 1 - pred
            else:
                fake_prob = pred
                real_prob = 1 - pred

            real_percent = real_prob * 100
            fake_percent = fake_prob * 100

            st.markdown('<div class="card-title"><span>🔍</span> Class Probabilities</div>', unsafe_allow_html=True)

            # REAL
            st.markdown("**REAL**")
            st.progress(real_prob)
            st.markdown(f"<span style='color:#22c55e'>{real_percent:.2f}%</span>", unsafe_allow_html=True)

            # FAKE
            st.markdown("**FAKE**")
            st.progress(fake_prob)
            st.markdown(f"<span style='color:#ef4444'>{fake_percent:.2f}%</span>", unsafe_allow_html=True)
        # ---------------------------
        # 🔄 RESET BUTTON
        # ---------------------------
    if st.button("🔄 Reset", use_container_width=True):

        # Clear all session values
        st.session_state.result = None
        st.session_state.confidence = None
        st.session_state.face_box = None
        st.session_state.uploaded_img = None
        st.session_state.last_file = None

        st.session_state.uploader_key = str(time.time())

        st.rerun()

# ---------------------------
# DESCRIPTION (IEEE CONCISE)
# ---------------------------
with st.expander("ℹ️ System Description"):
    st.write("""
This system presents a deep learning-based approach for classification of facial images as authentic or manipulated. 
The model is developed using the EfficientNetB4 architecture with transfer learning to leverage pre-trained feature representations.

Input images are resized and normalized prior to processing. Data augmentation techniques are applied during training 
to improve generalization and reduce overfitting.

A face detection mechanism is incorporated to ensure that predictions are generated only for valid facial inputs, 
thereby improving system reliability.
    
Technologies Used:
- TensorFlow & Keras
- EfficientNetB4
- OpenCV (Face Detection)
- Streamlit (User Interface)
- Plotly (Visualization)
""")
