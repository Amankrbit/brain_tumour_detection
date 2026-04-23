import sys
import os
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image

# --- 1. SYSTEM PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Now import your custom modules
from src.model import load_trained_model
from src.data_loader import preprocess_image_for_inference
from src.metrics import make_gradcam_heatmap, generate_gradcam_overlay

# --- 2. FAIL-PROOF MODEL DOWNLOADER ---
@st.cache_resource
def get_model():
    # REPLACE THIS with the link you copied from GitHub Releases in Step 1!
    model_url = "https://github.com/Amankrbit/brain_tumour_detection/releases/download/v1.0/advanced_densenet.keras"
    
    with st.spinner("Downloading AI Model... This only happens on the first run."):
        # tf.keras.utils.get_file handles the /tmp folder and downloading safely!
        model_path = tf.keras.utils.get_file(
            "advanced_densenet.keras",
            origin=model_url
        )
        
    return load_trained_model(model_path)

# --- 3. APP UI SETUP ---
st.set_page_config(page_title="Brain Tumor AI", layout="wide")
st.title("🧠 Brain Tumor Diagnostic Assistant with Explainable AI")

st.markdown("""
Upload a Brain MRI scan. The AI will classify the tumor type and generate a **Grad-CAM heatmap** to highlight the specific region of the brain that influenced its decision.
""")

# Load the model
try:
    model = get_model()
    CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- 4. IMAGE PROCESSING ---
uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    st.markdown("### Analysis Results")
    col1, col2, col3 = st.columns(3)
    
    with st.spinner("Analyzing scan..."):
        # Preprocess
        img_array, processed_img = preprocess_image_for_inference(img)
        
        # Inference & Heatmap
        heatmap, predictions = make_gradcam_heatmap(img_array, model)
        
        # Get results
        idx = np.argmax(predictions)
        label = CLASS_NAMES[idx].capitalize()
        conf = predictions[idx] * 100
        
        # Generate Grad-CAM Overlay
        overlay = generate_gradcam_overlay(processed_img, heatmap)

    # Display Results
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original MRI", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Preprocessed (Cropped)", use_container_width=True)
    with col3:
        st.image(overlay, caption="Grad-CAM Explainability", use_container_width=True)

    st.success(f"**Diagnosis:** {label} | **Confidence:** {conf:.2f}%")

    # --- 5. EXPLAINABLE AI SECTION ---
    st.markdown("---")
    with st.expander("🔍 How did the AI make this decision? (Explainable AI)"):
        st.write("""
        This diagnostic tool uses **Grad-CAM** (Gradient-weighted Class Activation Mapping) to provide transparency into the neural network's decision-making process.
        
        Instead of acting as a 'black box', the AI generates a heatmap over the original MRI scan:
        * 🔴 **Red/Yellow Regions:** High importance. These are the specific biological textures and shapes the AI heavily relied on to make its diagnosis.
        * 🔵 **Blue Regions:** Low importance. The AI ignored these areas.
        
        **Clinical Value:** This Explainable AI feature allows radiologists to verify that the model is looking at the actual tumor site rather than image artifacts (like skull markers or text).
        """)
