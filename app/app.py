import sys
import os
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image
from groq import Groq  

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
    model_url = "https://github.com/Amankrbit/brain_tumour_detection/releases/download/v1.0/advanced_densenet.keras"
    
    with st.spinner("Downloading AI Model... This only happens on the first run."):
        model_path = tf.keras.utils.get_file(
            "advanced_densenet.keras",
            origin=model_url
        )
        
    return load_trained_model(model_path)

# --- 3. BOUNDING BOX GENERATOR ---
def draw_tumor_bounding_box(image, heatmap, threshold=160):
    """
    Finds the hottest region of the Grad-CAM heatmap and draws a bounding box around it.
    """
    # Resize the raw heatmap to perfectly match the original image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Ensure the heatmap is in the 0-255 range
    if heatmap_resized.dtype != np.uint8:
        heatmap_resized = np.uint8(255 * heatmap_resized)
        
    # Threshold the heatmap: Keep only the hottest areas
    _, thresh = cv2.threshold(heatmap_resized, threshold, 255, cv2.THRESH_BINARY)
    
    # Find the contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_with_box = np.copy(image)
    
    if contours:
        # Find the largest hot spot area to ignore random noise
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Draw a sleek Cyan bounding box
        cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(img_with_box, "AI Detected Region", (x, y - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
    return img_with_box

# --- 4. APP UI & PRO CSS SETUP ---
st.set_page_config(page_title="Brain Tumor AI", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
        color: #e2e8f0;
    }
    header {visibility: hidden;}
    h1 {
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -1px;
        padding-bottom: 10px;
    }
    h2, h3 { color: #f8fafc !important; font-weight: 600 !important; }
    [data-testid="stImage"] {
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        overflow: hidden;
        transition: transform 0.3s ease;
    }
    [data-testid="stImage"]:hover {
        transform: scale(1.02);
    }
    [data-testid="stChatMessage"] {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        padding: 15px !important;
        backdrop-filter: blur(10px) !important;
        margin-bottom: 10px !important;
    }
    [data-testid="stChatMessage"] * {
        color: #ffffff !important; 
    }
    [data-testid="stBottom"] > div {
        background: transparent !important; 
    }
    [data-testid="stChatInput"] {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 14px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.6) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #ffffff !important; 
        caret-color: #38bdf8 !important; 
    }
    [data-testid="stChatInputSubmit"] svg {
        fill: #38bdf8 !important; 
    }
    div[data-testid="stButton"] > button {
        background: rgba(15, 23, 42, 0.6) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        backdrop-filter: blur(5px) !important;
    }
    div[data-testid="stButton"] > button p {
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
        color: #38bdf8 !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: rgba(56, 189, 248, 0.15) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.5) !important;
        transform: translateY(-3px) !important;
    }
    div[data-testid="stButton"] > button:hover p {
        color: #ffffff !important; 
    }
    .stAlert {
        border-radius: 12px;
        backdrop-filter: blur(5px);
    }
    div[data-testid="stAlert"]:has(.st-emotion-cache-1n7c1ee) {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        color: #d1fae5;
    }
    div[data-testid="stAlert"]:has(.st-emotion-cache-121r7m3) {
        background-color: rgba(56, 189, 248, 0.1);
        border-left: 4px solid #38bdf8;
        color: #e0f2fe;
    }
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 12px;
        color: #818cf8 !important;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.3);
    }
    strong {
        color: #38bdf8;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Brain Tumor AI Diagnostics")

# --- STRICT CLINICAL DISCLAIMER ---
st.title("🧠 Brain Tumor AI Diagnostics")

# --- STRICT CLINICAL DISCLAIMER ---
st.warning("""
**STRICT CLINICAL DISCLAIMER:** This application is an experimental AI tool designed for educational and preliminary screening purposes only. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. The AI's classifications and heatmaps are mathematical predictions, not medical facts. Always consult with a certified neurologist or oncologist regarding any medical conditions or before making any healthcare decisions. Do not ignore professional medical advice because of something you have read or seen on this application.
""")

st.markdown("Upload a Brain MRI scan. The AI will classify the tumor type and generate a **Grad-CAM heatmap** and **Bounding Box** to highlight the specific region of the brain that influenced its decision.")

# Load the model
try:
    model = get_model()
    CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- 5. PATIENT INFORMATION FORM ---
st.markdown("### 📋 Patient Details")
col_name, col_age, col_gender = st.columns(3)

with col_name:
    patient_name = st.text_input("Patient Name", placeholder="e.g., John Doe")
with col_age:
    # Strictly limit age between 1 and 150
    patient_age = st.number_input("Age", min_value=1, max_value=150, value=None, step=1, placeholder="0-150")
with col_gender:
    patient_gender = st.selectbox("Gender", ["Select...", "Male", "Female", "Other"])

# --- 6. CONDITIONAL IMAGE PROCESSING ---
# Check if all fields are filled. If not, stop and wait.
if not patient_name or patient_age is None or patient_gender == "Select...":
    st.info("⚠️ Please fill out the Patient Name, Age, and Gender above to unlock the MRI upload tool.")
else:
    uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        with st.spinner("Analyzing scan..."):
            # Preprocess
            img_array, processed_img = preprocess_image_for_inference(img)
            
            # Inference & Heatmap
            heatmap, predictions = make_gradcam_heatmap(img_array, model)
            
            # Get results
            idx = np.argmax(predictions)
            label = CLASS_NAMES[idx].capitalize()
            if label == "No_tumor":
                label = "No Tumor"
                region_text = "N/A (Clear Scan)"
            elif label == "Pituitary":
                region_text = "Pituitary Gland (Localized via Grad-CAM)"
            else:
                region_text = "Cerebral Tissue (Localized via Grad-CAM)"
                
            conf = predictions[idx] * 100
            
            # Generate Overlays
            overlay = generate_gradcam_overlay(processed_img, heatmap)
            bbox_image = draw_tumor_bounding_box(processed_img, heatmap, threshold=160)

        # --- GENERATE THE CLINICAL REPORT SUMMARY ---
        st.markdown("---")
        st.markdown("### 📄 Clinical AI Screening Report")
        
        # Displaying the exact format you requested
        st.success(f"""
        **Name:** {patient_name}  
        **Age:** {int(patient_age)}  
        **Gender:** {patient_gender}  
        **AI Diagnostic Region of Brain:** {region_text}  
        **Suspected AI Diagnostic:** {label}  
        **Model Confidence:** {conf:.2f}%
        """)

        # Display Results in 4 Columns
        st.markdown("#### Visual Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original MRI", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Preprocessed", use_container_width=True)
        with col3:
            st.image(overlay, caption="Grad-CAM Heatmap", use_container_width=True)
        with col4:
            st.image(cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB), caption="AI Bounding Box", use_container_width=True)

    # --- 6. EXPLAINABLE AI SECTION ---
    with st.expander("🔍 How did the AI make this decision? (Explainable AI)"):
        st.write("""
        This diagnostic tool uses **Grad-CAM** (Gradient-weighted Class Activation Mapping) to provide transparency into the neural network's decision-making process.
        
        Instead of acting as a 'black box', the AI generates a heatmap over the original MRI scan:
        * 🔴 **Red/Yellow Regions:** High importance. These are the specific biological textures and shapes the AI heavily relied on to make its diagnosis.
        * 🔵 **Blue Regions:** Low importance. The AI ignored these areas.
        
        **Clinical Value:** This Explainable AI feature allows radiologists to verify that the model is looking at the actual tumor site rather than image artifacts.
        """) 

    # --- 7. AI MEDICAL ASSISTANT CHATBOT (GROQ) ---
    st.markdown("---")
    st.markdown(f"### 💬 AI Clinical Assistant")

    try:
        if "GROQ_API_KEY" not in st.secrets:
            st.error("🚨 Error: Streamlit cannot find 'GROQ_API_KEY' in your secrets.")
            client = None
        else:
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    except Exception as e:
        st.error(f"🚨 Groq Initialization Error: {e}")
        client = None

    if client:
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        if "suggestions" not in st.session_state:
            if label == "Glioma":
                st.session_state.suggestions = ["What exactly is a Glioma?", "What are the standard treatment options?", "How fast do Gliomas typically grow?"]
            elif label == "Meningioma":
                st.session_state.suggestions = ["Are Meningiomas usually benign or malignant?", "Will a Meningioma always require brain surgery?", "What are the common symptoms?"]
            elif label == "Pituitary":
                st.session_state.suggestions = ["How does a Pituitary tumor affect hormones?", "What non-surgical treatments exist?", "Can this cause vision problems?"]
            else: 
                st.session_state.suggestions = ["If my MRI is clear, what else could be causing my headaches?", "Should I still follow up with a neurologist?", "How reliable is this AI at detecting early-stage tumors?"]

        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            
            system_context = f"""
            You are a highly professional neuro-oncology AI assistant. 
            The patient's MRI scan was just analyzed by our DenseNet deep learning model and the diagnosis is: {label} with a confidence of {conf:.2f}%.
            Answer the user's questions accurately, empathetically, and concisely based on this specific diagnosis.
            Remind the user gently in your first response that you are an AI assistant and they should consult their doctor.
            
            CRITICAL INSTRUCTION: At the very end of your response, you MUST generate 3 highly relevant follow-up questions the user might want to ask next based on your answer. 
            Wrap these 3 questions in a <suggestions> tag and separate them with a pipeline character (|). 
            Example format exactly like this:
            <suggestions>Question 1?|Question 2?|Question 3?</suggestions>
            """
            
            api_messages = [{"role": "system", "content": system_context}] + st.session_state.messages

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    with st.spinner("Analyzing clinical data..."):
                        response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=api_messages,
                            stream=False
                        )
                    
                    raw_response = response.choices[0].message.content
                    
                    if "<suggestions>" in raw_response and "</suggestions>" in raw_response:
                        display_text = raw_response.split("<suggestions>")[0].strip()
                        sugg_text = raw_response.split("<suggestions>")[1].split("</suggestions>")[0]
                        new_suggestions = [s.strip() for s in sugg_text.split("|") if s.strip()]
                    else:
                        display_text = raw_response
                        new_suggestions = []
                        
                    message_placeholder.markdown(display_text)
                    
                    st.session_state.messages.append({"role": "assistant", "content": display_text})
                    st.session_state.suggestions = new_suggestions
                    st.rerun() 
                    
                except Exception as e:
                    st.error(f"Chatbot encountered an error: {e}")

        if not st.session_state.messages or st.session_state.messages[-1]["role"] == "assistant":
            
            if st.session_state.suggestions:
                st.markdown("**💡 Suggested Inquiries:**")
                for i, sugg in enumerate(st.session_state.suggestions):
                    if st.button(sugg, key=f"btn_{len(st.session_state.messages)}_{i}", use_container_width=True):
                        st.session_state.messages.append({"role": "user", "content": sugg})
                        st.rerun()
            
            if prompt := st.chat_input("Or type your own medical inquiry here..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
