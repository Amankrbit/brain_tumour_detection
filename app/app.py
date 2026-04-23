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
        if label == "No_tumor":
            label = "No Tumor"
            
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
    with st.expander("🔍 How did the AI make this decision? (Explainable AI)"):
        st.write("""
        This diagnostic tool uses **Grad-CAM** (Gradient-weighted Class Activation Mapping) to provide transparency into the neural network's decision-making process.
        
        Instead of acting as a 'black box', the AI generates a heatmap over the original MRI scan:
        * 🔴 **Red/Yellow Regions:** High importance. These are the specific biological textures and shapes the AI heavily relied on to make its diagnosis.
        * 🔵 **Blue Regions:** Low importance. The AI ignored these areas.
        
        **Clinical Value:** This Explainable AI feature allows radiologists to verify that the model is looking at the actual tumor site rather than image artifacts (like skull markers or text).
        """) 

  # --- 6. AI MEDICAL ASSISTANT CHATBOT (GROQ) ---
    st.markdown("---")
    st.markdown(f"### 💬 Ask the AI Assistant about {label}")

    # Initialize the Groq client securely
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
        # 1. Initialize states for infinite memory and continuous suggestions
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        if "suggestions" not in st.session_state:
            # Seed the VERY FIRST set of questions based on the MRI diagnosis
            if label == "Glioma":
                st.session_state.suggestions = ["What exactly is a Glioma?", "What are the standard treatment options?", "How fast do Gliomas typically grow?"]
            elif label == "Meningioma":
                st.session_state.suggestions = ["Are Meningiomas usually benign or malignant?", "Will a Meningioma always require brain surgery?", "What are the common symptoms?"]
            elif label == "Pituitary":
                st.session_state.suggestions = ["How does a Pituitary tumor affect hormones?", "What non-surgical treatments exist?", "Can this cause vision problems?"]
            else: 
                st.session_state.suggestions = ["If my MRI is clear, what else could be causing my headaches?", "Should I still follow up with a neurologist?", "How reliable is this AI at detecting early-stage tumors?"]

        # 2. Display previous chat history
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 3. Check if it's the AI's turn to respond
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            
            # --- PROMPT ENGINEERING MAGIC ---
            # We explicitly order the AI to generate follow-up questions wrapped in secret tags
            system_context = f"""
            You are a highly professional neuro-oncology AI assistant. 
            The patient's MRI scan was just analyzed by our DenseNet deep learning model and the diagnosis is: {label} with a confidence of {conf:.2f}%.
            Answer the user's questions accurately, empathetically, and concisely based on this specific diagnosis.
            
            CRITICAL INSTRUCTION: At the very end of your response, you MUST generate 3 highly relevant follow-up questions the user might want to ask next based on your answer. 
            Wrap these 3 questions in a <suggestions> tag and separate them with a pipeline character (|). 
            Example format exactly like this:
            <suggestions>Question 1?|Question 2?|Question 3?</suggestions>
            """
            
            api_messages = [{"role": "system", "content": system_context}] + st.session_state.messages

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    with st.spinner("Thinking..."):
                        response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=api_messages,
                            stream=False
                        )
                    
                    raw_response = response.choices[0].message.content
                    
                    # Parse the secret suggestions out of the AI's response using Python
                    if "<suggestions>" in raw_response and "</suggestions>" in raw_response:
                        display_text = raw_response.split("<suggestions>")[0].strip()
                        sugg_text = raw_response.split("<suggestions>")[1].split("</suggestions>")[0]
                        new_suggestions = [s.strip() for s in sugg_text.split("|") if s.strip()]
                    else:
                        display_text = raw_response
                        new_suggestions = [] # Fallback if AI forgets
                        
                    message_placeholder.markdown(display_text)
                    
                    # Update memory and trigger a UI refresh!
                    st.session_state.messages.append({"role": "assistant", "content": display_text})
                    st.session_state.suggestions = new_suggestions
                    st.rerun() 
                    
                except Exception as e:
                    st.error(f"Chatbot encountered an error: {e}")

        # 4. Display the clickable buttons and text input (Waiting for user)
        if not st.session_state.messages or st.session_state.messages[-1]["role"] == "assistant":
            
            # Render the continuous dynamic buttons
            if st.session_state.suggestions:
                st.markdown("**💡 Suggested Follow-up Questions:**")
                for i, sugg in enumerate(st.session_state.suggestions):
                    # We use unique keys to prevent Streamlit button errors
                    if st.button(sugg, key=f"btn_{len(st.session_state.messages)}_{i}"):
                        # Act as if the user typed this question
                        st.session_state.messages.append({"role": "user", "content": sugg})
                        st.rerun()
            
            # Allow the user to ignore the buttons and type manually
            if prompt := st.chat_input("Or type your own question here..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
                    st.error(f"Chatbot encountered an error: {e}")
