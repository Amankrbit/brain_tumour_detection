# [Keep everything above this line from your previous code: Imports, get_model, draw_tumor_bounding_box, CSS, Disclaimer, etc.]

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
    # Only show the uploader if the form is complete
    uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        st.markdown("### Analysis Results")
        
        with st.spinner("Analyzing scan..."):
            # Preprocess
            img_array, processed_img = preprocess_image_for_inference(img)
            
            # Inference & Heatmap
            heatmap, predictions = make_gradcam_heatmap(img_array, model)
            
            # Get results
            idx = np.argmax(predictions)
            label = CLASS_NAMES[idx].capitalize()
            
            # Assigning the region based on the label for the report
            if label == "No_tumor":
                label = "No Tumor"
                region_text = "N/A (Clear Scan)"
            elif label == "Pituitary":
                region_text = "Pituitary Gland"
            else:
                region_text = "Cerebral Tissue"
                
            conf = predictions[idx] * 100
            
            # Generate Overlays
            overlay = generate_gradcam_overlay(processed_img, heatmap)
            bbox_image = draw_tumor_bounding_box(processed_img, heatmap, threshold=160)

        # --- GENERATE THE CLINICAL REPORT SUMMARY ---
        st.markdown("---")
        st.markdown("### 📄 Clinical AI Screening Report")
        
        # Displaying the exact format you requested in a styled box
        st.success(f"""
        **Name:** {patient_name}  
        **Age:** {int(patient_age)}  
        **Gender:** {patient_gender}  
        **AI Diagnostic Region of Brain:** {region_text}  
        **Suspected AI Diagnostic:** {label}  
        **Model Confidence:** {conf:.2f}%
        """)

        # Display Results in 4 Columns
        st.markdown("#### Visual Localization")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original MRI", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Preprocessed", use_container_width=True)
        with col3:
            st.image(overlay, caption="Grad-CAM Heatmap", use_container_width=True)
        with col4:
            st.image(cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB), caption="AI Bounding Box", use_container_width=True)

    # --- 7. EXPLAINABLE AI SECTION ---
    with st.expander("🔍 How did the AI make this decision? (Explainable AI)"):

# [Keep everything below this line from your previous code: The rest of the Expander and the Groq Chatbot section]
