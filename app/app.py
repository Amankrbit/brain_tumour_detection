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
   # --- 7. EXPLAINABLE AI SECTION ---
    with st.expander("🔍 How did the AI make this decision? (Explainable AI)"):
        st.write("""
        This diagnostic tool uses **Grad-CAM** (Gradient-weighted Class Activation Mapping) to provide transparency into the neural network's decision-making process.
        
        Instead of acting as a 'black box', the AI generates a heatmap over the original MRI scan:
        * 🔴 **Red/Yellow Regions:** High importance. These are the specific biological textures and shapes the AI heavily relied on to make its diagnosis.
        * 🔵 **Blue Regions:** Low importance. The AI ignored these areas.
        
        **Clinical Value:** This Explainable AI feature allows radiologists to verify that the model is looking at the actual tumor site rather than image artifacts.
        """) 

    # --- 8. AI MEDICAL ASSISTANT CHATBOT (GROQ) ---
    st.markdown("---")
    st.markdown("### 💬 AI Clinical Assistant")

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

# [Keep everything below this line from your previous code: The rest of the Expander and the Groq Chatbot section]
