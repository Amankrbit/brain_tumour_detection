🧠 Brain Tumor Diagnostic Assistant with Explainable AI (XAI)

📌 Project Overview
This repository contains a complete, end-to-end deep learning pipeline for multi-class brain tumor classification using MRI scans. Built as a Master's level research project, it categorizes scans into four classes: Glioma, Meningioma, Pituitary, and No Tumor.

Beyond basic classification, this project integrates Explainable AI (Grad-CAM) to generate diagnostic heatmaps, allowing clinicians to visually verify the neural network's decision-making process. The final model is deployed via a highly interactive Streamlit web application.

✨ Key Technical Features
Automated Skull Stripping: Custom OpenCV contour-detection pipeline to crop non-informative background space, maximizing tumor pixel density before resizing.

Advanced Transfer Learning: Utilizes DenseNet121 (fine-tuned by unfreezing the final 75 layers) to capture the subtle microscopic textures of intra-axial vs. extra-axial lesions.

Categorical Focal Loss: Implemented to mathematically penalize the model for confusing visually similar tumors (e.g., Glioma vs. Meningioma), overcoming standard cross-entropy limitations.

Class Weight Balancing: Dynamically penalizes the model for minority class misclassifications.

Explainable AI (XAI): Uses Gradient-weighted Class Activation Mapping (Grad-CAM) to highlight the specific spatial regions of the MRI that triggered the model's prediction.

📊 Dataset & Preprocessing
The model was trained on a 4-class MRI dataset. Preprocessing steps included:

Grayscale conversion and Gaussian Blurring.

Extreme-point contour detection to crop dead space (with a 15-pixel buffer to protect superficial meningeal boundaries).

Resizing to 224x224 pixels.

Data augmentation (Horizontal flipping exclusively, to preserve strict biological spatial locations like the pituitary gland).

🚀 Performance Metrics
The final fine-tuned DenseNet121 model achieved state-of-the-art results for 2D MRI classification:

Overall Accuracy: ~95%

Healthy Brain Detection: 100% Sensitivity (0 False Positives for 'No Tumor')

Focal Loss Success: Misclassification between Glioma and Meningioma was reduced by over 70% compared to the baseline cross-entropy model.

📦 Model Weights
Due to GitHub file size limitations, the trained model is hosted externally.

👉 Download Model: [https://drive.google.com/drive/folders/1phkCm78u090s7Otrjy2rOxgp5S9VpyNd?usp=drive_link](https://drive.google.com/drive/folders/1phkCm78u090s7Otrjy2rOxgp5S9VpyNd?usp=drive_link)

After downloading, place the model file inside:

saved_models/

📂 Repository Structure

├── app/
│   └── app.py                        # Streamlit web application frontend
├── datasets/
│   ├── 01_raw/                       # Original, uncropped datasets (Training/Testing)
│   └── 03_processed/                 # Cleaned, cropped, and resized 224x224 images
├── notebooks/
│   ├── 01_eda_and_visualization.ipynb    # Data distribution and visual analysis
│   ├── 02_data_preprocessing.ipynb       # Skull stripping and OpenCV pipelines
│   ├── 03_model_training_baseline.ipynb  # Custom baseline CNN architecture
│   ├── 04_model_training_advanced.ipynb  # DenseNet121 + Focal Loss fine-tuning
│   └── 05_evaluation_and_xai.ipynb       # Confusion matrices and Grad-CAM generation
├── saved_models/
│   └── advanced_densenet.keras       # Place downloaded model here
├── src/
│   ├── **init**.py
│   ├── data_loader.py                # Reusable OpenCV cropping functions
│   ├── metrics.py                    # Grad-CAM heatmap generation logic
│   └── model.py                      # Model loading utilities
└── README.md                         # Project documentation

⚙️ Installation & Setup

1. Clone the repository

git clone [https://github.com/yourusername/brain-tumor-detection.git](https://github.com/yourusername/brain-tumor-detection.git)
cd brain-tumor-detection

2. Create a virtual environment

conda create -n brain_env python=3.9
conda activate brain_env

3. Install dependencies

pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn streamlit pillow

💻 How to Run the Project

Option A: Run the Deep Learning Pipeline (Notebooks)
Execute notebooks sequentially from 01 to 05.

Option B: Run the Web Application

streamlit run app/app.py

App runs at: [http://localhost:8501](http://localhost:8501)

🔬 Future Work
Move from 2D MRI slices to 3D volumetric (NIfTI) data using 3D CNNs.

### Note to Evaluators

This repository represents the practical implementation of my Master's thesis research. For detailed methodology, refer to the thesis document.
