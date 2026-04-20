import tensorflow as tf
import os

def load_trained_model(model_path):
    """Loads the trained Keras model from the disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    
    print("Loading model into memory...")
    # Using compile=False is safer when loading models across different environments
    model = tf.keras.models.load_model(model_path, compile=False)
    return model
