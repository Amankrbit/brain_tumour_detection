import cv2
import numpy as np
import tensorflow as tf

def crop_brain_contour(image):
    """Finds the brain contour and crops out the dead black space."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
        
    c = max(contours, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    buffer = 15 # Increased buffer to protect tumor edges
    new_image = image[
        max(0, extTop[1] - buffer) : min(image.shape[0], extBot[1] + buffer),
        max(0, extLeft[0] - buffer) : min(image.shape[1], extRight[0] + buffer)
    ]
    return new_image

def preprocess_image_for_inference(image, target_size=(224, 224)):
    """Prepares a single image array for the neural network."""
    cropped_img = crop_brain_contour(image)
    resized_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_CUBIC)
    
    img_array = tf.keras.utils.img_to_array(resized_img)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, resized_img
