from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from PIL import Image
import io
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = load_model('image_classifier.py')

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input shape of the model
    img = image.resize((512, 512))
    # Convert image to numpy array
    img_array = np.asarray(img)
    # Normalize pixel values
    img_array = img_array / 255.0
    # Expand dimensions to create a batch of size 1
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define a function to handle the prediction
def predict_image():
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename()
    # Load image into PIL format
    img = Image.open(file_path)
    # Preprocess the image
    img_array = preprocess_image(img)
    # Perform prediction using the model
    prediction = model.predict(img_array)
    # Convert prediction to human-readable format
    predicted_class = np.argmax(prediction)
    # Display prediction result in a label
    result_label.config(text=f"Predicted class: {predicted_class}")

# Define GUI
root = tk.Tk()
root.title("Image Classifier")

# Create a button to select an image
select_button = tk.Button(root, text="Select Image", command=predict_image)
select_button.pack()

# Create a label to display prediction result
result_label = tk.Label(root, text="")
result_label.pack()

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, threaded=False)  # Disable threading for Tkinter compatibility
