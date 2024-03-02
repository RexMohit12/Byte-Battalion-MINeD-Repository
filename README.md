# Byte-Battalion-MINeD-Repository
Readme File: Team Byte Battalion - Nirma University

**Introduction:**
This repository contains the code for a scientific image classifier developed by Team Byte Battalion from Nirma University. The classifier leverages Convolutional Neural Networks (CNNs) to accurately classify images across various scientific domains.

Installation:
1. Ensure you have Python installed on your system. You can download it from [Python's official website](https://www.python.org/).
2. Clone this repository to your local machine using the following command:
   
   git clone https://github.com/RexMohit12/Byte-Battalion-MINeD-Repository
   
3. Navigate to the project directory:
   
   cd project_directory
   
4. Install the required Python packages using pip:
   
   pip install -r requirements.txt
   

Running the Scripts:
1. Run the Flask API script (`flask_api.py`) to start the Flask server for handling image classification requests:
   
   python flask_api.py
   
   This script initializes the Flask application and exposes an endpoint (`/result`) to receive image classification requests.

2. Run the GUI script (`image_classifier_gui.py`) to launch the graphical user interface for selecting and classifying images:
   
   python image_classifier_gui.py
  
   This script opens a GUI window where you can select an image file. After selection, the script preprocesses the image, performs classification using the trained CNN model, and displays the predicted class label.

3. Additionally, the script for building the CNN model (`cnn_model.py`) is provided to demonstrate the architecture and layers of the model. You can run this script to train and evaluate the model or make modifications as needed.

Note: Ensure that the trained model file (`model.h5`) is present in the project directory before running the Flask API and GUI scripts. You can train your own model using the provided CNN architecture script (`cnn_model.py`) or replace `model.h5` with a pre-trained model.

For any issues or queries, please feel free to contact Team Byte Battalion from Nirma University.
