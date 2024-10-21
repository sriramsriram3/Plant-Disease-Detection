# Plant Disease Recognition System 🌿🔍

## Overview
The Plant Disease Recognition System is designed to help farmers, researchers, and agricultural professionals quickly identify plant diseases by analyzing images of crops. Our system uses state-of-the-art machine learning algorithms to detect diseases in plants, ensuring better crop management and healthier harvests.

## How It Works
1. **Upload Image**: Visit the [Disease Recognition](#) page and upload a picture of a plant showing signs of disease.
2. **Analysis**: Our advanced image recognition model will process the image and detect any plant disease present.
3. **Results**: You will receive results in seconds, along with actionable recommendations for disease management.

## Features
- **Accuracy**: Powered by advanced deep learning models, our system provides highly accurate results.
- **User-Friendly**: The system offers an intuitive interface, making it easy for anyone to upload images and receive results.
- **Fast and Efficient**: Get quick results to make timely decisions about your crops.

## Dataset
This project uses a dataset that was augmented from the original PlantVillage dataset, which consists of RGB images of healthy and diseased plant leaves categorized into 38 different classes. The dataset is split into training, validation, and test sets as follows:
- **Training Set**: 70,295 images
- **Validation Set**: 17,572 images
- **Test Set**: 33 images (used later for prediction)

### About the Dataset
- The dataset was created using offline augmentation to balance the classes and improve model training.
- The original dataset can be accessed via [GitHub Repository](#).
- Classes include various diseases affecting crops like apples, corn, grapes, and more.
# Plant Disease Recognition System 🌿🔍


## Project Structure
```plaintext
Plant-Disease-Recognition/
├── data/
│   ├── train/          # 70,295 training images categorized into 38 classes
│   ├── validation/     # 17,572 validation images
│   ├── test/           # 33 images for testing and prediction
├── models/             # Pre-trained and custom models for disease detection
├── src/                # Source code for the project
│   ├── preprocessing.py  # Code for data preprocessing and augmentation
│   ├── train.py          # Script to train the model
│   ├── predict.py        # Code for prediction
│   └── utils.py          # Utility functions for handling data and models
├── requirements.txt     # Python dependencies
├── app.py               # Main application file for the web interface
└── README.md            # Project documentation


## Prerequisites
Python 3.x
TensorFlow or PyTorch
Flask or Streamlit (for the web application)
OpenCV for image processing
Any required libraries can be installed using requirements.txt.

## installation

 pip install -r requirements.txt

# Running the Web Application
1. Start the web application :
'bash
python app.py
2. Open a web browser and go to http://127.0.0.1:5000 to access the Plant Disease Recognition System.

## Dataset Usage
Training the model: Use the train.py script to train a model on the dataset.
Prediction: Upload a test image using the web app or use the predict.py script for batch prediction.

## Why Choose Us?
Cutting-edge technology: We use advanced machine learning models for real-time disease detection.
Fast and accurate: Quickly detect plant diseases with high accuracy, helping in early intervention and prevention.
Scalability: Our system is designed to scale for use in large farms, research labs, or even mobile applications.

## Contribution
Feel free to contribute to the project by submitting pull requests or reporting issues. We welcome collaboration from the community.

## Lisence
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
The original dataset is credited to the PlantVillage project.
Special thanks to all contributors and researchers who made this project possible.
