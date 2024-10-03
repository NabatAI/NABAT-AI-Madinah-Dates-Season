# NABAT-AI-Madinah-Dates-Season
This is an app that leverage using generative AI to minimize water consumption in agriculture
Nabat AI provides farmers and householders with precise irrigation strategies, optimizing water use while maintaining crop health.
The app uses a real-time data from weather forecasts, location, soil, insects, fertilizers, plant information to create a tailored solution that enhance sustainability and productivity. Through this cutting-edge technology and commitment to environmental stewardship, Nabat AI aims to revolutionize water managment in frarming.


# Tomato Image Classification

This repository contains code for a simple image classification model that uses a Convolutional Neural Network (CNN) to classify images of tomatoes.

## Requirements

- Python 3.x
- TensorFlow
- scikit-learn
- NumPy

You can install the required packages using pip:

```bash
pip install tensorflow scikit-learn numpy

Setup

Prepare a directory with images of tomatoes and set the path in train_model.py under the variable original_dir.
Ensure your images are in .jpg, .jpeg, or .png format.


The code is designed to work for training a CNN model to classify images of tomatoes, assuming the following conditions are met:

Correct Environment: Make sure you have the required libraries installed (TensorFlow, scikit-learn, NumPy).
Directory Structure: The original directory specified (original_dir) must contain images in supported formats (JPEG, PNG). The script will move these images into the training and testing directories, so it expects that the directory exists and contains images.
Image Content: The script assumes that all images in original_dir are of the same class (tomatoes). If you have multiple classes, you would need to adjust the code to accommodate those.
File Paths: Ensure the file paths in the code match your local setup. Modify them if necessary.
Resource Availability: Training a neural network can be resource-intensive. Make sure your machine has enough memory and processing power.
