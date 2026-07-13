# Real-Time Facial Expression Recognition System

A Flask-based web application that detects human faces in real time and classifies
their emotional expression using a Convolutional Neural Network (CNN).

## Overview
Facial expressions are a key non-verbal communication signal. This project uses
deep learning and computer vision to automatically detect faces from a live webcam
feed or uploaded image and classify the expression into one of seven emotion
categories: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

## Features
- Real-time face detection using OpenCV Haar Cascade Classifier
- CNN-based emotion classification (trained on grayscale 48x48 face crops)
- Live webcam video stream with bounding boxes and predicted emotion labels
- Flask web interface with image upload prediction endpoint
- Data augmentation during training (rotation, zoom, flip) for better generalization

## Tech Stack
- Language: Python
- Deep Learning: TensorFlow / Keras (CNN)
- Computer Vision: OpenCV (Haar Cascade face detection)
- Web Framework: Flask
- Data Handling: NumPy, scikit-learn

## Model Architecture (CNN)
Conv2D(32) -> Conv2D(64) -> MaxPooling2D -> Dropout
-> Conv2D(128) -> MaxPooling2D -> Dropout
-> Conv2D(256) -> MaxPooling2D -> Dropout
-> Flatten -> Dense(256) -> Dropout -> Dense(7, softmax)

Optimizer: Adam | Loss: Categorical Crossentropy | Input: 48x48 grayscale images

## Project Structure
main.py - CNN model, training, Flask app, webcam inference
dataset/ - Training images organized by emotion folder
model/facial_expression_model.h5 - Saved trained model
templates/index.html - Flask front-end (webcam view)
README.md

## How It Works
1. OpenCV's Haar Cascade detects face regions in each video frame or uploaded image
2. Each detected face is cropped, resized to 48x48, and normalized
3. The trained CNN predicts probabilities across 7 emotion classes
4. The predicted emotion and confidence score are overlaid on the video feed

## Running the App
pip install -r requirements.txt
python main.py

Then open http://127.0.0.1:5000/ in your browser to view the live webcam feed,
or POST an image to /predict_image for single-image prediction.

